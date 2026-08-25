"""M5 tests: release metadata/package validation, dist artifact checks,
signing/notarization preflight behavior, and the remote-model-code policy.

Run from the workspace root:
    python3 -m unittest freetoken.test_m5 -v
"""

import hashlib
import importlib.util
import json
import os
import plistlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(ROOT, "FreeTokenApp")
TOOLS = os.path.join(APP_DIR, "tools")
DIST = os.path.join(APP_DIR, "dist")


def _load_make_release():
    spec = importlib.util.spec_from_file_location(
        "make_release", os.path.join(TOOLS, "make_release.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mr = _load_make_release()


def _run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, cwd=APP_DIR,
                          timeout=300, **kw)


class TestReleaseMetadata(unittest.TestCase):
    def test_info_plist_stable_and_consistent(self):
        p1 = mr.info_plist()
        p2 = mr.info_plist()
        self.assertEqual(p1, p2)  # deterministic
        self.assertEqual(p1["CFBundleIdentifier"], "local.savetoken.app")
        self.assertEqual(p1["CFBundleExecutable"], "SaveToken")
        self.assertEqual(p1["LSMinimumSystemVersion"], "13.0")  # == Package.swift .v13
        self.assertEqual(p1["CFBundleShortVersionString"], mr.APP_VERSION)
        self.assertEqual(p1["CFBundlePackageType"], "APPL")

    def test_package_swift_platform_matches_plist(self):
        pkg = open(os.path.join(APP_DIR, "Package.swift")).read()
        self.assertIn(".macOS(.v13)", pkg)
        self.assertEqual(mr.MIN_MACOS, "13.0")

    def test_assemble_dist_metadata_and_checksums(self):
        with tempfile.TemporaryDirectory() as d:
            binpath = os.path.join(d, "FreeTokenApp")
            payload = b"fake-mach-o-payload" * 100
            with open(binpath, "wb") as f:
                f.write(payload)
            dist = os.path.join(d, "dist")
            m1 = mr.assemble_dist(binpath, dist)
            # executable hash is real
            self.assertEqual(
                m1["executable"]["sha256"],
                hashlib.sha256(payload).hexdigest())
            self.assertEqual(m1["executable"]["size"], len(payload))
            self.assertEqual(m1["bundle_id"], "local.savetoken.app")
            # plist written into the bundle and parseable
            app_root = os.path.join(dist, "SaveToken.app")
            with open(os.path.join(app_root, "Contents", "Info.plist"), "rb") as f:
                plist = plistlib.load(f)
            self.assertEqual(plist["CFBundleIdentifier"], "local.savetoken.app")
            self.assertEqual(plist["LSMinimumSystemVersion"], "13.0")
            # sha256 sidecar matches manifest
            sha_line = open(os.path.join(dist, "SaveToken.sha256")).read()
            self.assertTrue(sha_line.startswith(m1["executable"]["sha256"]))
            # file inventory complete and hashed
            rel = "SaveToken.app/Contents/MacOS/SaveToken"
            self.assertIn(rel, m1["files"])
            self.assertTrue(m1["files"][rel]["executable"])
            # deterministic: second assembly produces identical manifest JSON
            manifest_path = os.path.join(dist, "RELEASE_MANIFEST.json")
            first = open(manifest_path).read()
            mr.assemble_dist(binpath, dist)
            self.assertEqual(first, open(manifest_path).read())
            # models are never bundled
            self.assertNotIn("safetensors", json.dumps(m1))
            self.assertTrue(m1["models"].startswith("NOT bundled"))

    def test_missing_binary_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                mr.assemble_dist(os.path.join(d, "nope"), os.path.join(d, "x"))


@unittest.skipUnless(os.path.isdir(DIST), "dist not built yet (run package_app.sh)")
class TestBuiltDistArtifact(unittest.TestCase):
    def test_dist_manifest_matches_actual_files(self):
        manifest = json.load(open(os.path.join(DIST, "RELEASE_MANIFEST.json")))
        exe = manifest["executable"]
        exe_path = os.path.join(DIST, exe["path"])
        self.assertTrue(os.path.isfile(exe_path))
        self.assertEqual(os.path.getsize(exe_path), exe["size"])
        h = hashlib.sha256()
        with open(exe_path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        self.assertEqual(h.hexdigest(), exe["sha256"])
        for rel, meta in manifest["files"].items():
            full = os.path.join(DIST, rel)
            self.assertTrue(os.path.isfile(full), rel)
            self.assertEqual(os.path.getsize(full), meta["size"], rel)
        # no model weights inside the bundle
        for rel in manifest["files"]:
            self.assertNotIn("safetensors", rel)

    def test_dist_plist_metadata(self):
        with open(os.path.join(DIST, "SaveToken.app", "Contents",
                               "Info.plist"), "rb") as f:
            plist = plistlib.load(f)
        self.assertEqual(plist["CFBundleIdentifier"], "local.savetoken.app")
        self.assertEqual(plist["LSMinimumSystemVersion"], "13.0")


class TestSigningPreflight(unittest.TestCase):
    def test_check_reports_available_state(self):
        r = _run(["sh", "tools/sign_app.sh", "check"])
        self.assertEqual(r.returncode, 0, r.stderr)
        ids = subprocess.run(
            ["security", "find-identity", "-v", "-p", "codesigning"],
            capture_output=True, text=True).stdout
        has_dev_id = "Developer ID Application:" in ids
        if has_dev_id:
            self.assertIn("Developer ID available", r.stdout)
        else:
            self.assertIn("NOT FOUND", r.stdout)
            self.assertIn("BLOCKED", r.stdout)          # honest blocked report
            self.assertIn("ad-hoc", r.stdout)            # development path offered
            self.assertNotIn("notarized", r.stdout.lower().replace("notarization", ""))

    def test_ad_hoc_signs_a_bundle_and_verifies(self):
        with tempfile.TemporaryDirectory() as d:
            macos = os.path.join(d, "Fake.app", "Contents", "MacOS")
            os.makedirs(macos)
            exe = os.path.join(macos, "Fake")
            with open(exe, "wb") as f:
                f.write(b"#!/bin/sh\nexit 0\n")
            os.chmod(exe, 0o755)
            with open(os.path.join(d, "Fake.app", "Contents", "Info.plist"), "wb") as f:
                plistlib.dump({"CFBundleIdentifier": "local.fake",
                               "CFBundleExecutable": "Fake"}, f)
            r = _run(["sh", "tools/sign_app.sh", "ad-hoc",
                      "--app", os.path.join(d, "Fake.app")])
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("AD-HOC signing complete", r.stdout)
            v = subprocess.run(
                ["codesign", "--verify", "--deep", "--strict",
                 os.path.join(d, "Fake.app")],
                capture_output=True, text=True)
            self.assertEqual(v.returncode, 0, v.stderr)
            dv = subprocess.run(
                ["codesign", "-dv", os.path.join(d, "Fake.app")],
                capture_output=True, text=True).stderr
            self.assertIn("Signature=adhoc", dv)

    def test_notarize_preflight_without_profile_is_blocked(self):
        r = _run(["sh", "tools/sign_app.sh", "notarize-preflight",
                  "--profile", "definitely-not-a-real-profile-xyz"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("BLOCKED", r.stdout)
        self.assertIn("Nothing was submitted", r.stdout)


class TestRemoteModelCodePolicy(unittest.TestCase):
    def test_no_trust_remote_code_anywhere(self):
        """mlx-lm custom remote code must never be enabled by this project."""
        needle = "trust_remote_code" + "=True"  # split so this test isn't self-matched
        offenders = []
        for dirpath, _, files in os.walk(os.path.join(ROOT, "freetoken")):
            if "__pycache__" in dirpath:
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                if os.path.abspath(path) == os.path.abspath(__file__):
                    continue
                with open(path) as f:
                    if needle.replace(" ", "") in f.read().replace(" ", ""):
                        offenders.append(path)
        self.assertEqual(offenders, [])

    def test_server_binds_loopback_only(self):
        from freetoken.server import validate_host
        with self.assertRaises(ValueError):
            validate_host("0.0.0.0")
        self.assertEqual(validate_host("127.0.0.1"), "127.0.0.1")


class TestPortability(unittest.TestCase):
    """PUBLIC_RELEASE_PLAN §1: no maintainer-specific absolute paths may
    appear in production code, tests, or public docs."""

    SCAN_DIRS = [
        os.path.join(ROOT, "freetoken"),
        os.path.join(APP_DIR, "Sources"),
        os.path.join(APP_DIR, "Tests"),
    ]
    SCAN_FILES = [
        os.path.join(ROOT, "freetoken", "README.md"),
        os.path.join(APP_DIR, "README.md"),
    ]
    EXTENSIONS = (".py", ".swift", ".sh")

    def test_no_maintainer_paths(self):
        needle = "/Users/" + "khaled"  # split so this test never self-matches
        offenders = []
        for base in self.SCAN_DIRS:
            for dirpath, dirnames, files in os.walk(base):
                dirnames[:] = [d for d in dirnames
                               if d not in ("__pycache__", ".build")]
                for name in files:
                    if not name.endswith(self.EXTENSIONS):
                        continue
                    path = os.path.join(dirpath, name)
                    if os.path.abspath(path) == os.path.abspath(__file__):
                        continue
                    with open(path) as f:
                        if needle in f.read():
                            offenders.append(path)
        for path in self.SCAN_FILES:
            with open(path) as f:
                if needle in f.read():
                    offenders.append(path)
        self.assertEqual(offenders, [])

    def test_release_manifest_has_no_machine_paths(self):
        manifest_path = os.path.join(DIST, "RELEASE_MANIFEST.json")
        if not os.path.isfile(manifest_path):
            self.skipTest("dist not built yet")
        with open(manifest_path) as f:
            self.assertNotIn("/Users/", f.read())


if __name__ == "__main__":
    unittest.main()
