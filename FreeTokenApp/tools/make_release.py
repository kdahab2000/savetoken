#!/usr/bin/env python3
"""Deterministic release packaging for SaveToken.app.

Produces dist/SaveToken.app + dist/RELEASE_MANIFEST.json + a SHA-256 file.
No timestamps are embedded: identical inputs produce identical metadata.
Only the built binary is packaged — never a model artifact (multi-GB weights
stay in m0_spike/ and m1_calibration/, referenced at runtime by the loopback
server only).
"""

import argparse
import hashlib
import json
import os
import plistlib
import shutil
import sys
import subprocess

APP_NAME = "SaveToken.app"
EXECUTABLE_NAME = "SaveToken"
APP_VERSION = "0.5.0"
BUNDLE_VERSION = "1"
BUNDLE_ID = "local.savetoken.app"
BUNDLE_NAME = "SaveToken"
MIN_MACOS = "13.0"
ICON_NAME = "SaveToken.icns"
ICON_SOURCE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Resources",
    ICON_NAME,
)
_HASH_CHUNK = 8 * 1024 * 1024


def info_plist() -> dict:
    """Stable, minimal, hardened-runtime-ready Info.plist contents."""
    return {
        "CFBundleExecutable": EXECUTABLE_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleName": BUNDLE_NAME,
        "CFBundleDisplayName": BUNDLE_NAME,
        "CFBundleIconFile": ICON_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": BUNDLE_VERSION,
        "LSMinimumSystemVersion": MIN_MACOS,
        "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
        # Loopback HTTP is permitted by ATS default; this keeps behavior
        # explicit. No non-local hosts are ever contacted.
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
    }


def write_info_plist(app_root: str) -> str:
    path = os.path.join(app_root, "Contents", "Info.plist")
    with open(path, "wb") as f:
        plistlib.dump(info_plist(), f, sort_keys=True)
    return path


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def bundle_file_hashes(app_root: str) -> dict:
    """Relative path -> {size, sha256, executable} for every file in the bundle."""
    out = {}
    app_root = os.path.realpath(app_root)
    for dirpath, dirnames, filenames in os.walk(app_root):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, os.path.dirname(app_root))
            out[rel] = {
                "size": os.path.getsize(full),
                "sha256": sha256_file(full),
                "executable": os.access(full, os.X_OK),
            }
    return out


def signing_status(app_root: str) -> str:
    """Report the bundle's current signing state without exposing secrets."""
    try:
        result = subprocess.run(
            ["codesign", "-dv", "--verbose=2", app_root],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown (codesign unavailable)"
    details = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        return "unsigned (run tools/sign_app.sh)"
    if "adhoc" in details.lower() or "Signature=adhoc" in details:
        return "ad-hoc signed (hardened runtime)"
    if "Developer ID Application:" in details:
        return "Developer ID signed"
    return "signed (codesign)"


def _write_metadata(dist_dir: str) -> dict:
    """Compute manifest + sha256 sidecar from whatever bundle is in dist_dir."""
    app_root = os.path.join(dist_dir, APP_NAME)
    dest = os.path.join(app_root, "Contents", "MacOS", EXECUTABLE_NAME)
    if not os.path.isfile(dest):
        raise FileNotFoundError(f"bundle executable not found: {dest}")
    manifest = {
        "app": APP_NAME,
        "version": APP_VERSION,
        "bundle_version": BUNDLE_VERSION,
        "bundle_id": BUNDLE_ID,
        "min_macos": MIN_MACOS,
        "executable": {
            "path": f"{APP_NAME}/Contents/MacOS/{EXECUTABLE_NAME}",
            "size": os.path.getsize(dest),
            "sha256": sha256_file(dest),
        },
        "files": bundle_file_hashes(app_root),
        "signing": signing_status(app_root),
        "models": (
            "NOT bundled: managed by Ollama or separately provisioned "
            "for the optional SaveToken MLX server"
        ),
        "notes": [
            "SaveToken HTTP endpoints are loopback-only: Ollama on "
            "127.0.0.1:11434 or SaveToken MLX on 127.0.0.1:8321.",
            "Ollama cloud-tagged models may use remote inference through "
            "the local Ollama daemon; the UI labels them CLOUD.",
            "No SaveToken telemetry or prompt logging; research and "
            "development only, not for clinical use.",
        ],
    }
    manifest_path = os.path.join(dist_dir, "RELEASE_MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    sha_path = os.path.join(dist_dir, f"{EXECUTABLE_NAME}.sha256")
    with open(sha_path, "w") as f:
        f.write(f"{manifest['executable']['sha256']}  "
                f"{APP_NAME}/Contents/MacOS/{EXECUTABLE_NAME}\n")
    return manifest


def assemble_dist(bin_path: str, dist_dir: str) -> dict:
    """Copy the release binary into dist/SaveToken.app and write metadata."""
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"release binary not found: {bin_path}")
    os.makedirs(dist_dir, exist_ok=True)
    app_root = os.path.join(dist_dir, APP_NAME)
    if os.path.exists(app_root):
        shutil.rmtree(app_root)
    macos_dir = os.path.join(app_root, "Contents", "MacOS")
    os.makedirs(macos_dir)
    resources_dir = os.path.join(app_root, "Contents", "Resources")
    os.makedirs(resources_dir)
    dest = os.path.join(macos_dir, EXECUTABLE_NAME)
    shutil.copyfile(bin_path, dest)
    os.chmod(dest, 0o755)
    if not os.path.isfile(ICON_SOURCE):
        raise FileNotFoundError(f"app icon not found: {ICON_SOURCE}")
    shutil.copyfile(ICON_SOURCE, os.path.join(resources_dir, ICON_NAME))
    write_info_plist(app_root)
    return _write_metadata(dist_dir)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", help="path to the built release binary")
    p.add_argument("--dist", required=True, help="output dist directory")
    p.add_argument("--refresh", action="store_true",
                   help="recompute metadata for an existing dist bundle "
                        "(use after signing)")
    args = p.parse_args(argv)
    if args.refresh:
        manifest = _write_metadata(args.dist)
    else:
        if not args.bin:
            p.error("--bin is required unless --refresh is given")
        manifest = assemble_dist(args.bin, args.dist)
    print(json.dumps({
        "app": os.path.join(args.dist, APP_NAME),
        "version": manifest["version"],
        "bundle_id": manifest["bundle_id"],
        "sha256": manifest["executable"]["sha256"],
        "manifest": os.path.join(args.dist, "RELEASE_MANIFEST.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
