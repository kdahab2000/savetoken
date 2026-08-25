"""M3 tests: manifest validation, exact checksum/size verification, mismatch
refusal, safe artifact discovery, prepare stages, listing/selection, explicit
model switching, and API smoke runs for both bf16 and 4-bit.

Run from the workspace root:
    python3 -m unittest freetoken.test_m3 -v
"""

import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from freetoken.capacity import CapacityConfig
from freetoken.engine import Engine
from freetoken.manifest import (MANIFEST_SCHEMA_VERSION, Manifest, ModelEntry,
                                ManifestError, dump_manifest, load_manifest,
                                safe_join)
from freetoken.server import (EngineState, ModelSwitchError, build_server)
from freetoken.store import (ModelStore, StoreError, finalize_download,
                             prepare, sha256_file)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, "freetoken", "models.manifest.json")
BF16_ID = "qwen3.5-healthcare-bf16"
Q4_ID = "qwen3.5-healthcare-4bit"

SERVER = None
PORT = None
STATE = None


def setUpModule():
    global SERVER, PORT, STATE
    store = ModelStore(ROOT, MANIFEST_PATH)
    entry = store.get(BF16_ID)
    engine = Engine(store.select(BF16_ID), CapacityConfig())
    engine.load()
    STATE = EngineState(engine, store=store, active_id=BF16_ID)
    SERVER = build_server(STATE, "127.0.0.1", 0)
    PORT = SERVER.server_address[1]
    threading.Thread(target=SERVER.serve_forever, daemon=True).start()


def tearDownModule():
    SERVER.shutdown()
    SERVER.server_close()


def _get(path):
    c = HTTPConnection("127.0.0.1", PORT, timeout=120)
    c.request("GET", path)
    r = c.getresponse()
    body = json.loads(r.read())
    c.close()
    return r.status, body


def _post(path, obj):
    c = HTTPConnection("127.0.0.1", PORT, timeout=600)
    c.request("POST", path, body=json.dumps(obj),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    body = json.loads(r.read())
    c.close()
    return r.status, body


def _entry(**over):
    base = dict(
        id="test-model", revision="r1", family="test", format="mlx",
        dtype="bfloat16", path="models/test", weights_file="model.safetensors",
        size_bytes=5, sha256="0" * 64, license="Apache-2.0",
        research_only=True, notes="test", context_limit=262144,
        expected_resident_mb=100,
    )
    base.update(over)
    return ModelEntry(**base)


class TestSafePath(unittest.TestCase):
    def test_traversal_refused(self):
        for bad in ("../evil", "a/../../b", "..", "a/../b", "/etc/passwd",
                    "~/.ssh", "a\x00b"):
            with self.assertRaises(ManifestError, msg=bad):
                safe_join(ROOT, bad)

    def test_symlink_escape_refused(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "models"))
            link = os.path.join(root, "models", "sneaky")
            os.symlink("/tmp", link)
            with self.assertRaises(ManifestError):
                safe_join(root, "models/sneaky/passwd")

    def test_safe_join_ok(self):
        p = safe_join(ROOT, "m0_spike/mlx_models/qwen3.5-healthcare-bf16")
        self.assertTrue(p.startswith(os.path.realpath(ROOT) + os.sep))


class TestManifest(unittest.TestCase):
    def test_real_manifest_loads(self):
        m = load_manifest(MANIFEST_PATH)
        self.assertEqual(m.schema_version, MANIFEST_SCHEMA_VERSION)
        self.assertEqual(m.ids(), [BF16_ID, Q4_ID])
        bf16 = m.get(BF16_ID)
        self.assertEqual(bf16.context_limit, 262144)
        self.assertTrue(bf16.research_only)
        self.assertEqual(bf16.format, "mlx")
        self.assertEqual(bf16.dtype, "bfloat16")
        self.assertIsNone(bf16.url)
        q4 = m.get(Q4_ID)
        self.assertEqual(q4.quant, {"bits": 4, "group_size": 32,
                                    "mode": "affine"})
        self.assertEqual(q4.prefill_curve, "q4-m1")
        self.assertLess(q4.expected_resident_mb, bf16.expected_resident_mb)

    def test_field_validation(self):
        cases = [
            dict(id="Bad ID!"),
            dict(revision=""),
            dict(format="gguf"),
            dict(dtype="int2"),
            dict(path="../escape"),
            dict(weights_file="/abs.safetensors"),
            dict(size_bytes=0),
            dict(sha256="xyz"),
            dict(sha256="A" * 64),
            dict(context_limit=0),
            dict(expected_resident_mb=-1),
            dict(prefill_curve="made-up"),
            dict(url="http://insecure.example/m"),
        ]
        for over in cases:
            with self.assertRaises(ManifestError, msg=str(over)):
                _entry(**over).validate()

    def test_schema_version_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            with open(p, "w") as f:
                json.dump({"schema_version": 99, "models": []}, f)
            with self.assertRaises(ManifestError):
                load_manifest(p)

    def test_duplicate_ids_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            e = _entry()
            dump_manifest(Manifest(MANIFEST_SCHEMA_VERSION, [e, e]), p)
            with self.assertRaises(ManifestError):
                load_manifest(p)

    def test_dump_refuses_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            m = Manifest(MANIFEST_SCHEMA_VERSION, [_entry()])
            dump_manifest(m, p)
            with self.assertRaises(ManifestError):
                dump_manifest(m, p)
            dump_manifest(m, p, force=True)  # explicit force is allowed


class TestVerification(unittest.TestCase):
    def setUp(self):
        self.store = ModelStore(ROOT, MANIFEST_PATH)

    def test_verify_real_artifacts_exact(self):
        for model_id in (BF16_ID, Q4_ID):
            entry = self.store.get(model_id)
            vr = self.store.verify(model_id)
            self.assertTrue(vr.ok, msg=model_id)
            self.assertEqual(vr.actual_size, entry.size_bytes)
            self.assertEqual(vr.actual_sha256, entry.sha256)

    def test_verify_unknown_model(self):
        with self.assertRaises(StoreError):
            self.store.verify("no-such-model")

    def test_mismatch_refused_and_artifact_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "models", "test"))
            wpath = os.path.join(root, "models", "test", "model.safetensors")
            with open(wpath, "wb") as f:
                f.write(b"hello")
            entry = _entry(size_bytes=999, sha256="f" * 64)
            mpath = os.path.join(root, "manifest.json")
            dump_manifest(Manifest(MANIFEST_SCHEMA_VERSION, [entry]), mpath)
            store = ModelStore(root, mpath)
            vr = store.verify("test-model")
            self.assertFalse(vr.ok)
            self.assertTrue(vr.exists)
            self.assertFalse(vr.size_ok)
            self.assertFalse(vr.sha256_ok)
            self.assertEqual(vr.actual_size, 5)
            self.assertEqual(vr.actual_sha256, sha256_file(wpath))
            with open(wpath, "rb") as f:
                self.assertEqual(f.read(), b"hello")  # untouched

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as root:
            entry = _entry()
            mpath = os.path.join(root, "manifest.json")
            dump_manifest(Manifest(MANIFEST_SCHEMA_VERSION, [entry]), mpath)
            store = ModelStore(root, mpath)
            vr = store.verify("test-model")
            self.assertFalse(vr.ok)
            self.assertFalse(vr.exists)


class TestFinalizeDownload(unittest.TestCase):
    def test_atomic_promote_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            partial = os.path.join(d, "w.partial")
            dest = os.path.join(d, "w")
            content = b"payload-bytes"
            with open(partial, "wb") as f:
                f.write(content)
            sha = sha256_file(partial)
            finalize_download(partial, dest, len(content), sha)
            self.assertTrue(os.path.isfile(dest))
            self.assertFalse(os.path.exists(partial))
            # refused overwrite of an existing artifact
            with open(partial, "wb") as f:
                f.write(content)
            with self.assertRaises(StoreError) as ctx:
                finalize_download(partial, dest, len(content), sha)
            self.assertEqual(ctx.exception.code, "exists")

    def test_bad_checksum_never_promoted(self):
        with tempfile.TemporaryDirectory() as d:
            partial = os.path.join(d, "w.partial")
            dest = os.path.join(d, "w")
            with open(partial, "wb") as f:
                f.write(b"payload-bytes")
            with self.assertRaises(StoreError) as ctx:
                finalize_download(partial, dest, 13, "0" * 64)
            self.assertEqual(ctx.exception.code, "checksum_mismatch")
            self.assertFalse(os.path.exists(dest))
            self.assertTrue(os.path.exists(partial))  # evidence retained
            with self.assertRaises(StoreError) as ctx:
                finalize_download(partial, dest, 999, sha256_file(partial))
            self.assertEqual(ctx.exception.code, "size_mismatch")


class TestPrepare(unittest.TestCase):
    def setUp(self):
        self.store = ModelStore(ROOT, MANIFEST_PATH)

    def test_prepare_local_mlx_model(self):
        for model_id in (BF16_ID, Q4_ID):
            report = prepare(self.store, model_id)
            self.assertEqual(report["stages"]["locate"], "found")
            self.assertEqual(report["stages"]["download"], "skipped")
            self.assertEqual(report["stages"]["verify"], "ok")
            self.assertEqual(report["stages"]["convert"], "skipped")
            self.assertEqual(report["stages"]["finalize"], "ok")
            self.assertEqual(report["path"], self.store.select(model_id))

    def test_prepare_unknown_model(self):
        with self.assertRaises(StoreError) as ctx:
            prepare(self.store, "nope")
        self.assertEqual(ctx.exception.code, "unknown_model")

    def test_missing_entry_without_url_refuses_download(self):
        with tempfile.TemporaryDirectory() as root:
            entry = _entry()  # path does not exist, url=None
            mpath = os.path.join(root, "manifest.json")
            dump_manifest(Manifest(MANIFEST_SCHEMA_VERSION, [entry]), mpath)
            store = ModelStore(root, mpath)
            with self.assertRaises(StoreError) as ctx:
                prepare(store, "test-model")
            self.assertEqual(ctx.exception.code, "not_available")
            with self.assertRaises(StoreError) as ctx:
                prepare(store, "test-model", allow_download=True)
            self.assertEqual(ctx.exception.code, "not_available")

    def test_missing_entry_with_url_requires_explicit_permission(self):
        with tempfile.TemporaryDirectory() as root:
            entry = _entry(url="https://example.invalid/w.safetensors")
            mpath = os.path.join(root, "manifest.json")
            dump_manifest(Manifest(MANIFEST_SCHEMA_VERSION, [entry]), mpath)
            store = ModelStore(root, mpath)
            with self.assertRaises(StoreError) as ctx:
                prepare(store, "test-model")  # allow_download=False
            self.assertEqual(ctx.exception.code, "download_disabled")


class TestSwitchRefusalWithoutServer(unittest.TestCase):
    def test_switch_refuses_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "models", "test"))
            with open(os.path.join(root, "models", "test",
                                   "model.safetensors"), "wb") as f:
                f.write(b"tampered-or-just-different")
            entry = _entry(size_bytes=26, sha256="0" * 64)
            mpath = os.path.join(root, "manifest.json")
            dump_manifest(Manifest(MANIFEST_SCHEMA_VERSION, [entry]), mpath)
            store = ModelStore(root, mpath)
            dummy = Engine(os.path.join(root, "models", "test"))
            state = EngineState(dummy, store=store, active_id="other-model")
            with self.assertRaises(ModelSwitchError) as ctx:
                state.switch("test-model")
            self.assertEqual(ctx.exception.code, "checksum_mismatch")
            self.assertIs(state.engine, dummy)  # nothing was replaced

    def test_switch_requires_manifest(self):
        dummy = Engine("/nonexistent/path")
        state = EngineState(dummy, store=None)
        with self.assertRaises(ModelSwitchError) as ctx:
            state.switch("anything")
        self.assertEqual(ctx.exception.code, "no_manifest")


class TestServerM3(unittest.TestCase):
    def test_models_endpoint_lists_catalog(self):
        status, body = _get("/v1/models")
        self.assertEqual(status, 200)
        ids = [d["id"] for d in body["data"]]
        self.assertEqual(set(ids), {BF16_ID, Q4_ID})
        active = body["data"][0]
        self.assertEqual(active["id"], STATE.active_id)
        self.assertTrue(active["active"])
        self.assertEqual(active["revision"], "m0-conv-20260824"
                         if active["id"] == BF16_ID else "m1-quant-20260824")
        self.assertEqual(active["format"], "mlx")
        self.assertEqual(active["context_limit"], 262144)
        self.assertGreater(active["weights_resident_gb"], 0.5)
        self.assertIn("active_cap", active)
        other = body["data"][1]
        self.assertFalse(other["active"])
        self.assertEqual(other["expected_resident_mb"], 1177
                         if other["id"] == Q4_ID else 3764)

    def test_chat_smoke_bf16(self):
        status, body = _post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_tokens": 8})
        self.assertEqual(status, 200)
        self.assertGreater(len(body["choices"][0]["message"]["content"]), 0)

    def test_switch_rejects_unknown_and_same(self):
        status, err = _post("/v1/models/switch", {"model": "no-such-model"})
        self.assertEqual(status, 400)
        self.assertEqual(err["error"]["code"], "unknown_model")
        status, err = _post("/v1/models/switch", {"model": STATE.active_id})
        self.assertEqual(status, 400)
        self.assertEqual(err["error"]["code"], "already_active")

    def test_switch_to_q4_smoke_and_back(self):
        start = STATE.active_id
        target = Q4_ID if start == BF16_ID else BF16_ID
        status, info = _post("/v1/models/switch", {"model": target})
        self.assertEqual(status, 200, msg=info)
        self.assertTrue(info["switched"])
        self.assertEqual(info["previous"], start)
        self.assertEqual(STATE.active_id, target)
        status, health = _get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["active_model_id"], target)
        self.assertGreater(health["weights_resident_gb"], 0.5)
        # catalog order: active model first
        status, body = _get("/v1/models")
        self.assertEqual(body["data"][0]["id"], target)
        # generation smoke on the switched model
        status, chat = _post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_tokens": 8})
        self.assertEqual(status, 200)
        self.assertEqual(chat["model"], target)
        self.assertGreater(len(chat["choices"][0]["message"]["content"]), 0)
        # switch back: explicit again, never silent
        status, info = _post("/v1/models/switch", {"model": start})
        self.assertEqual(status, 200)
        self.assertEqual(STATE.active_id, start)


if __name__ == "__main__":
    unittest.main()
