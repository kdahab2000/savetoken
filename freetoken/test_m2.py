"""M2 tests: admission boundaries, localhost binding, streaming format,
cancellation, and real-model smoke.

Run from the workspace root:
    python3 -m unittest freetoken.test_m2 -v
"""

import json
import os
import socket
import threading
import time
import unittest
from http.client import HTTPConnection

from freetoken.capacity import (
    CACHE_BLOCK_TOKENS,
    CapacityConfig,
    FIXED_STATE_BYTES,
    KV_BYTES_PER_TOKEN,
    SAFETY_MARGIN_BYTES,
    active_cap,
    admit,
    estimate_peak_memory_bytes,
    estimate_prefill_seconds,
)
from freetoken.engine import Engine, GenerationCancelled
from freetoken.server import build_server, validate_host, workspace_root

MODEL = os.environ.get(
    "FREETOKEN_TEST_MODEL",
    os.path.join(workspace_root(), "m0_spike", "mlx_models",
                 "qwen3.5-healthcare-bf16"))

ENGINE = None
SERVER = None
PORT = None


def setUpModule():
    global ENGINE, SERVER, PORT
    ENGINE = Engine(MODEL, CapacityConfig(allow_extended=True))
    ENGINE.load()
    SERVER = build_server(ENGINE, "127.0.0.1", 0)  # ephemeral port
    PORT = SERVER.server_address[1]
    threading.Thread(target=SERVER.serve_forever, daemon=True).start()


def tearDownModule():
    SERVER.shutdown()
    SERVER.server_close()


def _conn():
    return HTTPConnection("127.0.0.1", PORT, timeout=120)


def _get(path):
    c = _conn()
    c.request("GET", path)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, json.loads(body)


def _post(path, obj):
    c = _conn()
    c.request("POST", path, body=json.dumps(obj),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, json.loads(body)


def _raw_stream(body_obj, stop_after_data_chunks=None, stop_when_contains=None):
    """Send a streaming request over a raw socket; return (socket, bytes read)."""
    body = json.dumps(body_obj).encode()
    s = socket.create_connection(("127.0.0.1", PORT), timeout=60)
    req = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: close\r\n\r\n" + body
    )
    s.sendall(req)
    data = b""
    while True:
        if stop_when_contains and stop_when_contains in data:
            break
        if stop_after_data_chunks and data.count(b"data: ") > stop_after_data_chunks:
            break
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    return s, data


SMALL_CHAT = [{"role": "user", "content": "The capital of France is"}]


class TestCapacity(unittest.TestCase):
    def test_validated_constants(self):
        cfg = CapacityConfig()
        self.assertEqual(cfg.kv_bytes_per_token, 12288)
        self.assertEqual(cfg.cache_block_tokens, 256)
        self.assertEqual(cfg.fixed_state_bytes, 19537920)
        self.assertEqual(cfg.safety_margin_bytes, 2 * 2**30)
        self.assertEqual(cfg.model_context_limit, 262144)

    def test_admission_boundaries(self):
        d = CapacityConfig()                    # extended off
        e = CapacityConfig(allow_extended=True)  # extended on
        cases = [
            # (cfg, input, max_new, allow_max, ok, code_or_mode)
            (d, 100, 65436, False, True, "default"),          # exactly 65536
            (d, 100, 65437, False, False, "default_cap_exceeded"),
            (e, 100, 65437, False, True, "extended"),         # just past default cap
            (e, 100, 130972, False, True, "extended"),        # exactly 131072
            (e, 100, 130973, False, False, "extended_cap_exceeded"),
            (e, 100, 130973, True, True, "maximum"),          # max-mode opt-in
            (e, 100, 262044, True, True, "maximum"),          # exactly 262144
            (e, 100, 262045, True, False, "context_limit_exceeded"),
            (d, 200000, 100000, True, False, "context_limit_exceeded"),
        ]
        for cfg, inp, mx, allow_max, ok, expect in cases:
            a = admit(cfg, inp, mx, allow_maximum_context=allow_max)
            self.assertEqual(a.total_tokens, inp + mx)
            self.assertEqual(a.ok, ok, msg=f"case {(inp, mx, allow_max)}")
            if ok:
                self.assertEqual(a.mode, expect)
            else:
                self.assertEqual(a.error_code, expect)

    def test_extended_warning_mentions_prefill(self):
        e = CapacityConfig(allow_extended=True)
        a = admit(e, 60000, 6000)  # extended tier
        self.assertTrue(a.ok)
        self.assertIn("prefill", a.warning)
        self.assertIsNotNone(a.estimated_prefill_seconds)
        m = admit(e, 100, 200000, allow_maximum_context=True)
        self.assertTrue(m.ok)
        self.assertIn("prefill", m.warning)
        self.assertIsNone(admit(e, 100, 100).warning)  # default tier: no warning

    def test_memory_estimate_uses_constants(self):
        cfg = CapacityConfig()
        n = 262144
        expect = (cfg.weights_resident_bytes + KV_BYTES_PER_TOKEN * n
                  + FIXED_STATE_BYTES + SAFETY_MARGIN_BYTES)
        self.assertEqual(estimate_peak_memory_bytes(cfg, n), expect)
        # block rounding: 1000 tokens allocates 1024 slots
        small = estimate_peak_memory_bytes(cfg, 1000) - (
            cfg.weights_resident_bytes + FIXED_STATE_BYTES + SAFETY_MARGIN_BYTES)
        self.assertEqual(small, KV_BYTES_PER_TOKEN * 4 * CACHE_BLOCK_TOKENS)

    def test_prefill_estimate(self):
        cfg = CapacityConfig()
        prev = -1.0
        for n in (1000, 8192, 50000, 131072, 262144):
            est = estimate_prefill_seconds(cfg, n)
            self.assertGreater(est, prev)
            prev = est
        self.assertAlmostEqual(estimate_prefill_seconds(cfg, 8192), 4.55)
        self.assertAlmostEqual(estimate_prefill_seconds(cfg, 262144), 757.8)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            CapacityConfig(default_cap=200000, extended_cap=100000).validate()


class TestServerConfig(unittest.TestCase):
    def test_rejects_non_loopback(self):
        for bad in ("0.0.0.0", "192.168.1.10", "8.8.8.8"):
            with self.assertRaises(ValueError):
                validate_host(bad)
            with self.assertRaises(ValueError):
                build_server(ENGINE, bad, 0)

    def test_shared_server_binds_loopback(self):
        self.assertEqual(SERVER.server_address[0], "127.0.0.1")
        self.assertEqual(active_cap(ENGINE.capacity), 131072)  # extended enabled


class TestHTTP(unittest.TestCase):
    def test_health_and_models(self):
        status, health = _get("/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["loaded"])
        status, models = _get("/v1/models")
        self.assertEqual(status, 200)
        m = models["data"][0]
        self.assertEqual(m["id"], "qwen3.5-healthcare-bf16")
        self.assertEqual(m["context_limit"], 262144)
        self.assertEqual(m["active_cap"], 131072)

    def test_unknown_route_and_bad_body(self):
        status, err = _get("/nope")
        self.assertEqual(status, 404)
        status, err = _post("/v1/chat/completions", {"messages": []})
        self.assertEqual(status, 400)
        self.assertEqual(err["error"]["code"], "invalid_messages")

    def test_chat_smoke_nonstreaming(self):
        status, body = _post("/v1/chat/completions",
                             {"messages": SMALL_CHAT, "max_tokens": 8})
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "chat.completion")
        content = body["choices"][0]["message"]["content"]
        self.assertGreater(len(content), 0)
        self.assertIn(body["choices"][0]["finish_reason"], ("stop", "length"))
        self.assertGreater(body["usage"]["prompt_tokens"], 0)
        self.assertEqual(body["usage"]["total_tokens"],
                         body["usage"]["prompt_tokens"]
                         + body["usage"]["completion_tokens"])

    def test_streaming_format(self):
        status_lines = []
        s, data = _raw_stream(
            {"messages": SMALL_CHAT, "max_tokens": 8, "stream": True})
        s.close()
        text = data.decode("utf-8", "replace")
        self.assertIn("HTTP/1.1 200", text.split("\r\n")[0])
        payload_lines = [l[len("data: "):] for l in text.splitlines()
                         if l.startswith("data: ")]
        self.assertGreater(len(payload_lines), 2)
        self.assertEqual(payload_lines[-1], "[DONE]")
        chunks = [json.loads(p) for p in payload_lines[:-1]]
        self.assertTrue(all(c["object"] == "chat.completion.chunk" for c in chunks))
        self.assertEqual(chunks[0]["choices"][0]["delta"].get("role"), "assistant")
        content = "".join(c["choices"][0]["delta"].get("content", "")
                          for c in chunks)
        self.assertGreater(len(content), 0)
        self.assertIn(chunks[-1]["choices"][0]["finish_reason"], ("stop", "length"))

    def test_reject_over_context_limit(self):
        status, err = _post("/v1/chat/completions",
                            {"messages": SMALL_CHAT, "max_tokens": 300000})
        self.assertEqual(status, 400)
        self.assertEqual(err["error"]["code"], "context_limit_exceeded")
        self.assertIn("262144", err["error"]["message"])

    def test_reject_extended_without_flag_and_max_without_optin(self):
        try:
            ENGINE.capacity.allow_extended = False
            status, err = _post("/v1/chat/completions",
                                {"messages": SMALL_CHAT, "max_tokens": 70000})
            self.assertEqual(status, 400)
            self.assertEqual(err["error"]["code"], "default_cap_exceeded")
        finally:
            ENGINE.capacity.allow_extended = True
        status, err = _post("/v1/chat/completions",
                            {"messages": SMALL_CHAT, "max_tokens": 140000})
        self.assertEqual(status, 400)
        self.assertEqual(err["error"]["code"], "extended_cap_exceeded")
        self.assertIn("allow_maximum_context", err["error"]["message"])

    def test_extended_request_carries_prefill_warning(self):
        # total ≈ 15 + 65600 > default cap → admitted in extended mode
        s, data = _raw_stream(
            {"messages": SMALL_CHAT, "max_tokens": 65600, "stream": True},
            stop_when_contains=b'"warning"')
        s.close()  # abort immediately; role chunk arrives before prefill
        text = data.decode("utf-8", "replace")
        self.assertIn('"warning"', text)
        self.assertIn("prefill", text)
        deadline = time.time() + 20
        cancelled_before = ENGINE.stats["generations_cancelled"]
        while time.time() < deadline:
            if ENGINE.stats["generations_cancelled"] > cancelled_before:
                break
            time.sleep(0.5)
        self.assertGreater(ENGINE.stats["generations_cancelled"], cancelled_before)

    def test_cancellation_on_client_disconnect(self):
        cancelled_before = ENGINE.stats["generations_cancelled"]
        tokens_before = ENGINE.stats["tokens_generated"]
        s, data = _raw_stream(
            {"messages": SMALL_CHAT, "max_tokens": 4096, "stream": True},
            stop_after_data_chunks=3)
        self.assertIn(b"data: ", data)
        s.close()
        deadline = time.time() + 20
        while time.time() < deadline:
            if ENGINE.stats["generations_cancelled"] > cancelled_before:
                break
            time.sleep(0.5)
        self.assertGreater(ENGINE.stats["generations_cancelled"], cancelled_before)
        tokens_at_cancel = ENGINE.stats["tokens_generated"]
        self.assertLess(tokens_at_cancel - tokens_before, 4096)
        time.sleep(1.0)
        self.assertEqual(ENGINE.stats["tokens_generated"], tokens_at_cancel)
        status, _ = _get("/health")  # server still healthy
        self.assertEqual(status, 200)


class TestEngine(unittest.TestCase):
    def test_engine_level_cancel(self):
        ids = ENGINE.tokenize_chat(SMALL_CHAT)
        self.assertGreater(len(ids), 0)
        ev = threading.Event()
        cancelled_before = ENGINE.stats["generations_cancelled"]
        gen = ENGINE.stream_chat(ids, 64, ev)
        seen = 0
        with self.assertRaises(GenerationCancelled):
            for _ in gen:
                seen += 1
                if seen == 2:
                    ev.set()
        self.assertGreaterEqual(seen, 2)
        self.assertEqual(ENGINE.stats["generations_cancelled"], cancelled_before + 1)

    def test_stream_produces_text(self):
        ids = ENGINE.tokenize_chat(SMALL_CHAT)
        gen = ENGINE.stream_chat(ids, 6)
        segs = [s for s in gen if s]
        self.assertGreater(len(segs), 0)


if __name__ == "__main__":
    unittest.main()
