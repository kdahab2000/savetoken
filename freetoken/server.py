"""Localhost-only OpenAI-compatible HTTP API over the MLX engine.

Routes:
  GET  /health                liveness + active model status
  GET  /v1/models             manifest models (ids, revisions, formats,
                              context limits) + active model details
  POST /v1/chat/completions   OpenAI-compatible; stream=true uses SSE
  POST /v1/models/switch      explicit model switch (never implicit)

Privacy: binds only to a loopback address; never logs prompt/response content.
"""

import argparse
import gc
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx

from .capacity import (CapacityConfig, PREFILL_CURVES, active_cap, admit,
                       estimate_peak_memory_bytes)
from .engine import Engine, GenerationCancelled
from .manifest import ManifestError
from .store import ModelStore, StoreError

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
MAX_BODY_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MODEL_ID = "qwen3.5-healthcare-bf16"


def validate_host(host: str) -> str:
    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"refusing to bind non-loopback host {host!r}; this server is "
            "localhost-only by design"
        )
    return host


class ModelSwitchError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class EngineState:
    """Holds the live engine. A switch is explicit, verified, and atomic."""

    def __init__(self, engine: Engine, store=None, active_id: str = None):
        self.engine = engine
        self.store = store
        self.active_id = active_id or engine.model_id
        self._switch_lock = threading.Lock()

    def switch(self, model_id: str) -> dict:
        with self._switch_lock:
            if self.store is None:
                raise ModelSwitchError(
                    "no_manifest",
                    "server was started with a direct model path; restart with "
                    "a manifest model id to enable switching",
                )
            if model_id == self.active_id:
                raise ModelSwitchError(
                    "already_active", f"model {model_id!r} is already active")
            try:
                entry = self.store.get(model_id)
                vr = self.store.verify(model_id)
            except StoreError as e:
                raise ModelSwitchError(e.code, e.message)
            if not vr.ok:
                code = "missing" if not vr.exists else (
                    "size_mismatch" if not vr.size_ok else "checksum_mismatch")
                raise ModelSwitchError(
                    code,
                    f"refusing to load {model_id!r}: verification failed "
                    f"(exists={vr.exists}, size_ok={vr.size_ok}, "
                    f"sha256_ok={vr.sha256_ok})",
                )
            prev = self.active_id
            path = self.store.select(model_id)
            cur = self.engine.capacity
            new_capacity = CapacityConfig(
                default_cap=cur.default_cap,
                extended_cap=cur.extended_cap,
                allow_extended=cur.allow_extended,
                prefill_curve=PREFILL_CURVES[entry.prefill_curve],
            )
            new_engine = Engine(path, new_capacity)
            new_engine.load(measure="delta")  # old model still resident
            old = self.engine
            self.engine = new_engine
            self.active_id = model_id
            del old
            gc.collect()
            mx.clear_cache()
            return {
                "switched": True,
                "model": model_id,
                "previous": prev,
                "revision": entry.revision,
                "weights_resident_gb": round(
                    new_engine.capacity.weights_resident_bytes / 1e9, 2),
            }


def make_handler(state: EngineState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "freetoken/0.2"
        protocol_version = "HTTP/1.1"

        # ---- plumbing -------------------------------------------------
        def log_message(self, fmt, *args):  # method/path/status only — never bodies
            sys.stderr.write("%s\n" % (fmt % args))

        def _send_json(self, status: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: int, code: str, message: str, **extra) -> None:
            err = {"message": message, "type": "invalid_request_error", "code": code}
            err.update(extra)
            self._send_json(status, {"error": err})

        def _read_body_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("missing or oversized request body")
            return json.loads(self.rfile.read(length))

        # ---- routes ---------------------------------------------------
        def do_GET(self):
            engine = self.server.state.engine
            if self.path == "/health":
                self._send_json(200, {
                    "status": "ok",
                    "model": engine.model_id,
                    "active_model_id": self.server.state.active_id,
                    "loaded": engine.model is not None,
                    "weights_resident_gb": round(
                        engine.capacity.weights_resident_bytes / 1e9, 2),
                })
            elif self.path == "/v1/models":
                self._models_list()
            else:
                self._send_error(404, "not_found", f"no route for {self.path}")

        def _models_list(self):
            engine = self.server.state.engine
            store = self.server.state.store
            cfg = engine.capacity
            data = []
            if store is not None:
                for e in store.manifest.models:
                    item = {
                        "id": e.id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "local",
                        "revision": e.revision,
                        "format": e.format,
                        "dtype": e.dtype,
                        "context_limit": e.context_limit,
                        "expected_resident_mb": e.expected_resident_mb,
                        "research_only": e.research_only,
                        "active": e.id == self.server.state.active_id,
                    }
                    if item["active"]:
                        item.update({
                            "default_cap": cfg.default_cap,
                            "extended_cap": cfg.extended_cap,
                            "allow_extended": cfg.allow_extended,
                            "active_cap": active_cap(cfg),
                            "weights_resident_gb": round(
                                cfg.weights_resident_bytes / 1e9, 2),
                        })
                    data.append(item)
                data.sort(key=lambda d: not d["active"])  # active first
            else:
                data.append({
                    "id": engine.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "context_limit": cfg.model_context_limit,
                    "default_cap": cfg.default_cap,
                    "extended_cap": cfg.extended_cap,
                    "allow_extended": cfg.allow_extended,
                    "active_cap": active_cap(cfg),
                    "active": True,
                })
            self._send_json(200, {"object": "list", "data": data})

        def do_POST(self):
            if self.path == "/v1/chat/completions":
                try:
                    req = self._read_body_json()
                except (ValueError, json.JSONDecodeError) as e:
                    self._send_error(400, "invalid_json", f"malformed request: {e}")
                    return
                try:
                    self._chat_completions(req)
                except Exception as e:  # never leak a stack trace to the client
                    self._send_error(500, "internal_error", f"generation failed: {e}")
            elif self.path == "/v1/models/switch":
                try:
                    req = self._read_body_json()
                except (ValueError, json.JSONDecodeError) as e:
                    self._send_error(400, "invalid_json", f"malformed request: {e}")
                    return
                self._switch(req)
            else:
                self._send_error(404, "not_found", f"no route for {self.path}")

        def _switch(self, req: dict):
            model_id = req.get("model")
            if not isinstance(model_id, str) or not model_id:
                self._send_error(400, "invalid_model",
                                 'body must contain {"model": "<manifest id>"}')
                return
            try:
                info = self.server.state.switch(model_id)
            except ModelSwitchError as e:
                self._send_error(e.status, e.code, e.message)
                return
            except StoreError as e:
                self._send_error(400, e.code, e.message)
                return
            self._send_json(200, info)

        def _chat_completions(self, req: dict):
            engine = self.server.state.engine
            messages = req.get("messages")
            if not isinstance(messages, list) or not messages or not all(
                isinstance(m, dict) and "role" in m and "content" in m
                for m in messages
            ):
                self._send_error(
                    400, "invalid_messages",
                    'body must contain a non-empty "messages" list of '
                    '{"role": ..., "content": ...} objects',
                )
                return

            max_tokens = req.get("max_tokens", req.get("max_completion_tokens",
                                                       DEFAULT_MAX_NEW_TOKENS))
            if not isinstance(max_tokens, int) or max_tokens < 1:
                self._send_error(400, "invalid_max_tokens",
                                 '"max_tokens" must be a positive integer')
                return

            prompt_ids = engine.tokenize_chat(messages)
            cfg = engine.capacity
            decision = admit(
                cfg,
                input_tokens=len(prompt_ids),
                max_new_tokens=max_tokens,
                allow_maximum_context=bool(req.get("allow_maximum_context", False)),
            )
            if not decision.ok:
                self._send_error(
                    400, decision.error_code, decision.error_message,
                    request_budget=decision.total_tokens,
                    active_cap=active_cap(cfg),
                    model_context_limit=cfg.model_context_limit,
                    estimated_peak_memory_gb=round(
                        estimate_peak_memory_bytes(cfg, decision.total_tokens) / 1e9, 2),
                )
                return

            stream = bool(req.get("stream", False))
            completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            if stream:
                self._stream_response(engine, completion_id, created, prompt_ids,
                                      max_tokens, decision.warning)
            else:
                self._blocking_response(engine, completion_id, created, prompt_ids,
                                        max_tokens, decision.warning)

        # ---- generation paths ------------------------------------------
        def _sse_write(self, obj) -> None:
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        def _chunk(self, engine, completion_id, created, delta, finish_reason=None):
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": engine.model_id,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": finish_reason}],
            }

        def _stream_response(self, engine, completion_id, created, prompt_ids,
                             max_tokens, warning):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            cancel = threading.Event()
            tokens = 0
            gen = engine.stream_chat(prompt_ids, max_tokens, cancel)
            try:
                first = {"role": "assistant", "content": ""}
                chunk = self._chunk(engine, completion_id, created, first)
                if warning:
                    chunk["warning"] = warning
                self._sse_write(chunk)
                for seg in gen:
                    tokens += 1
                    if seg:
                        self._sse_write(self._chunk(
                            engine, completion_id, created, {"content": seg}))
                finish = "length" if tokens >= max_tokens else "stop"
                self._sse_write(self._chunk(engine, completion_id, created, {},
                                            finish_reason=finish))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                cancel.set()
                gen.close()  # deterministic stop at the next token boundary
            except GenerationCancelled:
                pass

        def _blocking_response(self, engine, completion_id, created, prompt_ids,
                               max_tokens, warning):
            cancel = threading.Event()
            parts = []
            tokens = 0
            gen = engine.stream_chat(prompt_ids, max_tokens, cancel)
            try:
                for seg in gen:
                    tokens += 1
                    if seg:
                        parts.append(seg)
            except GenerationCancelled:
                pass
            finish = "length" if tokens >= max_tokens else "stop"
            body = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": engine.model_id,
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": "".join(parts)},
                             "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": tokens,
                    "total_tokens": len(prompt_ids) + tokens,
                },
            }
            if warning:
                body["warning"] = warning
            self._send_json(200, body)

    return Handler


def build_server(engine_or_state, host: str = "127.0.0.1", port: int = 8321):
    """Accepts an EngineState, or a bare Engine (legacy/M2 behavior: no store)."""
    validate_host(host)
    if isinstance(engine_or_state, EngineState):
        state = engine_or_state
    else:
        state = EngineState(engine_or_state, store=None,
                            active_id=engine_or_state.model_id)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True
    server.state = state
    return server


def workspace_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="FreeToken localhost inference server")
    p.add_argument("--model-id", default=None,
                   help=f"manifest model id (default {DEFAULT_MODEL_ID})")
    p.add_argument("--model", default=None,
                   help="legacy: direct path to a converted MLX model dir "
                        "(bypasses the manifest and runtime switching)")
    p.add_argument("--host", default="127.0.0.1",
                   help="loopback address only (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8321)
    p.add_argument("--allow-extended", action="store_true",
                   help=f"raise the active cap from the default "
                        f"{CapacityConfig.default_cap} to the extended cap")
    p.add_argument("--default-cap", type=int, default=CapacityConfig.default_cap)
    p.add_argument("--extended-cap", type=int, default=CapacityConfig.extended_cap)
    args = p.parse_args(argv)

    validate_host(args.host)
    capacity = CapacityConfig(
        allow_extended=args.allow_extended,
        default_cap=args.default_cap,
        extended_cap=args.extended_cap,
    )

    store = None
    if args.model:
        engine = Engine(args.model, capacity)
        active_id = args.model.rstrip("/").split("/")[-1]
    else:
        store = ModelStore(workspace_root())
        active_id = args.model_id or DEFAULT_MODEL_ID
        entry = store.get(active_id)
        vr = store.verify(active_id)
        if not vr.ok:
            print(f"error: manifest verification failed for {active_id!r}: "
                  f"exists={vr.exists} size_ok={vr.size_ok} "
                  f"sha256_ok={vr.sha256_ok} — refusing to start", file=sys.stderr)
            return 1
        capacity.prefill_curve = PREFILL_CURVES[entry.prefill_curve]
        engine = Engine(store.select(active_id), capacity)

    print(f"loading model {engine.model_path} ...", flush=True)
    engine.load()
    state = EngineState(engine, store=store, active_id=active_id)
    server = build_server(state, args.host, args.port)
    cfg = engine.capacity
    print(
        f"freetoken server: http://{args.host}:{args.port} | model={active_id} "
        f"| cap={active_cap(cfg)} (default={cfg.default_cap}, "
        f"extended={cfg.extended_cap}, max={cfg.model_context_limit}) "
        f"| weights={cfg.weights_resident_bytes / 1e9:.2f} GB resident"
        + ("" if store is None else
           f" | catalog={','.join(store.manifest.ids())}"),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
