# Optional SaveToken MLX backend

`freetoken` is a localhost-only, OpenAI-compatible inference server for Apple
Silicon using `mlx-lm`. Most users should use SaveToken with Ollama instead;
this backend is for advanced development with separately provisioned MLX model
artifacts.

No model weights are included in this repository or its releases. The example
manifest records the custom development checkpoints' relative locations,
sizes, and SHA-256 values, but deliberately has no download URLs.

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/models/switch`
- `POST /v1/chat/completions` with optional SSE streaming

The server accepts loopback hosts only. It has no telemetry or cloud fallback,
and request logs never contain prompt or response content.

## Run with provisioned weights

From the repository root:

```sh
python3 -m freetoken.manager verify qwen3.5-healthcare-bf16
python3 -m freetoken.server --model-id qwen3.5-healthcare-bf16 --port 8321
```

Use `--model /absolute/path/to/an/mlx-model` for an explicitly chosen local
model directory. The legacy direct-path mode bypasses manifest switching, so
use it only with artifacts you trust.

## Model manager security

- Manifest paths are traversal-checked and workspace-relative.
- Artifacts are verified by exact size and SHA-256 before selection.
- Downloads require a pinned HTTPS URL, checksum, and explicit
  `--allow-download`; the shipped manifest has no URLs.
- Partial downloads are promoted only after successful verification.
- `trust_remote_code` is never enabled.

## Tests

```sh
# Portable policy, packaging, and signing tests (no weights required after app build)
python3 -m unittest freetoken.test_m5 -v

# Runtime suites; require mlx-lm and the provisioned manifest models
python3 -m unittest freetoken.test_m2 freetoken.test_m3 -v
```

The custom healthcare checkpoints are research-only and are not for clinical
diagnosis or treatment.
