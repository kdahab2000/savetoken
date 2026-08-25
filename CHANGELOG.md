# Changelog

This project follows semantic versioning. Dates use ISO 8601.

## 0.5.0 — 2026-08-26

Initial public development preview.

- Native SwiftUI macOS chat interface.
- Local Ollama discovery and streaming chat with hidden reasoning disabled.
- Optional loopback-only SaveToken MLX backend.
- Persistent provider and Ollama model selection.
- Response cleanup for reasoning/control markers.
- Approval-gated SSH commands using named aliases from `~/.ssh/config`.
- Ordinary chat remains on the streaming path when SSH tools are enabled.
- Embedding-only Ollama models are excluded from the chat picker.
- Ollama cloud-tagged models are visibly labeled as remote.
- Deterministic app packaging, icon, release manifest, SHA-256 checksum, and
  hardened-runtime ad-hoc signing support.

Known release limitation: no Developer ID signature or Apple notarization.
