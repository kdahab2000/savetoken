# SaveToken macOS app

Native SwiftUI client for local Ollama and the optional SaveToken MLX backend.
The executable target uses Apple SDKs only and has no third-party Swift
dependencies.

See the repository [README](../README.md) for setup, privacy boundaries, SSH
safety, and release status.

## Develop and test

```sh
swift test --scratch-path "${TMPDIR%/}/SaveToken-build"
SCRATCH_PATH="${TMPDIR%/}/SaveToken-build" sh package_app.sh
sh tools/sign_app.sh ad-hoc
```

The Swift suite covers admission calculations, wire parsing, SSE framing,
reasoning cleanup, offline setup, settings persistence, Ollama selection, SSH
alias parsing, and release portability. Packaging creates these ignored build
artifacts under `dist/`:

- `SaveToken.app`
- `SaveToken.app.zip`
- `SaveToken.sha256`
- `RELEASE_MANIFEST.json`

No model weights or user settings are copied into the bundle.

## Architecture

- `AppState.swift` — providers, persisted selection, generation, SSH approval.
- `ServerClient.swift` — loopback-only SaveToken/Ollama HTTP clients.
- `SSHClient.swift` — configured-alias parsing and bounded batch-mode SSH.
- `ResponseTextSanitizer.swift` — hides reasoning and tokenizer control text.
- `Views.swift` / `ChatViews.swift` — SwiftUI interface.
- `tools/` — deterministic packaging and signing/notarization preflight.

The app targets macOS 13+. Public development artifacts are ad-hoc signed;
general distribution still requires Developer ID signing and notarization.
