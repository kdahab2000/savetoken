# Contributing

Contributions are welcome through issues and pull requests.

## Development setup

```sh
cd FreeTokenApp
swift test --scratch-path "${TMPDIR%/}/SaveToken-build"
SCRATCH_PATH="${TMPDIR%/}/SaveToken-build" sh package_app.sh
cd ..
python3 -m unittest freetoken.test_m5 -v
```

The `test_m2` and `test_m3` Python suites exercise the optional MLX runtime and
require the separately provisioned model artifacts referenced by the local
manifest. They are not required for an Ollama-only app change.

## Pull-request rules

- Keep all app HTTP endpoints loopback-only.
- Preserve explicit confirmation for every model-requested SSH command.
- Never add telemetry or prompt logging without a separate, explicit design
  and user opt-in.
- Add or update tests for behavior changes.
- Keep user-visible local, cloud, measured, and estimated labels accurate.
- Do not commit generated `.app`/zip files, build directories, model weights,
  transcripts, SSH configuration, credentials, `.env` files, or machine-
  specific absolute paths.

## Releases

Release artifacts are produced from a clean macOS build. Development builds
may be ad-hoc signed and must be labeled as such. General-availability builds
require Developer ID signing, Apple notarization, a SHA-256 checksum, and a
release manifest.
