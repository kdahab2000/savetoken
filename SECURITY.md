# Security policy

## Supported version

Security fixes are applied to the latest release and the default branch.

## Reporting a vulnerability

Do not open a public issue for an undisclosed vulnerability. Use GitHub's
private vulnerability reporting on the repository Security page. Include the
affected version, reproduction steps, impact, and any suggested mitigation.
Do not include real credentials, private keys, patient data, or other secrets.

## Security model

SaveToken is a local desktop client, not a security boundary:

- HTTP requests are constructed for `127.0.0.1` only.
- Ollama models whose tags end in `:cloud` or `-cloud` may use Ollama's remote service and are
  labeled **CLOUD** in the app.
- SaveToken has no telemetry and does not persist chat transcripts.
- The optional SSH feature reads alias names from the user's own
  `~/.ssh/config`. It never reads or transmits private-key contents.
- Model-requested SSH commands are disabled by default and require a visible
  approval dialog for every command.
- SSH uses batch mode, a connection timeout, bounded output, and a tool-round
  limit. Interactive password and TTY workflows are unsupported.
- Model text and proposed commands are untrusted input. Users must inspect a
  command before approving it.

## Distribution status

Development artifacts are ad-hoc signed and are not Apple-notarized. A public
release must state this clearly until a Developer ID signature and notarization
ticket are available. Never commit signing certificates, API tokens, SSH keys,
notary credentials, model weights, or `.env` files.
