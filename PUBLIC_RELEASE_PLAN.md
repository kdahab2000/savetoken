# SaveToken public-release checklist

## Completed for the 0.5.0 development preview

- Portable per-user settings under Application Support.
- Loopback-only defaults for Ollama and SaveToken MLX.
- No model weights, credentials, SSH configuration, transcripts, build caches,
  app bundles, or zip files committed to source control.
- Ollama-first setup; embedding models hidden from chat.
- Ollama cloud-tagged models labeled as remote.
- SSH disabled by default, restricted to configured aliases, batch-mode only,
  and approval-gated for every model-requested command.
- Consistent SaveToken name, icon, bundle identifier, README, license, security
  policy, contribution guide, changelog, tests, packaging tools, and CI.
- Ad-hoc signed development artifact with manifest and SHA-256 checksum.

## Required before general availability

- Enroll in the Apple Developer Program and install a Developer ID Application
  certificate.
- Sign the release with Developer ID, submit it to Apple's notary service, and
  staple the accepted ticket.
- Test the downloaded artifact on a clean Apple-Silicon Mac account.
- Add onboarding screenshots or a short demonstration GIF.
- Expand SSH tests for process timeout, cancellation, and output truncation.
- Decide whether the optional custom MLX checkpoints will have a lawful public
  download source. Until then, keep Ollama as the documented default.

## Release invariants

- Never commit or package model weights, private keys, passwords, API tokens,
  `.env` files, chat transcripts, or machine-specific absolute paths.
- Never add a non-loopback SaveToken HTTP endpoint.
- Never execute a model-requested SSH command without per-command approval.
- Never call an ad-hoc signed artifact notarized or production-ready.
