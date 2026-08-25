#!/bin/sh
# Reproducible release packaging:
#   1. swift build -c release (scratch path OUTSIDE iCloud-synced ~/Documents)
#   2. deterministic dist assembly via tools/make_release.py
#      (dist/SaveToken.app + RELEASE_MANIFEST.json + SaveToken.sha256)
# Model artifacts are never copied into the app or dist.
set -eu

cd "$(dirname "$0")"
SCRATCH="${SCRATCH_PATH:-${HOME}/FreeTokenApp-build}"

swift build -c release --scratch-path "$SCRATCH"
BIN="$(swift build -c release --scratch-path "$SCRATCH" --show-bin-path)/FreeTokenApp"

python3 tools/make_release.py --bin "$BIN" --dist dist

echo "release artifact: $(pwd)/dist/SaveToken.app"
echo "next: sh tools/sign_app.sh check      (detect signing identities)"
echo "      sh tools/sign_app.sh ad-hoc     (development signing)"
