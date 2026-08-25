#!/bin/sh
# SaveToken signing & notarization tooling.
#
# Safety rules implemented here:
#   * Secrets are never printed, stored, or read from the environment by us;
#     credentials live only in your keychain / notarytool keychain profile.
#   * Nothing is ever submitted to Apple unless a notarytool keychain profile
#     already exists AND you pass --submit explicitly.
#   * Without a Developer ID identity this tool produces an ad-hoc
#     development-signed app and an honest "blocked" report — it never fakes
#     Developer ID signing or notarization.
#
# Usage:
#   tools/sign_app.sh check                       detect signing identities
#   tools/sign_app.sh ad-hoc [--app PATH]         development signing (hardened runtime)
#   tools/sign_app.sh developer-id [--identity NAME] [--app PATH]
#   tools/sign_app.sh notarize-preflight --profile NAME
#   tools/sign_app.sh notarize-preflight --profile NAME --submit   (only if profile exists)
set -eu

HERE="$(cd "$(dirname "$0")/.." && pwd)"
APP_DEFAULT="$HERE/dist/SaveToken.app"
ENTITLEMENTS="$HERE/FreeTokenApp.entitlements"
ZIP_DEFAULT="$HERE/dist/SaveToken.app.zip"

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

developer_id_identities() {
    security find-identity -v -p codesigning 2>/dev/null \
        | grep 'Developer ID Application:' \
        | sed 's/^[[:space:]]*//' || true
}

spctl_note() {
    cat <<'EOF'
NOTE (spctl / Gatekeeper):
  `spctl --assess` rejects ad-hoc-signed builds by design. Gatekeeper's
  assessment requires a Developer ID Application signature plus an Apple
  notarization ticket for apps distributed outside the App Store. An ad-hoc
  signature ("Signature=adhoc") is cryptographically valid — codesign
  --verify passes and macOS will run the app locally (first launch may ask
  to confirm opening an unidentified app) — but it carries no Developer ID
  and no notarization record, so assessment correctly reports it as rejected.
  This is expected for development builds and is not a defect.
EOF
}

blocked_report() {
    cat <<'EOF'
SIGNING STATUS: BLOCKED (honest report — nothing was faked)
  * Developer ID Application identity: NOT FOUND in the local keychain.
  * Notarization: BLOCKED — requires a Developer ID certificate and an
    Apple-ID notarytool keychain profile; neither is configured here.

What WAS done: the app is ad-hoc signed with the hardened runtime, which is
valid for local development and local execution.

Manual steps remaining (do these yourself; never paste credentials into chat):
  1. Enroll in the Apple Developer Program and install the
     "Developer ID Application" certificate into your login keychain
     (Xcode → Settings → Accounts → Manage Certificates).
  2. Re-run: tools/sign_app.sh check          (identity should now appear)
  3. Sign:   tools/sign_app.sh developer-id [--identity "<full name>"]
  4. Create a notarytool keychain profile (interactive; your credentials):
       xcrun notarytool store-credentials
  5. Preflight: tools/sign_app.sh notarize-preflight --profile <name>
     Submit:   tools/sign_app.sh notarize-preflight --profile <name> --submit
  6. Staple after acceptance:
       xcrun stapler staple dist/SaveToken.app
EOF
}

verify_signature() {
    app="$1"
    codesign --verify --deep --strict --verbose=2 "$app"
    codesign -dv "$app" 2>&1 | grep -E 'Identifier|Signature size|Authority|TeamIdentifier' || true
    # Signing changes the binary; keep the release checksums truthful.
    if [ "$(cd "$(dirname "$app")" && pwd)" = "$HERE/dist" ]; then
        python3 "$HERE/tools/make_release.py" --refresh --dist "$HERE/dist" >/dev/null
        echo "release metadata refreshed: $(cat "$HERE/dist/SaveToken.sha256")"
        # Keep Finder/iCloud resource forks out of the distribution archive.
        ditto --norsrc -c -k --keepParent "$app" "$ZIP_DEFAULT"
        echo "notarization zip refreshed: $ZIP_DEFAULT"
    fi
}

cmd_check() {
    echo "== codesigning identities (codesigning policy) =="
    ids="$(developer_id_identities)"
    if [ -n "$ids" ]; then
        echo "$ids"
        echo
        echo "Developer ID available. Sign with:"
        echo "  tools/sign_app.sh developer-id [--identity \"<full identity name>\"]"
    else
        echo "(none)"
        echo
        echo "No Developer ID identity. Available path: ad-hoc development signing:"
        echo "  tools/sign_app.sh ad-hoc"
        blocked_report
    fi
}

prepare_bundle() {
    # Extended attributes (iCloud/Finder/provenance) make codesign refuse the
    # bundle ("resource fork, Finder information, or similar detritus").
    xattr -cr "$1" 2>/dev/null || true
}

cmd_ad_hoc() {
    app="$1"
    [ -d "$app" ] || { echo "error: app bundle not found: $app (run sh package_app.sh first)" >&2; exit 1; }
    prepare_bundle "$app"
    codesign --force --options runtime --timestamp=none \
        --entitlements "$ENTITLEMENTS" --sign - "$app"
    verify_signature "$app"
    echo "AD-HOC signing complete (hardened runtime enabled)."
    spctl_note
}

cmd_developer_id() {
    app="$1"; identity="$2"
    [ -d "$app" ] || { echo "error: app bundle not found: $app" >&2; exit 1; }
    ids="$(developer_id_identities)"
    if [ -z "$ids" ]; then
        blocked_report
        exit 1
    fi
    if [ -z "$identity" ]; then
        count="$(printf '%s\n' "$ids" | grep -c 'Developer ID Application:')"
        if [ "$count" -ne 1 ]; then
            echo "Multiple Developer ID identities found; choose one with --identity:" >&2
            echo "$ids" >&2
            exit 1
        fi
        identity="$(printf '%s\n' "$ids" | head -1 | sed 's/.*"\(.*\)".*/\1/')"
    fi
    echo "$ids" | grep -F "$identity" >/dev/null || {
        echo "error: identity not among valid codesigning identities: $identity" >&2; exit 1; }
    prepare_bundle "$app"
    codesign --force --options runtime \
        --entitlements "$ENTITLEMENTS" --sign "$identity" "$app"
    verify_signature "$app"
    echo "Developer ID signing complete. Next: notarize-preflight --profile <name>"
}

cmd_notarize() {
    profile="$1"; submit="$2"
    [ -n "$profile" ] || { echo "error: --profile NAME is required" >&2; exit 1; }
    # Preflight: does the keychain profile exist? No secrets are printed.
    if ! xcrun notarytool history --keychain-profile "$profile" >/dev/null 2>&1; then
        cat <<EOF
NOTARIZATION: BLOCKED
  Keychain profile '$profile' is not configured (or notarytool cannot use it).
  Create it yourself (interactive; credentials stay in your keychain):
      xcrun notarytool store-credentials
  Then re-run this preflight. Nothing was submitted.
EOF
        exit 0
    fi
    echo "Profile '$profile' is configured and usable."
    if [ "$submit" != "yes" ]; then
        cat <<EOF
Preflight only — nothing submitted. To submit, you need a zipped build:
    ditto -c -k --keepParent dist/SaveToken.app dist/SaveToken.app.zip
Then explicitly run:
    tools/sign_app.sh notarize-preflight --profile $profile --submit
EOF
        exit 0
    fi
    # Explicit submission path — only reachable when the profile exists.
    [ -f "$ZIP_DEFAULT" ] || { echo "error: $ZIP_DEFAULT missing; create it with ditto (see above)" >&2; exit 1; }
    xcrun notarytool submit "$ZIP_DEFAULT" --keychain-profile "$profile" --wait
    echo "Submission finished. If accepted, staple with:"
    echo "    xcrun stapler staple dist/SaveToken.app"
}

# ---- arg parsing -----------------------------------------------------------
MODE="${1:-check}"
case "$MODE" in
    -h|--help|help) usage 0 ;;
esac
shift || true

APP="$APP_DEFAULT"; IDENTITY=""; PROFILE=""; SUBMIT="no"
while [ $# -gt 0 ]; do
    case "$1" in
        --app) APP="$2"; shift 2 ;;
        --identity) IDENTITY="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --submit) SUBMIT="yes"; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

case "$MODE" in
    check) cmd_check ;;
    ad-hoc) cmd_ad_hoc "$APP" ;;
    developer-id) cmd_developer_id "$APP" "$IDENTITY" ;;
    notarize-preflight) cmd_notarize "$PROFILE" "$SUBMIT" ;;
    *) echo "unknown mode: $MODE" >&2; usage 1 ;;
esac
