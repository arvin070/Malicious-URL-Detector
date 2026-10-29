#!/usr/bin/env bash
# Build URLGuard for iPhone. MUST be run on macOS with Xcode installed --
# there is no way to produce an .ipa on Linux or Windows, because the iOS SDK,
# the compiler toolchain and the code-signing tools ship only with Xcode.
#
#   ./build-ios.sh            generate the project and build for the simulator
#   ./build-ios.sh archive    build a signed archive and export an .ipa
#
# Before `archive` works you must set TEAM_ID below (or export it), and the
# App ID, App Group and NetworkExtension capability must exist in your Apple
# Developer account. See README.md.
set -euo pipefail

cd "$(dirname "$0")"

TEAM_ID="${TEAM_ID:-}"
SCHEME="URLGuard"
BUILD_DIR="build"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: iOS builds require macOS with Xcode." >&2
  echo "       The Swift core can still be tested anywhere: swift test" >&2
  exit 1
fi

command -v xcodegen >/dev/null || {
  echo "installing xcodegen..."; brew install xcodegen; }

echo "==> verifying the feature port against the golden vectors"
# Run this FIRST, every time. It is the check that the iOS build scores URLs
# the same way the Python model was trained to. A build that ships a drifted
# feature extractor is a model that is quietly wrong.
swift test

echo "==> generating URLGuard.xcodeproj"
xcodegen generate

if [[ "${1:-}" != "archive" ]]; then
  echo "==> building for the simulator"
  xcodebuild -project URLGuard.xcodeproj -scheme "$SCHEME" \
    -sdk iphonesimulator -configuration Debug \
    -derivedDataPath "$BUILD_DIR" build
  echo
  echo "Built for the simulator. To install on a real iPhone you need a"
  echo "signed build: set TEAM_ID and run ./build-ios.sh archive"
  exit 0
fi

if [[ -z "$TEAM_ID" ]]; then
  echo "error: set TEAM_ID to your Apple Developer team id." >&2
  echo "       Find it at https://developer.apple.com/account -> Membership" >&2
  exit 1
fi

echo "==> archiving"
xcodebuild -project URLGuard.xcodeproj -scheme "$SCHEME" \
  -sdk iphoneos -configuration Release \
  -archivePath "$BUILD_DIR/URLGuard.xcarchive" \
  DEVELOPMENT_TEAM="$TEAM_ID" \
  archive

cat > "$BUILD_DIR/ExportOptions.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>method</key><string>development</string>
  <key>teamID</key><string>${TEAM_ID}</string>
  <key>signingStyle</key><string>automatic</string>
  <key>compileBitcode</key><false/>
  <key>stripSwiftSymbols</key><true/>
</dict>
</plist>
PLIST

echo "==> exporting .ipa"
xcodebuild -exportArchive \
  -archivePath "$BUILD_DIR/URLGuard.xcarchive" \
  -exportOptionsPlist "$BUILD_DIR/ExportOptions.plist" \
  -exportPath "$BUILD_DIR/ipa"

echo
echo "Done: $BUILD_DIR/ipa/URLGuard.ipa"
echo
echo "A 'development' export installs on devices registered to your team and"
echo "is NOT permanent -- it expires when the provisioning profile does."
echo "For a build that keeps working, change method to 'app-store' and upload"
echo "to TestFlight or the App Store. See README.md."
