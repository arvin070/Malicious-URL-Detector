# Shipping URLGuard to real users, on both platforms

This is the part that is not code. Read it before promising anyone a
"permanent" install, because the two platforms differ sharply and one of them
has a hard expiry date.

## Android — done, and genuinely permanent

`android/app/build/outputs/apk/release/app-arm64-v8a-release.apk`

A signed release APK. Sideloaded, it keeps working indefinitely: Android does
not expire self-signed apps. Install with:

```bash
adb install -r app-arm64-v8a-release.apk
```

Three ABI splits are built; `arm64-v8a` covers essentially every phone made in
the last decade. `armeabi-v7a` is for older 32-bit devices and `x86_64` is for
the emulator.

**The signing key is a throwaway.** `android/app/urlguard-demo.keystore` is in
the repo on purpose so the expo build is reproducible on any machine, with the
password in `build.gradle.kts`. That is fine for a demo and wrong for
publishing: anyone with the repo can sign an update that Android will accept
as yours. Before Play Store:

```bash
keytool -genkeypair -v -keystore urlguard-release.keystore \
  -alias urlguard -keyalg RSA -keysize 4096 -validity 10000
```

Keep it out of the repo, back it up, and never lose it — Play ties the app
identity to that key permanently, and there is no recovery.

For Play you also need an **AAB**, not an APK: `gradle bundleRelease`.

## iOS — needs a Mac, and "permanent" needs a paid account

**There is no such thing as an APK for iOS.** The format is `.ipa`, and it can
only be produced by Xcode on macOS: the iOS SDK, the compiler and the signing
tools ship with Xcode and exist for no other platform. No Linux toolchain,
emulator or cross-compiler changes this. The build was written and its core
tested here; the `.ipa` has to be produced on a Mac.

```bash
cd ios && TEAM_ID=XXXXXXXXXX ./build-ios.sh archive
```

### How long an iOS install lasts

This is the question that actually matters for "permanent":

| How it is signed | Who can install | How long it lasts |
|---|---|---|
| Free Apple ID (personal team) | your own device | **7 days**, then it stops launching |
| Apple Developer Program, development | devices you register (100/yr) | 1 year, re-sign to renew |
| TestFlight | up to 10,000 testers by invite | 90 days per build |
| App Store | anyone | **indefinite** |

So a permanent iOS app means the **Apple Developer Program** — $99/year — and
then TestFlight for a group of users, or the App Store for everyone. A free
account cannot produce a build that survives a week. There is no way around
this; it is Apple policy, not a build setting.

### What the firewall needs from Apple

The firewall is an `NEPacketTunnelProvider`. For it to run at all:

1. Enable the **Network Extensions** capability on your App ID, with
   `Packet Tunnel` checked.
2. Create the App Group `group.com.aics.urlguard` and attach it to all four
   targets.
3. Rename `com.aics.urlguard` to a bundle id you own, in `ios/project.yml` and
   the four `.entitlements` files.

For **App Store review**, a VPN-mechanism app gets extra scrutiny, and the
guidelines are specific:

- **5.4** — apps using VPN APIs must be from an enrolled organization (not an
  individual account), must declare what data they collect and how it is used,
  and must not sell, use or disclose that data to third parties.
- The review notes should say plainly what this app does: the tunnel is
  **local only**, there is no remote server, DNS queries are answered on the
  device, and nothing is transmitted. Reviewers reject VPN apps that are vague
  about where traffic goes.
- The privacy manifest (`PrivacyInfo.xcprivacy`) must match actual behaviour.

Be aware that an individual developer account is **not** sufficient for 5.4 —
Apple requires an organization enrollment for VPN-capable apps, which needs a
D-U-N-S number.

## Keeping the two platforms in step

Both builds load the same ONNX model, metadata and suffix list from
`android/app/src/main/assets/`. The iOS project references those files rather
than keeping copies, so re-running:

```bash
python src/export_onnx.py --bundle models/v3/url_models.joblib --data data/urlguard_dataset.csv
```

updates both. This matters: the Android build once sat on the v2 model while
the server served v3, and nothing caught it because the two had separate
copies.

After any change to `src/features.py`, regenerate the golden vectors with the
same command and run **both** parity suites. They are what stop a platform
from silently scoring URLs differently from the model it ships:

```bash
cd android && gradle test     # 38 tests, Kotlin port
cd ios     && swift test      # 39 tests, Swift port
python -m unittest discover -s tests    # 194 tests, the pipeline itself
```
