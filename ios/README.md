# URLGuard for iOS

Native iOS client sharing the Android app's model, threshold and feature
extractor.

## What iOS can and cannot do

This matters more than the code, because the iOS build is **not** a feature
match for the Android one, and the difference is imposed by the platform:

| Capability | Android | iOS | Why |
|---|---|---|---|
| On-device URL scoring | yes | **yes** | same ONNX model, same 124 features |
| Scan from any app | intent filter | **Share → URLGuard** | iOS has no system link handler for a normal app |
| Warn before a page loads | any browser | **Safari only** | Safari Web Extension is the only hook Apple offers |
| Block the navigation | yes | **no, warns only** | see below |
| DNS firewall | `VpnService` | **yes**, local packet tunnel | see below |
| System content filter | n/a | **no** | `NEFilterDataProvider` needs a supervised/MDM device |

### The firewall

`NEDNSProxyProvider` and `NEFilterDataProvider` do require a supervised/MDM
device, but **`NEPacketTunnelProvider` does not** — it is available to any
developer account, and it is how App Store firewalls are actually built. The
`FirewallExtension` target runs a **local-only** tunnel: there is no server, no
exit node and nothing is tunnelled anywhere. Its only purpose is to put the
process in the path of the device's own DNS queries so they can be read,
scored against the bundled hostname model, and either forwarded or answered
NXDOMAIN.

iOS shows a VPN badge while it runs, because that is the mechanism. The
Firewall screen says so plainly rather than letting the badge imply traffic is
being sent somewhere.

Three rules it is built around, each with a test:

- **Fail open.** Every error path forwards the query. A firewall whose model
  fails to load and therefore blocks everything has taken the user's network
  down — far worse than a missed detection.
- **Only forward lookups.** A/AAAA only. A PTR query is a reverse lookup of an
  address already resolved; blocking one protects nobody. Measured on a real
  Android device: `224.166.102.103.in-addr.arpa` scored 0.97 and was blocked,
  which was pure false-positive noise.
- **Threshold 0.95, not the trained 0.65.** The training rule targets
  FPR ≤ 0.05, fine for a dismissible screen and wrong for a system-wide
  blocker: at 0.65 roughly 1 benign domain in 20 stops resolving and the phone
  feels broken. At 0.95 it is about 1 in 300, for roughly half the recall.

Only *blocked* lookups are recorded, in the shared App Group. A full query log
would be a record of everywhere the user went, which this app has no reason to
keep.

The Safari extension warns with a dismissible banner rather than blocking.
That is deliberate: the hard-benign benchmark for this model measured a real
false-positive rate on well-known sites, and a tool that strands people on
pages they wanted gets switched off — after which it protects nobody.

## Layout

```
ios/
  Package.swift              SwiftPM package -- builds and tests on Linux
  Sources/URLGuardCore/      the part that must match Python exactly
    UrlFeatures.swift          all 124 features, ported from src/features.py
    TldExtract.swift           tldextract's exact split behaviour
    Punycode.swift             RFC 3492 decode (Foundation has no IDN on Linux)
    CRC32.swift                zlib.crc32, for the hashed n-gram buckets
    RiskAssessment.swift       bands/confidence, mirrors webapp/assessment.py
    DnsPackets.swift           IPv4/UDP/DNS parsing and NXDOMAIN construction
    FirewallPolicy.swift       what to block, what to never touch, fail-open
  Tests/URLGuardCoreTests/   39 tests, including golden-vector parity
  App/                       Apple-only; needs Xcode
    URLGuard/                  SwiftUI app
    ShareExtension/            "Share → URLGuard" from any app
    SafariExtension/           warns in Safari
    FirewallExtension/         local packet tunnel; DNS filtering
  project.yml                XcodeGen spec
```

The split is the point. `URLGuardCore` has no Apple dependencies, so the code
that has to agree with `src/features.py` to the last decimal is **compiled and
tested without a Mac**. Everything that cannot be is confined to `App/`.

## Verifying the feature port (no Mac needed)

```bash
cd ios && swift test
```

`FeatureParityTests` runs all 625 golden vectors from
`android/app/src/test/resources/golden_vectors.json` — the same fixture the
Kotlin port is held to — and compares all 124 features per URL against the
values Python produced. This exists because the project has already shipped
three Python↔Kotlin parity bugs (port extraction on `[::1]:8080`, the
`_NullParse` path rule, `urlparse` stripping `;params`), each of which produced
plausible-looking features that were quietly wrong.

Regenerate the fixture whenever `src/features.py` changes:

```bash
python src/export_onnx.py --bundle models/v3/url_models.joblib --data data/urlguard_dataset.csv
```

### The Swift-specific trap

`String.count` in Swift counts **grapheme clusters**; Python's `len()` counts
**code points**. For `"é"` written as `e` + U+0301 those differ, and every
length, ratio and entropy feature would shift. Every length in `UrlFeatures`
goes through `pyLen`, which counts unicode scalars. There is a test for it.

## Building the app (needs macOS)

```bash
brew install xcodegen
cd ios && xcodegen generate
open URLGuard.xcodeproj
```

Then, before building to a device:

1. Set `DEVELOPMENT_TEAM` in `project.yml` (or pick a team in Xcode's
   Signing & Capabilities for all three targets).
2. Change `com.aics.urlguard` to a bundle id you own, in `project.yml` and in
   the three `.entitlements` files.
3. The App Group `group.com.aics.urlguard` must exist in your developer
   account — it is what lets the app and extensions share one scan history and
   one signed-in session.
4. Enable the extensions on the device: Settings → Apps → Safari → Extensions
   → URLGuard, and the Share Extension appears in the share sheet under "More".

`onnxruntime` is pulled in as a Swift package by `project.yml`; no manual
framework setup.

## Model assets

The app references `android/app/src/main/assets/` directly rather than keeping
its own copy:

- `url_model.onnx`
- `model_meta.json` (carries the threshold — currently 0.49)
- `public_suffix_list.txt`

One set of files for both platforms, so they cannot drift onto different model
versions. (They did once: the Android build sat on v2 while the server served
v3.) Re-run `src/export_onnx.py` and both platforms pick up the new model.

## Security hardening

- **Tokens in the Keychain**, `kSecAttrAccessibleAfterFirstUnlock`, shared with
  the extensions through a Keychain access group. Not `UserDefaults`, which is
  a plist in the container and lands in backups.
- **Optional Face ID / passcode lock** on the app, re-locking whenever it
  leaves the foreground so the history is not readable from the app switcher.
  Presented as a screen lock, which is what it is — not encryption.
- **No query log.** The firewall records blocks only. Scan history stays on the
  device unless the user signs in and enables deep scan, which is off by
  default.
- **No cookies**, ephemeral `URLSession`, bearer tokens only.
- **ATS left on**, with only `NSAllowsLocalNetworking` so a self-hosted server
  on a LAN address works. No blanket arbitrary-loads exemption.
- **No secrets in the bundle.** The only credential is the user's own token,
  obtained by signing in.
- **Privacy manifest** (`PrivacyInfo.xcprivacy`) declaring what is collected.

Deliberately **not** included: jailbreak detection. Every published check is
bypassed by the tooling it tries to detect, so it mainly produces false
assurance plus false positives on ordinary devices.

## What is verified and what is not

**Verified here, on Linux:** the 124-feature port against 625 golden vectors;
the TLD/punycode/CRC32 primitives; the parser quirks that have broken a port
before; the risk-band presentation; and — most importantly for the firewall —
the DNS packet layer and the blocking policy. 39 tests.

The packet tests matter more than the rest. That code sits in the data path of
every DNS query the phone makes, and a packet it corrupts is a phone that
cannot resolve anything. Reference packets and both checksums were computed
independently in Python and asserted against, rather than eyeballed.

**Not verified here:** everything in `App/`. It has never been compiled,
because that needs Xcode. For the firewall that means the tunnel plumbing —
`startTunnel`, the routing settings and the packet read loop — is written but
unproven; the decisions it makes and the packets it builds are tested. The ONNX output-tensor handling in
`UrlClassifier.probability(from:)` in particular is written against the two
shapes skl2onnx and onnxmltools emit, but which one this model produces has
only been confirmed on Android, not through the Swift bindings. Check it on
first run: an iOS score for a given URL must match the Android score for the
same URL to within floating-point noise.
