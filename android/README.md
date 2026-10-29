# URL Guard — Android

On-device phishing-link detection. Same model as the desktop project, served
through ONNX Runtime Mobile. No URL is ever fetched, resolved or sent off the
device.

See the root `README.md` section **"10b. Android app"** for the measured
accuracy of each layer, the threshold reasoning, and the parity guarantees.

## Build

```bash
# from the repo root — regenerate the model assets first
python src/export_onnx.py
python src/train.py --url-transform hostname --model-dir models/hostname
python src/export_onnx.py --bundle models/hostname/url_models.joblib \
    --model XGBoost --override-threshold 0.95 \
    --out-dir android/app/src/main/assets/hostname

cd android
gradle :app:testDebugUnitTest     # 15 tests, incl. 76,250 feature comparisons
gradle :app:assembleRelease
```

APKs land in `app/build/outputs/apk/release/`. Install the one matching the
device: **`app-arm64-v8a-release.apk`** for virtually all modern phones.

```bash
adb install -r app/build/outputs/apk/release/app-arm64-v8a-release.apk
```

## Toolchain notes

Pinned combination that works here — the alternatives genuinely fail:

* **AGP 9.3.2 + Gradle 9.7.1.** AGP 8.5.2 dies on Gradle 9 with
  `JvmVendorSpec.IBM_SEMERU` (removed in Gradle 9).
* **No `org.jetbrains.kotlin.android` plugin.** AGP 9 has built-in Kotlin
  support; applying the separate plugin fails with
  `extension with name 'kotlin' already registered`.
* **No foojay toolchain resolver.** Version 0.8.0 references the same removed
  Gradle API. The build compiles with the JBR on `PATH`, targeting JVM 17.

## After installing

1. Open the app and leave **"Check links I tap"** on.
2. Optionally tap **Set URL Guard as default browser** — without it Android
   shows a chooser each time instead of checking automatically. Safe links are
   handed straight to the real browser.
3. **"Check DNS lookups"** is off by default. It is the system-wide layer but
   also the much weaker one (hostname only); read the note on that screen
   before relying on it. It takes Android's single VPN slot.

Try it: paste `http://secure-paypal-login.tk/verify/9231` into the "Check a URL"
box, or send yourself that link in a messaging app and tap it.
