# AI-Based Detection of Malicious and Phishing URLs

Detects phishing/malicious URLs from **lexical and structural features of the
URL string alone** — no page is ever fetched — and compares **three
algorithms** (Decision Tree, Random Forest, XGBoost) rather than relying on a
single model.

> **Status:** trained on a real, published phishing corpus (11,429 URLs). The
> earlier synthetic demo dataset is kept only as a documented negative example
> — see [§11 Why the synthetic results were meaningless](#11-why-the-synthetic-results-were-meaningless).

```
phishing-url-monitor/
├── data/
│   ├── real_urls.csv             # REAL dataset (generated, 11,429 URLs)  <-- used for training
│   └── sample_urls.csv           # synthetic demo data (kept as a baseline only)
├── models/
│   ├── url_models.joblib                        # all 3 trained models + metadata
│   ├── algorithm_comparison.csv                 # headline results table
│   ├── algorithm_comparison_domain_disjoint.csv # harder domain-disjoint split
│   ├── confusion_matrices.csv / .png            # TN/FP/FN/TP per algorithm
│   ├── feature_importance.csv / .png            # importance per algorithm
│   ├── roc_comparison.png                       # ROC curves, all 3 overlaid
│   ├── metric_comparison.png                    # bar chart of all metrics
│   ├── unseen_validation.csv / unseen_roc.png   # UNSEEN external-corpus results
│   ├── unseen_confusion.csv                     # unseen TN/FP/FN/TP + consensus
│   └── unseen_errors.csv                        # every misclassified unseen URL
├── src/
│   ├── features.py                # URL -> 21-feature vector
│   ├── download_real_dataset.py   # downloads + cleans the real dataset
│   ├── audit_dataset.py           # class balance + data-leakage audit
│   ├── train.py                   # trains + compares all 3 algorithms
│   ├── monitor.py                 # scores a URL against all 3 models
│   ├── validate_unseen.py         # validation on an INDEPENDENT corpus
│   └── make_sample_data.py        # generates the synthetic baseline
├── webapp/                        # Flask API + web UI
├── extension/                     # Chrome extension (real-time blocking)
├── tests/test_pipeline.py         # features -> training -> inference -> API
└── requirements.txt
```

## 1. Setup

Verified on **EndeavourOS (Arch-based), kernel 7.2.7-arch1-1, x86_64, Python 3.14.7**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note for this repo:** the `.venv/` that shipped here originally was a
> **Windows** virtualenv (`home = C:\Users\...\Python312`, `Scripts/*.exe`) and
> cannot run on Linux. It has been moved to `.venv-windows-backup/` and a native
> Linux venv created at `.venv/`. Delete the backup once you no longer need it.

Verify the install:

```bash
python -c "import sklearn, xgboost, pandas, numpy, joblib, tldextract, flask; print('ok')"
```

Versions confirmed working: scikit-learn 1.9.1, xgboost 3.4.1, pandas 3.0.6,
numpy 2.5.3, joblib 1.6.0, tldextract 5.3.2, Flask 3.1.3, flask-cors 6.0.5,
matplotlib 3.11.2, pyarrow 25.0.1.

## 2. Reproduce everything from scratch

These four commands regenerate the dataset, the audit, every model and every
figure. They are deterministic (`random_state=42` throughout).

```bash
python src/download_real_dataset.py              # -> data/real_urls.csv
python src/audit_dataset.py                      # class balance + leakage audit
python src/train.py                              # -> models/* (defaults to real_urls.csv)
python src/compare_baseline.py                   # -> models/baseline_vs_improved.csv
python src/monitor.py --url "http://secure-paypal-login.tk/verify/9231"
python src/validate_unseen.py                    # unseen independent corpus
python -m unittest discover -s tests -v          # pipeline tests
```

### Reproducibility

Every run is bit-identical, not merely close. Two consecutive `src/train.py`
runs produce byte-identical `models/cv_results.csv` and identical stdout.

Two things were needed to get there:

* **`PYTHONHASHSEED` independence.** The hashed n-gram features use
  `zlib.crc32`, not Python's `hash()`, which is salted per process. With
  `hash()` the same URL would produce different features after a restart, and
  a saved model would silently mean something else.
* **Serialised prediction.** `RandomForestClassifier.predict_proba` accumulates
  300 per-tree arrays across threads. Floating-point addition is not
  associative, so the thread interleaving changed the last 1-2 bits (measured:
  up to 4.4e-16). That is invisible until a sample sits exactly on the decision
  threshold -- then 1 ULP flips its class and the 4th decimal of
  Precision/F1/FPR moves between otherwise identical runs (measured: Random
  Forest F1 0.8415 vs 0.8414). `train.freeze_for_prediction()` sets `n_jobs=1`
  for prediction only; fitting stays parallel, so the cost is nil.

## 2b. Feature extraction (`src/features.py`)

122 features, all computed from the URL **string alone**. Nothing is fetched,
resolved, crawled, executed or connected to -- enforced by `TestNetworkIsolation`,
which patches `socket.socket`, `socket.create_connection`, `socket.getaddrinfo`,
`socket.gethostbyname`, `ssl.SSLContext.wrap_socket` and `urllib.request.urlopen`
and extracts features over 500 real corpus URLs.

Three nested feature sets, selectable with `--feature-set`:

| set | count | contents |
|---|---|---|
| `legacy` | 21 | the original features, names/values/order unchanged |
| `structural` | 86 | + component lengths, separator and special-char counts, query-parameter count, path depth, percent-encoding, repeated separators, suspicious port, punycode/IDN, Unicode confusables, per-component entropy, digit/letter ratios, domain/subdomain structure, brand-abuse and token statistics |
| `full` | 122 | + 24 hashed character-trigram and 12 hashed token-bigram buckets |

The original 21 keep their exact names, values and position at the front of
`FEATURE_NAMES`, so older bundles still index the same columns.

The n-gram block is normalised to a frequency distribution rather than raw
counts (`url_length` already carries length) and is computed on the
scheme-stripped, lowercased URL -- the two training corpora differ in whether
they store a scheme, so a raw n-gram would encode *which corpus a row came
from* rather than anything about the URL.

## 2c. Leakage control (`src/leakage.py`)

One module is the single source of truth for "are these two URLs the same?"
and "do these two sets share a domain?". `canonicalize()` and the
registered-domain lookup were previously copy-pasted into four scripts.

`canonicalize()` (lowercase, drop scheme, drop leading `www.`, drop trailing
slash) is used **only** to compare rows. It is never fed to the feature
extractor and never written back to the dataset -- the model must see each URL
exactly as a browser would, because `is_https` and `has_www` are real signal.

## 3. Dataset

### Source

| | |
|---|---|
| **Corpus** | Web page phishing detection dataset |
| **Authors** | Hannousse, A. & Yahiouche, S. (2021) |
| **Paper** | *Towards benchmark datasets for machine learning based website phishing detection: An experimental study*, Computers & Security, 2021 — [arXiv:2010.12847](https://arxiv.org/abs/2010.12847) |
| **Mirror used** | Hugging Face `pirocheto/phishing-url` |
| **Licence** | CC-BY-4.0 |
| **Label provenance** | phishing URLs from **PhishTank** and **OpenPhish** verified feeds; legitimate URLs from **Alexa / Tranco** top-site lists |

The published corpus ships 87 pre-computed features. We deliberately keep
**only the raw `url` string and the `status` label** and recompute all 21
features ourselves with `src/features.py`, so this project's own feature
extractor remains the single source of truth.

### Format

`data/real_urls.csv` has exactly two columns:

```
url,label
```

with `0 = benign` and `1 = phishing`.

### Safety

URLs are handled **strictly as strings**. No script in this project resolves,
requests, crawls, renders or executes a dataset URL. All 21 features are
computed from the URL text, and `tldextract` runs fully offline
(`suffix_list_urls=()`), so no network call is made during feature extraction.

### Preprocessing (`src/download_real_dataset.py`)

| Step | Rows |
|---|---:|
| rows downloaded | 11,430 |
| after label normalisation (`phishing`→1, `legitimate`→0) | 11,430 |
| after dropping empty / whitespace-only URLs | 11,430 |
| after dropping structurally invalid URLs (no host, no dot, whitespace) | 11,430 |
| after dropping exact duplicate URLs | **11,429** (−1) |
| after dropping URLs carrying conflicting labels | 11,429 |

Final: **11,429 URLs — 5,715 benign (50.0%) / 5,714 phishing (50.0%)**.
Essentially balanced, so the majority-class baseline is 0.5000.

## 4. Data-leakage audit (`src/audit_dataset.py`)

The original synthetic dataset was separable by a *single* feature, which is
why every algorithm scored a meaningless 1.000. The audit makes that failure
mode impossible to miss: it reports, for every feature, how well that one
feature alone separates the classes.

**Strongest single feature on the real data: `num_digits`, AUC 0.6975.**
Nothing is anywhere near 1.00 — no leakage.

The three features called out as prime leak suspects all came back clean:

| Suspect | Result |
|---|---|
| **`is_https`** | benign 44.5% HTTPS vs phishing 33.3% HTTPS → **AUC 0.5559**, barely better than a coin flip. (On the synthetic data this one feature gave 100% accuracy.) |
| **TLD** | 343 distinct TLDs; `.com` is 48.2% phishing, i.e. almost exactly the base rate. Memorising the TLD alone caps out at **0.6074 in-sample**. |
| **Suspicious keywords** | present in 25.6% of phishing vs 2.9% of benign URLs → real but weak signal, **AUC 0.6150**. |

One structural caveat the audit *does* flag: 41.7% of rows share a registered
domain with another row (`blogspot.com` appears 383 times). A purely random
split can therefore put sibling URLs from one domain on both sides and let a
model memorise domains. §6 reports a domain-disjoint split to quantify that.

## 5. Results

> **Read the domain-aware numbers, not the random-split numbers.** On this
> dataset a random 80/20 split puts a registered domain seen in training into
> 46.0% of test rows. Section 5b quantifies what that is worth.

### 5b. Three validation protocols (`models/validation_comparison.csv`)

Same data, same seeds, same estimators -- only the split rule differs, so the
difference between columns *is* the leakage.

5-fold domain-aware CV, mean +/- std across folds
(`models/cv_results.csv`, full 122-feature set):

| Algorithm | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---|---|---|---|---|---|---|
| Decision Tree | 0.8188 ± 0.0145 | 0.7481 ± 0.0232 | 0.7978 ± 0.0520 | 0.7711 ± 0.0233 | 0.8446 ± 0.0162 | 0.1682 ± 0.0266 | 0.2022 ± 0.0520 |
| Random Forest | 0.8751 ± 0.0198 | 0.8223 ± 0.0368 | 0.8628 ± 0.0300 | 0.8415 ± 0.0238 | 0.9513 ± 0.0090 | 0.1172 ± 0.0297 | 0.1372 ± 0.0300 |
| **XGBoost** | 0.8882 ± 0.0227 | 0.8703 ± 0.0318 | 0.8336 ± 0.0517 | 0.8508 ± 0.0329 | **0.9583 ± 0.0106** | 0.0779 ± 0.0229 | 0.1664 ± 0.0517 |
| Logistic Regression | 0.8471 ± 0.0165 | 0.7859 ± 0.0210 | 0.8275 ± 0.0456 | 0.8055 ± 0.0241 | 0.9220 ± 0.0146 | 0.1407 ± 0.0191 | 0.1725 ± 0.0456 |

Cost of the random split, in ROC-AUC (random 80/20 minus domain-disjoint):

| Algorithm | random 80/20 | domain-disjoint | domain CV (mean) | leakage cost |
|---|---|---|---|---|
| Decision Tree | 0.9077 | 0.7851 | 0.8446 | **+0.1226** |
| Random Forest | 0.9752 | 0.9265 | 0.9513 | +0.0487 |
| XGBoost | 0.9799 | 0.9372 | 0.9583 | +0.0427 |
| Logistic Regression | 0.9454 | 0.8982 | 0.9220 | +0.0472 |

**Model selection.** The served model is whichever wins on *mean* domain-aware
CV ROC-AUC -- threshold-free, and averaged over folds rather than decided by
one partition's luck. Random Forest is not privileged. XGBoost currently wins
(0.9583 vs 0.9513), but the 0.0070 gap is smaller than the 0.0106 fold-to-fold
std, so `train.py` prints an explicit note that this evidence does not resolve
the ordering of XGBoost and Random Forest.

### 5c. Random 80/20 (kept for comparability with v1)

Reproducible **stratified 80/20** split (`random_state=42`). Hyperparameters
were **not** tuned against the test set; these are plain held-out numbers.
These are the optimistic numbers -- they are retained so older figures stay
comparable, not because they estimate deployed performance.

### `models/algorithm_comparison.csv`

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.8364 | 0.8665 | 0.7953 | 0.8294 | 0.9011 |
| **Random Forest** | **0.8906** | 0.9093 | **0.8679** | **0.8881** | 0.9548 |
| XGBoost | 0.8885 | **0.9119** | 0.8600 | 0.8852 | **0.9561** |

Best by F1-score: **Random Forest** (the model marked ★ in the web UI and used
for the extension's score).

### `models/confusion_matrices.csv`

| Algorithm | TN | FP | FN | TP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 1003 | 140 | 234 | 909 | 0.1225 | 0.2047 |
| Random Forest | 1044 | 99 | 151 | 992 | 0.0866 | 0.1321 |
| XGBoost | 1048 | 95 | 160 | 983 | 0.0831 | 0.1400 |

### Feature importance (`models/feature_importance.csv`)

Top signals for the best model (Random Forest):

```
num_digits               0.109
entropy                  0.106
num_subdomains           0.091
url_length               0.089
path_length              0.083
host_length              0.077
num_slashes              0.069
num_hyphens              0.068
suspicious_word_count    0.062
num_dots                 0.057
```

Note `is_https` now ranks **16th of 21 at 0.017 importance**. On the synthetic
data it was effectively the only feature the models used.

### 5d. Does the feature work help? (`models/baseline_vs_improved.csv`)

`src/compare_baseline.py` re-evaluates all three feature sets on **today's**
data, with the same seeds, the same canonical de-duplication, the same domain
groups and the same folds -- so the feature set is the only variable.

The archived v1 numbers are preserved separately as
`models/baseline_archived.csv`. They are **not** mixed into the table below:
they came from a different dataset snapshot, exact-string de-duplication and a
different split rule, so comparing them directly would attribute three changes
to the features alone.

Domain-aware CV ROC-AUC, mean (± fold std):

| Algorithm | legacy (21) | structural (86) | full (122) |
|---|---|---|---|
| Decision Tree | 0.8650 ± 0.0294 | 0.8638 ± 0.0169 | 0.8446 ± 0.0162 |
| Random Forest | 0.9289 ± 0.0186 | 0.9515 ± 0.0107 | 0.9513 ± 0.0090 |
| XGBoost | 0.9315 ± 0.0201 | 0.9572 ± 0.0128 | 0.9583 ± 0.0106 |
| Logistic Regression | 0.8609 ± 0.0256 | 0.9199 ± 0.0143 | 0.9220 ± 0.0146 |

Two separate readings:

* **The structural features move the number** for the three non-tree-stump
  models, by more than the fold std (XGBoost +0.0257, Random Forest +0.0226,
  Logistic Regression +0.0590 over legacy).
* **The hashed n-gram block does not.** `full` minus `structural` is
  -0.0002 to +0.0021 ROC-AUC, every one of them well inside the fold std. On
  the Decision Tree it is actively negative (-0.0192). The n-grams are cheap
  and they are implemented, but on this corpus they are not carrying signal the
  structural features do not already have.

`full` remains the serving default because it is marginally ahead of
`structural` on all three headline metrics for the selected model (XGBoost) --
but that margin is inside the noise, and `--feature-set structural` is a
defensible choice that drops 36 columns for no measured loss.

## 6. Generalisation check: domain-disjoint split

Because 41.7% of URLs share a domain, `train.py` also re-runs the same three
models on a split where **no registered domain appears on both sides**
(`GroupShuffleSplit`, same seed). This is the harder, more realistic estimate —
it measures generalisation to domains never seen in training.

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.7456 | 0.8633 | 0.6703 | 0.7546 | 0.8090 |
| Random Forest | 0.8104 | 0.8763 | 0.7860 | 0.8287 | 0.8991 |
| XGBoost | 0.7876 | 0.8881 | 0.7278 | 0.8000 | 0.8963 |

Roughly **6–8 accuracy points lower** than the headline table. Both numbers are
real; the §5 table is the standard stratified-split protocol used in the
literature, and this one is the honest answer to "how does it do on a domain it
has never seen?" Quote both in any write-up.

Skip it with `python src/train.py --no-group-check`.

## 7. Real-world validation on unseen URLs

`src/validate_unseen.py` scores the trained models against a **completely
independent corpus** -- different collection, different era, no retraining.

**External corpus:** `shawhin/phishing-site-classification` (ODbL), a curated
subset of the widely used Kaggle *Phishing Site URLs* corpus (taruntiwarihp),
built from PhishTank submissions plus benign crawls.

3,000 rows -> 1 invalid, 2 duplicates and **4 URLs that also appear in training**
removed -> **2,993 genuinely unseen URLs** (1,499 benign / 1,494 phishing).
5.0% of registered domains are shared with training; these are **kept**, since
domain overlap is realistic in deployment.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.6559 | 0.6245 | 0.7791 | 0.6933 | 0.7138 |
| Random Forest | 0.7337 | 0.6869 | 0.8574 | 0.7627 | 0.8423 |
| **XGBoost** | **0.7564** | **0.7212** | 0.8347 | **0.7738** | **0.8585** |
| Consensus (>=2/3) | 0.7300 | 0.6830 | 0.8568 | 0.7601 | 0.8340 |

| Model | TN | FP | FN | TP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 799 | 700 | 330 | 1164 | 0.4670 | 0.2209 |
| Random Forest | 915 | 584 | 213 | 1281 | 0.3896 | 0.1426 |
| XGBoost | 1017 | 482 | 247 | 1247 | 0.3215 | 0.1653 |
| Consensus | 905 | 594 | 214 | 1280 | 0.3963 | 0.1432 |

**This is the number to quote as real-world performance**, not the 0.89 from §5.
Three findings matter:

1. **Accuracy falls from 0.89 to 0.73-0.76 across corpora.** In-corpus held-out
   metrics overstate deployment performance substantially.
2. **The ranking flips.** Random Forest wins in-corpus (F1 0.8881); **XGBoost
   generalises best** cross-corpus (F1 0.7738, AUC 0.8585). Selecting the best
   model on in-corpus F1 alone would have picked the weaker generaliser.
3. **False positives dominate.** FPR 32-40% versus FNR 14-17%: the models are
   far better at catching phishing than at leaving benign sites alone. The
   consensus vote does **not** help -- it tracks Random Forest (FPR 0.3963) and
   is worse than XGBoost alone.

The three models agreed on only **70.0%** of unseen URLs (versus near-total
agreement on the synthetic data), so the per-model meters in the web UI now
show genuine disagreement.

### Error analysis

Most confident **false positives** are long, query-heavy URLs on legitimate
sites -- exactly the shape the model learned to distrust:

```
0.970  sfgate.com/cgi-bin/object/article?f=/c/a/2011/06/09/BADE1JRSPI.DTL&object=...
0.968  litigation-essentials.lexisnexis.com/webcd/app?action=DocumentDisplay&crawlid=...
0.961  thecivilwarhomepagediscussion2824.yuku.com/topic/1265
```

Most confident **false negatives** are short, clean-looking phishing URLs with
no lexical tell at all -- the fundamental blind spot of URL-only detection:

```
0.052  technix.it/
0.059  intoaddedprio.ru
0.060  mail.motor-corp.com/Dir/WebHosting/YahooMail/email.html
```

Full error list: `models/unseen_errors.csv`.

### Confound: the external corpus has no URL schemes

Public URL corpora almost always strip the scheme (`www.example.com/path`), so
`is_https` is 0 for every external row. The script quantifies the cost by
re-scoring the internal corpus with schemes stripped the same way:

| Internal corpus, best model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| with scheme (as trained) | 0.9470 | 0.9454 | 0.9884 |
| scheme stripped | 0.8950 | 0.8873 | 0.9535 |

About 5 accuracy points of the drop are the missing scheme; the remaining ~14
points are **genuine corpus shift**. (These two rows are in-sample and
optimistic in absolute terms -- only the *gap* between them is meaningful.)

## 8. Tests

```bash
python -m unittest discover -s tests -v
```

27 tests covering `features -> training -> inference -> /api/scan -> /api/check`,
including: feature-vector shape and finiteness, a guard that feature extraction
opens **no network socket**, dataset schema and class balance, split
reproducibility, the `url_models.joblib` key contract the Flask API and
extension depend on, a regression guard that the Decision Tree has **not**
collapsed to a stump, both API endpoints (including CORS for the
`chrome-extension://` origin) and a check that `/api/scan` and `/api/check`
always return the same verdict.

## 9. Web app

```bash
python webapp/app.py
```

Open **http://127.0.0.1:5000** — each scan shows all 3 algorithms' scores as
separate meter bars, the best-performing model marked with a ★, a consensus
verdict, and the top contributing signals ranked by the best model's real
feature importances.

## 10. Real-time browser protection (Chrome extension)

A Chrome extension that checks **every site you navigate to** against the
trained model *before* the page loads, and shows a warning page instead of
letting a flagged phishing site open.

### How it works
1. You click a link (or type a URL and hit enter)
2. Before the page loads, the extension's background script sends the URL to
   the local Flask API (`/api/check`)
3. If the models flag it malicious, the extension redirects the tab to a
   warning page — you can go back to safety, or proceed anyway
4. If it's benign, browsing continues normally with no interruption

### Setup

**Step 1 — start the backend** (must stay running in a terminal):
```bash
python webapp/app.py
```

**Step 2 — load the extension in Chrome:**
1. Open Chrome and go to `chrome://extensions`
2. Turn on **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `extension/` folder from this project
5. The shield icon should appear in your toolbar

**Step 3 — try it:** visit a normal site (nothing happens), then type a URL
with phishing-like structure and you should see the warning page.

### Notes
- The backend must be running; if it isn't, the extension **fails open** (lets
  browsing continue) rather than blocking everything
- It only intercepts **main-frame navigation**, not sub-resources
- ⚠️ At a 0.5 threshold the model has an **~8.7% false-positive rate**, so it
  *will* occasionally block legitimate sites — see §12

## 8b. Website analysis layer (multi-layer detection)

URLGuard now has a second detector. The URL model reads the address; the
website model reads the retrieved page; a risk engine combines them.

```
URL model        -> URL-pattern risk      (always available)
Website model    -> page-pattern risk     (only when the page was fetched)
DNS / reputation -> additional evidence   (not held -- always "Not available")
                            |
                      Risk engine -> score + per-layer breakdown
```

### The guarantee that changed, stated plainly

Every earlier version could claim *"the URL is never fetched."* That is no
longer true of the whole system, so the claim was split rather than quietly
dropped:

| | contacts the address? |
|---|---|
| URL model, link interception, DNS layer | **No** — unchanged |
| Website analysis | **Yes** — and it is opt-in per scan |

`/api/meta` advertises `url_analysis_never_contacts_url: true` and
`website_analysis_contacts_url: true`. The old blanket `never_contacts_url`
flag was removed, and a test asserts it is gone: leaving it would have been a
false claim.

### Sandbox (`src/web_fetch.py`)

This is the one place that touches an untrusted address, so it is the one place
where a mistake turns a scanner into an attack tool.

* **No code execution.** HTML is parsed as a static document. No browser, no
  JavaScript engine. A `<script src=…>` is counted; it is never fetched or run.
* **No interaction.** GET only. No form submitted, no credential sent, no
  cookie stored or returned.
* **No SSRF.** Every hostname is resolved and *every* resolved address checked
  against loopback, private, link-local, multicast and reserved ranges — before
  connecting and again after each redirect. Redirects are followed manually
  because urllib's handler would bypass those checks.
* **Bounded.** 800 KB, 12 s, 5 redirects.
* **Announced.** The User-Agent identifies the request as an automated scan.

Verified: `127.0.0.1`, `169.254.169.254`, `10/8`, `192.168/16`, `::1`,
`0.0.0.0`, `file:`, `ftp:` and `data:` are all refused.

**Not mitigated, and documented rather than hidden:** DNS rebinding. Addresses
are validated at resolution time, so a server answering differently to the
validating and connecting lookups could still slip through. Closing it needs
connect-time address pinning with TLS SNI handling.

### Features (`src/web_features.py`) — 84, in 7 groups

structure (17) · credential (15) · destination (8) · resources (16) ·
concealment (6) · identity (13) · transport (9)

Brand matching uses a curated **brand → official domain** map, not substring
matching. `paypal-secure-verify.tk` contains "paypal" and is not PayPal; a
substring test called that a match, which is the textbook impersonation case
inverted.

### Dataset (`src/build_web_dataset.py`)

`URLGuard-Web-2026-10-04`: 9,000 URLs attempted → 2,780 retrieved → **2,276
usable pages** (1,209 benign / 1,067 malicious). Splits are **inherited** from
the URL dataset, so no registered domain can land on one model's train side and
the other's test side.

**The dominant limitation, measured:** retrieval succeeded for **75.3% of
benign URLs but only 19.8% of malicious ones** — phishing sites get taken down.
The surviving malicious sample is therefore not a random sample of phishing.
504 further pages were dropped as parking/suspension notices, which would
otherwise have taught the model that phishing pages look like parking pages.
Labels are **inherited from the URL** (weak labelling), not independent
page-level review.

### Website model (`src/train_web.py`)

Five models compared on page features only — deliberately no URL features, or
it would relearn the URL model and the engine would be blending a model with a
copy of itself.

Selected **URLGuard-WEB-XGB-v1.0**, threshold 0.95. Holdout, scored once:

```
Accuracy 0.7665  Precision 0.9720  Recall 0.5591  F1 0.7099
ROC-AUC  0.9448  PR-AUC    0.9480  FPR    0.0169  FNR 0.4409
Confusion  TN=175  FP=3  FN=82  TP=104        3.80 ms/page
```

High precision, low recall, by design: a corroborating layer that cries wolf is
worse than one that stays quiet. Importance by group: resources 0.33,
structure 0.17, concealment 0.17, credential 0.13.

### Risk engine (`webapp/risk_engine.py`) — two bugs it exists because of

**1. Averaging two scales was wrong.** The models have different thresholds
(URL 63/100, website 95/100), so the same number means different things.
`example.com` scored 94 on the website model — *below* its own threshold, i.e.
"not malicious" — and averaging still dragged the combined score from 7 to 42.
The website layer now adjusts the URL score by its distance from **its own**
threshold: nothing at threshold, up to +35 above, down to −12 below.

The push is asymmetric deliberately. A page can look clean because it is clean,
or because it is JavaScript-rendered, cloaked against scanners, or already
taken down. A clean page earns a small reduction, never a clean bill of health.

**2. Single-feature floors were wrong.** Flooring on brand mismatch alone rated
**Wikipedia's article about phishing at 60/100** — the article mentions brands,
so "brand claimed, not that brand's domain" was true and irrelevant.
Impersonation is a brand claim **plus** a credential form. Every floor now
requires a combination.

Measured after both fixes:

| URL | URL-only | Website | Final |
|---|---:|---:|---:|
| Wikipedia's phishing article | 15 | 1 | **3** (was 60) |
| example.com | 7 | 94 (below thr) | **7** (was 42) |
| github.com/explore | 69 | 1 | **57** — page evidence corrects a URL false positive |
| live phishing login pages | 90–99 | 97–99 | **98–99** (+14 to +28) |

**A layer that produced no evidence never acts as evidence.** If the page could
not be fetched the website row reads *Not available* with the reason, and the
score is the URL score alone — "we could not look" and "we looked and found
nothing" are rendered differently.

### Clients

The web scanner gains an opt-in **"Also analyse the webpage"** checkbox and a
**Detection breakdown**. Android sends the request to the backend and never
downloads or renders an arbitrary page on the device. DNS-only results are
labelled exactly *"Hostname-only assessment; webpage content was
unavailable."*

```bash
python src/build_web_dataset.py --limit 9000 --workers 24
python src/train_web.py
```

## 9a. Version 3: multi-source dataset, model selection, auth

v3 replaces the single-corpus pipeline with a provenance-tracked, multi-source
dataset and a model chosen on more than accuracy. `src/train.py` and the v2
bundle are unchanged and still reproduce their published numbers.

### Sources (`src/datasets/sources.py`)

| Source | Rows kept | Label trust | Timestamps | Licence |
|---|---:|---|---|---|
| PhishTank (verified only) | 6,492 | verified | yes | Free research use, attribution |
| Tranco top sites | 5,629 | **presumed** | no | Free research; cite the paper |
| Existing project corpus | 3,571 | reported | no | As upstream |
| Phishing.Database | 2,976 | reported | no | MIT |
| Existing hard negatives | 2,595 | reported | no | As upstream |
| URLhaus (full dump) | 539 | verified | yes | CC0 |
| OpenPhish community | 44 | reported | no | Community feed, attribution |

**21,846 rows / 19,074 registered domains**, balanced 10,923 / 10,923. Every
row carries its source, a collection timestamp and a label-trust level.

Tranco is labelled **presumed**, never "verified benign": it ranks domains by
*popularity*, which is not a safety judgement. Nothing downstream describes
those rows as confirmed safe.

### Leakage controls (`src/build_dataset.py`)

1. **Canonical de-duplication** — 3,066 rows removed.
2. **Domain grouping** — train/val/holdout share **0 registered domains**; a
   test asserts it.
3. **Per-domain cap** (5) — 40,126 rows removed so one campaign cannot dominate.
4. **Holdout** scored exactly once, after all decisions.
5. **Temporal split** drops domains straddling the cut-off.
6. **Structural-confound measurement**, below.

### The failure this pipeline is really about

Multi-source data introduces a trap that domain grouping does not catch: each
feed has a *shape*, and shape can stand in for the label. The build measures
single-property AUC and refuses to proceed quietly.

Three artifacts were found this way, each introduced by one line that
synthesised a URL from a Tranco domain:

| Artifact | Symptom | Cause |
|---|---|---|
| `has_path` AUC **0.92** | benign 11% vs malicious 96% had a path | Tranco gives bare domains |
| `is_https` 5x malicious | benign 9% vs malicious 47% https | `http://` prefix was invented |
| `path_ratio_of_url` SHAP **+4.55** | `github.com` scored **0.9995** | trailing `/` gave benign roots a path that real phishing roots lack |

Tranco publishes a domain and a rank. It says nothing about scheme or path, so
`parse_tranco` now emits the bare domain and invents neither. Scheme is then
assigned at the **same rate in both classes**, because across these feeds
scheme is a storage convention rather than evidence.

After stratification and scheme balancing:

```
url_length 0.6077   path_length 0.5754   query_length 0.5021
is_https   0.5000   has_path    0.5000      (0.50 = no information)
```

**The stratification ratio was chosen by measurement, not taste.** Looser
ratios keep more rows but re-admit the confound, and that costs real accuracy:

| ratio | strongest structural AUC | rows | holdout FPR | FPs on 17 well-known sites |
|---|---|---:|---|---:|
| **1.0** | **0.61** | 21.8k | **0.0262** | **5** |
| 2.0 | 0.76 | 32.6k | 0.0402 | 7 |
| 3.0 | 0.82 | 42.5k | — | — |
| 6.0 | 0.89 | 67.2k | — | — |

Strict parity keeps the fewest rows and still wins. Removing the confound
matters more here than dataset size.

### Models (`src/train_v3.py`)

Validation is domain-disjoint; the holdout is untouched until the end.

| Model | Acc | Prec | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR | ms/URL |
|---|---|---|---|---|---|---|---|---|---:|
| Logistic Regression | 0.9164 | 0.9275 | 0.9036 | 0.9154 | 0.9633 | 0.9663 | 0.0708 | 0.0964 | 1.18 |
| Decision Tree | 0.9054 | 0.9063 | 0.9044 | 0.9053 | 0.9272 | 0.9218 | 0.0936 | 0.0956 | 0.70 |
| Random Forest | 0.9304 | 0.9395 | 0.9201 | 0.9297 | 0.9799 | 0.9825 | 0.0593 | 0.0799 | 10.79 |
| **XGBoost** | **0.9400** | 0.9484 | 0.9308 | **0.9395** | **0.9835** | **0.9859** | **0.0508** | 0.0692 | 6.57 |
| Neural Net (MLP) | 0.9239 | 0.9165 | 0.9329 | 0.9247 | 0.9724 | 0.9753 | 0.0851 | 0.0671 | 1.07 |

Selection is **not** by accuracy: candidates must first reach FPR ≤ 0.05 at
some threshold, and models within one fold-standard-deviation of the AUC leader
are treated as tied and decided on inference cost. That rule picked XGBoost
over Random Forest for the hostname model on a 0.0009 AUC gap — which also
avoided a 34 MB ONNX export.

**Holdout, scored once** (`URLGuard-XGB-v3.0`, threshold 0.63):

```
Accuracy 0.9477  Precision 0.9728  Recall 0.9220  F1 0.9467
ROC-AUC  0.9865  PR-AUC    0.9886  FPR    0.0262  FNR 0.0780
Confusion  TN=1601  FP=43  FN=130  TP=1536        6.57 ms/URL
```

### Temporal validation

Train on URLs seen before 2026-07-30, test on newer ones (12,575 / 5,335):

| Model | ROC-AUC | PR-AUC | F1 | FPR | Δ vs validation |
|---|---|---|---|---|---|
| **XGBoost** | 0.9814 | 0.9784 | 0.9298 | 0.0245 | **−0.0021** |
| Random Forest | 0.9786 | 0.9762 | 0.9207 | 0.0352 | −0.0013 |
| Neural Net | 0.9671 | 0.9618 | 0.9004 | 0.0444 | −0.0053 |
| Logistic Regression | 0.9640 | 0.9517 | 0.8865 | 0.0692 | +0.0007 |
| Decision Tree | 0.9155 | 0.8929 | 0.8796 | 0.0772 | −0.0117 |

Performance on newer URLs is essentially unchanged.

**Read this with the stated caveat.** Only the malicious feeds publish dates;
the benign sources are undated lists. The malicious side is time-ordered, the
benign side is domain-disjoint but not. Inventing benign dates (e.g. the
collection timestamp) would have made every benign row share one date and
turned the experiment into nonsense. So this measures drift in *phishing
campaigns*, which is where drift actually happens — not a full both-sides
temporal shift. The manifest records the asymmetry and a test asserts it.

### Authentication (`webapp/auth.py`)

Argon2id password hashing; short-lived JWT access tokens with rotating refresh
tokens stored **hashed** server-side; per-user scan history that no other
account can read; rate limiting on login, registration and reset.

The signing secret comes from `URLGUARD_SECRET` or is generated once into
`instance/secret.key` (0600). There is no hard-coded fallback — a shipped
default is a shipped master key, and the Android client would carry it.

Email enumeration is closed off: failed logins, duplicate registrations and
password-reset requests return the same shape whether or not the address
exists. Reading another user's scan returns **404, not 403**, so the endpoint
is not an existence oracle.

```bash
python src/build_dataset.py                 # -> data/urlguard_dataset.csv + manifest
python src/train_v3.py                      # -> models/v3/
python src/train_v3.py --scope hostname     # -> models/v3_hostname/
python src/export_onnx.py --bundle models/v3/url_models.joblib
python webapp/app.py
```

## 9b. Frontend: conservative risk presentation

The web UI (`webapp/`) and the Android screens present the same model through a
deliberately cautious vocabulary. A score is a **risk band**, never a verdict.

### The four kinds of statement, kept separate

`webapp/assessment.py` refuses to blur these together:

| | what it is | where it comes from |
|---|---|---|
| **Observation** | a fact about the URL text | the 122 features |
| **Signal** | an observation *empirically* associated with phishing | benign-corpus percentile + lift |
| **Model assessment** | risk score, band, confidence | the classifier |
| **External fact** | DNS, IP, domain age, reputation | **none held — reported as "Not available"** |

### Risk bands

| score | band | wording |
|---|---|---|
| 0–30 | **Low risk** | "No significant suspicious characteristics were detected." |
| 31–70 | **Suspicious** | "Some characteristics associated with potentially harmful URLs were detected." |
| 71–100 | **High risk** | "Multiple characteristics … Exercise caution before proceeding." |

Low risk is never called *safe*; high risk is never called *confirmed malicious*.
A test (`TestAssessmentLayer`) fails the build if any band text contains
"definitely", "confirmed", "guarantee", "100%", "is malicious" or "is safe".

### Two numbers that could easily have lied

**The score never displays 0 or 100.** A gradient-boosted ensemble never outputs
exactly 0 or 1, but rounding 0.9997 to "100/100" tells the user the system is
certain. Probabilities strictly inside (0,1) are clamped to 1..99.

**Confidence is decision margin, not class probability.** The obvious definition
— the probability of the predicted class — breaks here because the decision
boundary is 0.70, not 0.50: a URL scoring 0.55 would be reported as "benign,
45% confidence", which reads as a malfunction rather than as the borderline
case it is. Confidence is therefore *distance from the threshold*, so
`github.com/explore` (raw 0.550) correctly reads **Suspicious · 21% ·
Borderline** while `wikipedia.org` reads **Low risk · 98.7% · Decisive**.

### Signals are earned, not assumed

`src/build_signal_thresholds.py` computes each threshold as a percentile of the
**benign** corpus, so the UI can state what a value is unusual *relative to*
("longer than 95% of known-benign URLs") instead of asserting "unusually long"
from a hard-coded number.

It also measures **lift** — how much more often phishing exceeds the threshold
than benign does — and that produced an uncomfortable result worth stating:

| measurement | lift | shown as |
|---|---:|---|
| suspicious keyword count | 7.54 | **risk signal** |
| subdomain depth | 6.42 | **risk signal** |
| longest subdomain label | 5.90 | **risk signal** |
| hostname length | 5.85 | **risk signal** |
| dots in URL | 5.30 | **risk signal** |
| hostname entropy | 5.19 | **risk signal** |
| **URL length** | **0.95** | neutral observation |
| **special characters** | **0.25** | neutral observation |
| **percent-encoding** | **0.13** | neutral observation |
| **hyphen count** | **0.05** | neutral observation |

Eight of the sixteen intuitive "suspicious" properties are **not** more common
among phishing URLs in this corpus; hyphens are strongly *inverted*. The brief
asked for "⚠ Unusually long URL" as an example signal — on this data that would
be manufacturing evidence, so URL length is reported as a plain observation with
its percentile, and only measurements with lift ≥ 1.2 are rendered as warnings.

### Explanations come from the model, not from a template

The "Why did the system assign this risk level?" panel uses **XGBoost SHAP
contributions** (`pred_contribs=True`) for that specific URL — the model's own
attribution, which reconstructs its score exactly. Where SHAP is unavailable the
panel says *"Detailed model explanation is not available for this scan."* rather
than inventing a cause.

### What is deliberately missing

DNS resolution, IP address, domain age and reputation are **never** shown,
because obtaining them means contacting the network and this project's whole
guarantee is that it does not. They render as *Not available* with the reason
attached. The `scope: "hostname"` mode additionally states *"Only the hostname
was available for this assessment"* and warns that the hostname-only model is
substantially weaker (ROC-AUC 0.764 vs 0.958).

### Inputs the model is not qualified to judge

`javascript:`, `mailto:`, `ftp://` and `file://` URIs are **refused**, not
scored. The classifier was trained on http/https web URLs; returning "LOW RISK"
for `javascript:alert(1)` would be a confident answer to a question it cannot
answer. Host:port forms (`example.com:8080/x`) are still accepted.

### Monitoring

`/api/stats` and `/api/history` report only scans this installation actually
performed. An empty store returns `sufficient_data: false` and the UI shows
**"Insufficient data"** — never zeroes dressed up as findings, never sample rows.

### New endpoints (all additive)

`/api/scan` keeps every field the Chrome extension and the older tests rely on
and gains `assessment`, `detection_ms`, `scanned_at`, `scope` and `subject`.
New: `GET /api/history`, `GET /api/history/<id>`, `DELETE /api/history`,
`GET /api/stats`, `GET /api/meta`.

```bash
python src/build_signal_thresholds.py    # -> models/signal_thresholds.json
python webapp/app.py
```

## 10b. Android app (`android/`) -- on-device, background protection

An Android build of the same detector. The Chrome extension cannot be ported
directly: Android has no `webNavigation` API, and **no app can see another
browser's navigations** without either a VPN or an AccessibilityService. The app
therefore runs two layers with very different capabilities, and says so on its
own main screen rather than quoting one accuracy figure for both.

### The two layers

| | Link layer | DNS layer |
|---|---|---|
| Trigger | a link is **tapped** (SMS, WhatsApp, Gmail, ...) | any DNS lookup, any app |
| Mechanism | `http`/`https` intent filter | `VpnService`, DNS only |
| Input | the **full URL** | the **hostname only** |
| Model | 122 features, XGBoost | retrained hostname-only, XGBoost |
| Domain-aware CV ROC-AUC | **0.958 ± 0.011** | **0.764 ± 0.021** |
| Serving threshold | 0.70 | 0.95 |
| At that threshold | FPR 0.045, recall 0.755 | FPR 0.0033, recall 0.169 |

The DNS layer is weak **because HTTPS hides the path and query**. A DNS packet
carries a hostname and nothing else, so the full model's strongest features
(`path_depth`, `query_entropy`, `max_param_value_length`, ...) are all
identically zero at serving time. Rather than serve the full model inputs it was
never trained on, `src/train.py --url-transform hostname` retrains honestly on
hostnames and the result is reported as what it is: much worse.

That is also why its threshold is 0.95 rather than the 0.65 the training rule
picked. The rule targets FPR ≤ 0.05, which is fine for a dismissible screen but
not for a system-wide blocker -- at 0.05, roughly **1 benign domain in 20** stops
resolving and the phone feels broken. 0.95 trades away more than half the recall
to wrongly block about 1 domain in 300.

**Read the link layer as the real defence and the DNS layer as a coarse net.**
Typing a URL directly into Chrome is only covered by the weak layer.

### On-device inference

`models/url_models.joblib` → ONNX (`src/export_onnx.py`) → ONNX Runtime Mobile.
Nothing is sent off the device; the URL is still never fetched or resolved.

The export is **verified, not assumed**: every one of the 11,429 corpus URLs is
re-scored through both the original estimator and ONNX Runtime.

```
max |delta|           4.768e-07     (float32 rounding)
verdict flips @0.70   0  (0.0000%)
```

### Feature parity

Serving the validated model requires feeding it the exact numbers it was trained
on, so `src/features.py` was ported to Kotlin (`UrlFeatures.kt`) **and the port
is asserted against Python**, not trusted. `src/export_onnx.py` emits 625 golden
vectors (600 sampled from the corpus + 25 hand-picked parser breakers) and the
Android unit tests compare all 122 features on each, 76,250 comparisons, plus
the end-to-end score.

The port deliberately reproduces Python's quirks, because the model was trained
on them -- `features.py` computes `host = netloc.split(":")[0]`, so
`http://user:pass@evil.com` has host `"user"`, and `[::1]:8080` has port `""`.
Two real bugs were caught this way and fixed. The public-suffix list is **dumped
from the live Python extractor** (`assets/public_suffix_list.txt`) rather than
fetched separately, so it cannot drift from the list used in training.

### Build

```bash
python src/export_onnx.py                       # full model  -> assets/
python src/train.py --url-transform hostname --model-dir models/hostname
python src/export_onnx.py --bundle models/hostname/url_models.joblib \
    --model XGBoost --override-threshold 0.95 \
    --out-dir android/app/src/main/assets/hostname
cd android && gradle :app:testDebugUnitTest :app:assembleRelease
```

Output (one APK per ABI -- ONNX Runtime's native library is 23-39 MB *per ABI*,
so a universal APK is 142 MB):

```
app-arm64-v8a-release.apk     37.3 MB    <- almost all modern phones
app-armeabi-v7a-release.apk   28.1 MB    <- older 32-bit devices
app-x86_64-release.apk        43.4 MB    <- emulator only
```

Signed with a **throwaway** keystore (`app/urlguard-demo.keystore`, password
`urlguard`) committed so the build reproduces anywhere. It is not a publishing
key; generate a fresh one, kept out of the repo, before any real distribution.

### Permissions, and what is deliberately not requested

`INTERNET` is used **only** to relay DNS queries upstream once judged safe --
never to fetch a classified URL. The VPN routes **only** its own virtual DNS
address (`10.111.222.2/32`), so ordinary traffic keeps its normal path and is
never read: the guard sees DNS questions, not content.

No `AccessibilityService`. It would give full URLs in any browser and true
background coverage, but it reads the screen of **every** app, and Google Play
bans it for non-accessibility use. The accuracy was not worth that trade.

Every failure path **fails open** -- a model that cannot load, a malformed
packet, an upstream timeout all result in the link or lookup proceeding. A
security tool that breaks connectivity when it malfunctions gets uninstalled,
and then protects nothing.

### Android test suite (24 tests, `gradle :app:testDebugUnitTest`)

* `FeatureParityTest` -- 122 features × 625 URLs vs Python; feature *order* too
* `InferenceParityTest` -- end-to-end scores vs Python, 0 verdict differences
* `HostnameModelParityTest` -- the DNS model, on a fixture of bare hostnames
  (what a DNS query actually carries), plus a check that everyday hosts are not
  blocked at the serving threshold
* `DnsPacketsTest` -- IPv4/UDP/DNS parsing, NXDOMAIN construction, and
  one's-complement checksum validation (a checksum bug here would present as
  "the user's internet is broken", so it is tested directly)
* `DnsScopeTest` -- regression tests for the reverse-DNS false positives below

### Verified on a running device (API 34 emulator)

| check | result |
|---|---|
| App launches, model loads | OK |
| Score `http://secure-paypal-login.tk/verify/9231` | **0.9997** — identical to the Flask API |
| Tapped phishing link | → `WarningActivity` (browser never opened) |
| Tapped benign link | → handed off to Chrome |
| VPN tunnel established | `tun0` up at `10.111.222.1/32` |
| DNS: `secure-paypal-login.tk` (0.99) | blocked, `unknown host` |
| DNS: `google.com` (0.07), `wikipedia.org` (0.09) | resolved and reachable |

**Two real bugs were found only by running it**, both fixed:

1. **Benign links went nowhere.** Once the app holds the BROWSER role, querying
   for a handoff target *without* `CATEGORY_BROWSABLE` resolves through the
   preferred-activity machinery and returns only this app — so no browser was
   ever found. Fixed by setting the category explicitly, querying with
   `MATCH_ALL`, and falling back to a chooser.

2. **Reverse-DNS lookups were being blocked.** `224.166.102.103.in-addr.arpa`
   scored 0.97 and was blocked: the hostname model sees a long digit-heavy name
   and reasonably calls it suspicious. But a PTR query is a *reverse* lookup of
   an address that already resolved — blocking one protects nobody and breaks
   reverse DNS. The DNS layer now classifies only `A`/`AAAA` queries and skips
   `*.arpa` plus the RFC 6761/8375 special-use names.

The second one is worth dwelling on: the model was not malfunctioning, it was
answering a question that should never have been asked of it. Scope is part of
the system, not an afterthought.

## 11. Why the synthetic results were meaningless

Kept as a documented negative result, since it is a more interesting finding
than three perfect scores. Reproduce with
`python src/make_sample_data.py && python src/train.py --data data/sample_urls.csv`.

On `data/sample_urls.csv` all three algorithms scored **1.000 on every metric**.
Inspecting the saved models showed why:

- the **Decision Tree** collapsed to **depth 1** — a single split on
  `is_https <= 0.5`, both children perfectly pure (gini 0.0000, 400/400)
- **XGBoost** ran 300 rounds but only **63** produced a split (all on
  `is_https`); the other **237 trees were bare leaves** contributing nothing
- the **Random Forest** rooted its trees on **15 different features** and still
  hit 1.000 — because the generator built benign and phishing URLs from
  disjoint templates, so hyphen count, keyword count and TLD length were *each*
  independently sufficient

That is textbook **shortcut learning**: the dataset, not the algorithms, was
doing the work. After retraining on the real corpus the same three models are
non-degenerate — Decision Tree depth 12 / 867 nodes, Random Forest trees
averaging 1,645 nodes, XGBoost with **zero** no-op trees and 20 of 21 features
actually used in splits.

## 12. Known limitations

1. **Lexical features only.** No page content, WHOIS, DNS, hosting or
   reputation signals. This is a deliberate safety trade-off (nothing is ever
   fetched), but it caps achievable accuracy.
2. **False positives are the dominant failure mode.** 8.7% FPR in-corpus,
   but **32-40% on the unseen external corpus** (§7). The "legitimate" training
   class is dominated by Alexa/Tranco **homepages**, so long query-heavy paths
   on real sites look unfamiliar. Spot-checking 15 well-known sites, 4 were
   flagged (`github.com/python`, `x.com/home`, `pypi.org/project/pandas/`,
   `docs.python.org/3/library/re.html`). The extension's 2-of-3 consensus does
   **not** fix this -- it is worse than XGBoost alone. **The extension is not
   fit for unsupervised everyday browsing at the default 0.5 threshold.**
3. **Collection-era bias.** The corpus was collected in 2020–21. Phishing URL
   style drifts, so performance on 2026 live feeds will be lower.
4. **Artificially balanced 50/50.** Real traffic is overwhelmingly benign, so
   the precision here is optimistic relative to deployment.
5. **Domain repetition.** 41.7% of rows share a domain; see §6 for the
   domain-disjoint numbers, which are 6–8 points lower.
6. **Untuned hyperparameters.** Carried over unchanged so the retrain is a
   like-for-like comparison. XGBoost has no `early_stopping_rounds` and no
   `scale_pos_weight` (harmless at 50/50, but it should be set before using a
   skewed dataset).
7. **`models/url_models.joblib` is ~40 MB** (300 real Random Forest trees),
   up from 0.6 MB when the trees were stumps.
8. **Model selection is corpus-dependent.** `train.py` picks the best model by
   in-corpus F1 (Random Forest), but XGBoost generalises better to the unseen
   corpus. The Flask API's `score` field and the extension's displayed score
   both use the in-corpus winner.
9. **Predictions are stable to ~1e-15, not bit-identical.** Random Forest
   averages 300 trees with `n_jobs=-1`, and floating-point addition is not
   associative, so repeated calls can differ by 1 ULP. Verdicts never change.
10. **No live-feed evaluation.** Validation uses an archived corpus, not a
   current PhishTank/OpenPhish feed. Project policy is that URLs are never
   fetched, so live-feed testing would still be string-only.
