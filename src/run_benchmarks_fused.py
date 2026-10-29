"""
Hard-benign and adversarial suites against the FUSED pipeline (URL + offline
threat intel + popularity), compared against URL-only.

The website layer is excluded here on purpose: these suites must not depend on
whether 60 third-party sites happen to be up, and must stay fast enough to run
in CI. Both configurations therefore see exactly the same evidence except the
reputation layer, so the difference is attributable to it.

Usage: python src/run_benchmarks_fused.py
"""
import json, os, sys, time
import joblib, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "webapp"))
from features import extract_features, FEATURE_NAMES  # noqa: E402
import benchmarks as B  # noqa: E402
import threat_intel  # noqa: E402
import risk_engine  # noqa: E402

BAND_THRESHOLD = 71     # "High risk" band; the UI's action point


def main():
    b = joblib.load("models/v3/url_models.joblib")
    model = b["models"][b["best_model_name"]]
    thr = float(b["threshold"])
    fn = b["feature_names"]

    def url_scores(urls):
        X = pd.DataFrame([extract_features(u) for u in urls],
                         columns=FEATURE_NAMES)[fn]
        return model.predict_proba(X)[:, 1]

    def evaluate(urls, use_intel):
        raw = url_scores(urls)
        out = []
        for u, p in zip(urls, raw):
            url_score = min(99, max(1, int(round(p * 100))))
            intel = threat_intel.lookup_all(u) if use_intel else None
            fused = risk_engine.combine(url_score, web=None, intel=intel)
            out.append(fused["risk_score"])
        return out

    print("=" * 78)
    print("BENCHMARKS -- URL-only vs URL+reputation fusion")
    print("=" * 78)

    hb_urls = [u for u, _ in B.ALL_HARD_BENIGN]
    hb_cats = [c for _, c in B.ALL_HARD_BENIGN]
    base = evaluate(hb_urls, False)
    t0 = time.perf_counter()
    fused = evaluate(hb_urls, True)
    lat = (time.perf_counter() - t0) / len(hb_urls) * 1000

    def rows(scores):
        return [{"url": u, "category": c, "score": s,
                 "flagged": s >= BAND_THRESHOLD, "expect": "benign"}
                for u, c, s in zip(hb_urls, hb_cats, scores)]

    rb, rf = rows(base), rows(fused)
    nb = sum(r["flagged"] for r in rb)
    nf = sum(r["flagged"] for r in rf)
    print(f"\n[1] Hard benign ({len(rb)} well-known sites), flagged at "
          f">= {BAND_THRESHOLD}/100")
    print(f"    URL only        {nb:>3}/{len(rb)}  ({nb/len(rb):.1%})")
    print(f"    + reputation    {nf:>3}/{len(rf)}  ({nf/len(rf):.1%})")
    sb, sf = B.summarise(rb), B.summarise(rf)
    print(f"\n    {'category':<20}{'url-only':>10}{'fused':>8}")
    for c in sorted(sb):
        print(f"    {c:<20}{sb[c]['incorrect']:>5}/{sb[c]['cases']:<4}"
              f"{sf[c]['incorrect']:>4}/{sf[c]['cases']:<3}")
    still = [r for r in rf if r["flagged"]]
    if still:
        print("\n    still flagged:")
        for r in sorted(still, key=lambda r: -r["score"]):
            print(f"      {r['score']:>3}  {r['url']}")

    cases = B.default_adversarial()
    aurls = [c["url"] for c in cases]
    ab, af = evaluate(aurls, False), evaluate(aurls, True)

    def arows(scores):
        return [{**c, "score": s, "flagged": s >= BAND_THRESHOLD}
                for c, s in zip(cases, scores)]

    rab, raf = arows(ab), arows(af)

    def wrong(rs):
        return sum(1 for r in rs if (r["expect"] == "benign") == r["flagged"])

    print(f"\n[2] Adversarial ({len(cases)} transformed URLs)")
    print(f"    URL only        {wrong(rab):>3}/{len(cases)} incorrect")
    print(f"    + reputation    {wrong(raf):>3}/{len(cases)} incorrect")
    sab, saf = B.summarise(rab), B.summarise(raf)
    print(f"\n    {'family':<24}{'url-only':>10}{'fused':>8}")
    for f in sorted(sab):
        print(f"    {f:<24}{sab[f]['incorrect']:>5}/{sab[f]['cases']:<4}"
              f"{saf[f]['incorrect']:>4}/{saf[f]['cases']:<3}")
    bad = [r for r in raf if (r["expect"] == "benign") == r["flagged"]]
    if bad:
        print("\n    still incorrect:")
        for r in bad[:12]:
            print(f"      {r['score']:>3} {'flagged' if r['flagged'] else 'missed':<8}"
                  f" [{r['family']}] {r['url'][:58]}")

    # Recall check: reputation must not rescue actual phishing.
    ds = pd.read_csv("data/urlguard_dataset.csv")
    ho = ds[(ds.split == "holdout")]
    hu = ho.url.tolist()
    hy = ho.label.tolist()
    hb_base = evaluate(hu, False)
    hb_fused = evaluate(hu, True)

    def stats(scores):
        tp = sum(1 for s, y in zip(scores, hy) if y == 1 and s >= BAND_THRESHOLD)
        fn_ = sum(1 for s, y in zip(scores, hy) if y == 1 and s < BAND_THRESHOLD)
        fp = sum(1 for s, y in zip(scores, hy) if y == 0 and s >= BAND_THRESHOLD)
        tn = sum(1 for s, y in zip(scores, hy) if y == 0 and s < BAND_THRESHOLD)
        return {"recall": round(tp / max(tp + fn_, 1), 4),
                "fpr": round(fp / max(fp + tn, 1), 4),
                "precision": round(tp / max(tp + fp, 1), 4),
                "TP": tp, "FP": fp, "FN": fn_, "TN": tn}

    s1, s2 = stats(hb_base), stats(hb_fused)
    print(f"\n[3] Holdout ({len(hu):,} rows), same >= {BAND_THRESHOLD} rule")
    print(f"    {'':<16}{'recall':>9}{'FPR':>9}{'precision':>11}")
    print(f"    {'URL only':<16}{s1['recall']:>9.4f}{s1['fpr']:>9.4f}{s1['precision']:>11.4f}")
    print(f"    {'+ reputation':<16}{s2['recall']:>9.4f}{s2['fpr']:>9.4f}{s2['precision']:>11.4f}")
    print(f"    recall change {s2['recall']-s1['recall']:+.4f}   "
          f"FPR change {s2['fpr']-s1['fpr']:+.4f}")
    print(f"\n[4] Latency with offline intel: {lat:.2f} ms/URL")

    out = {
        "band_threshold": BAND_THRESHOLD,
        "hard_benign": {"url_only": nb, "fused": nf, "cases": len(rb),
                        "by_category_url_only": sb, "by_category_fused": sf,
                        "still_flagged": still},
        "adversarial": {"url_only_incorrect": wrong(rab),
                        "fused_incorrect": wrong(raf), "cases": len(cases),
                        "by_family_fused": saf, "still_incorrect": bad},
        "holdout": {"url_only": s1, "fused": s2, "rows": len(hu)},
        "latency_ms_per_url": round(lat, 3),
    }
    with open("models/v3/benchmarks_fused.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("    wrote models/v3/benchmarks_fused.json")
    print("=" * 78)


if __name__ == "__main__":
    main()
