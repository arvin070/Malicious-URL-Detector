"""Generates data/sample_urls.csv -- a small SYNTHETIC dataset so the
pipeline runs immediately. Replace with download_real_dataset.py output
before trusting results for your report."""

import random
import csv
import os

random.seed(42)

BENIGN_TEMPLATES = [
    "https://www.{brand}.com", "https://www.{brand}.com/about",
    "https://{brand}.org/docs/{n}", "https://shop.{brand}.com/product/{n}",
    "https://blog.{brand}.com/{n}/post-title", "https://{brand}.io/api/v1/users/{n}",
]
BRANDS = ["github", "wikipedia", "nytimes", "spotify", "amazon", "python",
          "nasa", "harvard", "mozilla", "cloudflare", "stackoverflow"]

PHISH_TEMPLATES = [
    "http://{brand}-secure-login.{tld}/verify?id={n}",
    "http://{brand}.account-update.{tld}/signin.php",
    "http://192.168.{n}.{n2}/wp-admin/confirm-account",
    "http://{brand}-{n}.tk/paypal/webscr?cmd=login",
    "http://free-bonus-gift.{tld}/claim/{n}",
    "http://bit.ly/{n}suspicious",
    "http://secure-{brand}-banking-update.{tld}/login/verify/{n}",
]
PHISH_BRANDS = ["paypal", "chase", "appleid", "netflix", "amazon", "wellsfargo", "irs"]
SUS_TLDS = ["ru", "tk", "cf", "gq", "xyz", "top"]


def gen_benign(n):
    return [random.choice(BENIGN_TEMPLATES).format(brand=random.choice(BRANDS), n=random.randint(1, 9999))
            for _ in range(n)]


def gen_phish(n):
    return [random.choice(PHISH_TEMPLATES).format(
        brand=random.choice(PHISH_BRANDS), tld=random.choice(SUS_TLDS),
        n=random.randint(1, 999), n2=random.randint(1, 255)) for _ in range(n)]


def main():
    os.makedirs("data", exist_ok=True)
    rows = [(u, 0) for u in gen_benign(500)] + [(u, 1) for u in gen_phish(500)]
    random.shuffle(rows)
    out_path = os.path.join("data", "sample_urls.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
