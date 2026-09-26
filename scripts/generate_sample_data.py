"""
scripts/generate_sample_data.py

Produces a small SYNTHETIC url,label dataset purely so the rest of the
pipeline (data_prep -> train_classification -> train_clustering ->
train_innovation_model) can be run and tested end-to-end offline,
without needing network access to download a real dataset.

This is a dev/demo tool, not part of the phishing_detector package --
it does not belong in production and is kept separate from src/.

*** This is a stand-in, not your real data source. ***
For the actual assignment, replace data/raw/raw_urls.csv with a real
labelled dataset, e.g.:
  - UCI "Phishing Websites" / "PhishStorm" datasets
  - Kaggle "Phishing Site URLs" (Manu Siddhartha)
  - PhishTank (phishing) + Tranco / Majestic Million (legitimate) URL lists
Keep the same two columns: url, label  (label: 1 = phishing, 0 = legitimate)

Run from the repo root:
    python scripts/generate_sample_data.py
"""

import csv
import os
import random

random.seed(42)

LEGIT_DOMAINS = [
    "google.com", "wikipedia.org", "github.com", "nytimes.com", "bbc.co.uk",
    "amazon.com", "microsoft.com", "apple.com", "reddit.com", "stackoverflow.com",
    "spotify.com", "netflix.com", "linkedin.com", "dropbox.com", "adobe.com",
    "unimelb.edu.au", "abc.net.au", "commbank.com.au", "gov.uk", "who.int",
]
LEGIT_PATHS = ["", "/", "/about", "/search?q=news", "/blog/2026/update",
               "/user/profile", "/docs/api", "/products/laptop", "/en/help"]

BRANDS = ["paypal", "apple", "microsoft", "amazon", "netflix", "chase",
          "bankofamerica", "dhl", "irs", "outlook", "coinbase", "instagram"]
SUS_TLDS = ["xyz", "top", "tk", "click", "gq", "icu", "loan", "win", "info", "buzz"]
IP_HOSTS = ["192.168.{}.{}", "10.0.{}.{}", "45.33.{}.{}", "104.21.{}.{}"]

PHISH_PATH_WORDS = ["login", "secure", "verify", "account", "update",
                     "confirm", "signin", "webscr", "reset-password", "billing"]


def random_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(LEGIT_PATHS)
    scheme = "https"
    # add a bit of natural variety (ids/query strings) so repeated
    # domain+path combinations don't collapse into identical duplicate URLs
    suffix = ""
    if random.random() < 0.6:
        suffix = f"{'&' if '?' in path else '?'}ref={random.randint(1000, 99999)}"
    return f"{scheme}://www.{domain}{path}{suffix}"


def random_phishing_url():
    style = random.choice(["ip_host", "lookalike_domain", "punycode", "shortener", "long_subdomain"])
    brand = random.choice(BRANDS)
    word = random.choice(PHISH_PATH_WORDS)

    if style == "ip_host":
        template = random.choice(IP_HOSTS)
        ip = template.format(random.randint(1, 254), random.randint(1, 254))
        return f"http://{ip}/{brand}/{word}.php"

    if style == "lookalike_domain":
        tld = random.choice(SUS_TLDS)
        filler = random.choice(["secure", "account", "login", "verify", "id"])
        return f"http://{brand}-{filler}-{word}.{tld}/{word}"

    if style == "punycode":
        return f"https://xn--{brand[:3]}-{random.randint(10,99)}d.com/{word}"

    if style == "shortener":
        code = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=7))
        return f"http://bit.ly/{code}"

    # long_subdomain: many nested subdomains ending in a suspicious TLD
    tld = random.choice(SUS_TLDS)
    subs = ".".join(random.choices([brand, word, "secure", "id", "session"], k=4))
    return f"http://{subs}.{tld}/{word}.php?id={random.randint(1000,9999)}"


def generate(n_legit=600, n_phish=600, out_path="data/raw/raw_urls.csv"):
    rows = [(random_legit_url(), 0) for _ in range(n_legit)]
    rows += [(random_phishing_url(), 1) for _ in range(n_phish)]
    random.shuffle(rows)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path} "
          f"({n_legit} legitimate, {n_phish} phishing)")


if __name__ == "__main__":
    generate()
