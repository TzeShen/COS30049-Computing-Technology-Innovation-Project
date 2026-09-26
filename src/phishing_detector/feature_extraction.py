"""
feature_extraction.py

Turns a raw URL string into a fixed-width vector of engineered features
("signals") that describe *why* a URL might look suspicious.

Design goals
------------
- Fully offline: no DNS lookups, no HTTP requests, no WHOIS calls.
  Every feature is derived purely from the URL string itself.
- Each feature is individually interpretable, so it can be surfaced
  directly in the app's explanation UI (e.g. "uses an IP address
  instead of a domain name").
- Deterministic and fast enough to run per-keystroke in an interactive tool.

Feature groups
--------------
1. Lexical / length-based   : overall length, counts of special characters
2. Host-based               : IP-literal host, subdomain depth, hyphens in host
3. Look-alike / obfuscation : punycode (xn--), homoglyph character ratio,
                               excessive percent-encoding
4. TLD-based                : suspicious / rarely-legitimate top-level domains
5. Structural                : presence of '@', redirect-style double slashes,
                               scheme (http vs https), port number
"""

from __future__ import annotations

import ipaddress
import math
import re
from dataclasses import dataclass, fields
from urllib.parse import urlsplit

# ---------------------------------------------------------------------------
# Reference lists (kept small and explicit for transparency / marking).
# In a production system these would be sourced from a public registry;
# here they are illustrative, offline, static lists.
# ---------------------------------------------------------------------------

SUSPICIOUS_TLDS = {
    "zip", "mov", "xyz", "top", "tk", "gq", "ml", "ga", "cf",
    "click", "link", "work", "support", "country", "kim", "rest",
    "info", "biz", "icu", "buzz", "loan", "win", "men", "date",
}

# A short list of brand tokens commonly impersonated in phishing URLs.
# Used only to flag brand-name + unrelated-domain combinations, not to
# accuse any single brand string of anything on its own.
BRAND_TOKENS = {
    "paypal", "apple", "microsoft", "google", "amazon", "netflix",
    "facebook", "instagram", "bankofamerica", "chase", "wellsfargo",
    "dhl", "fedex", "irs", "outlook", "office365", "coinbase",
}

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly",
}

# Characters that visually resemble Latin letters/digits in some fonts,
# often used in homoglyph attacks (kept small & illustrative).
HOMOGLYPH_CHARS = set("0Oo1lI")


def _shannon_entropy(s: str) -> float:
    """Character-level Shannon entropy of a string (0 for empty string)."""
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def _is_ip_host(host: str) -> bool:
    """True if the host part is a literal IPv4/IPv6 address."""
    host = host.strip("[]")  # strip IPv6 brackets if present
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _get_tld(host: str) -> str:
    if not host or "." not in host:
        return ""
    return host.rsplit(".", 1)[-1].lower()


def _registrable_domain(host: str) -> str:
    """Very rough eTLD+1 guess (no public-suffix-list dependency,
    intentionally offline/lightweight): second-to-last label + TLD."""
    parts = host.split(".")
    if len(parts) < 2:
        return host
    return ".".join(parts[-2:])


@dataclass
class URLFeatures:
    # --- lexical / length ---
    url_length: int = 0
    hostname_length: int = 0
    path_length: int = 0
    query_length: int = 0
    digit_count: int = 0
    digit_ratio: float = 0.0
    special_char_count: int = 0
    hyphen_count: int = 0
    dot_count: int = 0
    slash_count: int = 0
    entropy: float = 0.0

    # --- host-based ---
    has_ip_address: int = 0
    subdomain_count: int = 0
    hostname_hyphen_count: int = 0
    hostname_digit_count: int = 0
    is_known_shortener: int = 0

    # --- look-alike / obfuscation ---
    has_punycode: int = 0
    homoglyph_char_ratio: float = 0.0
    percent_encoded_count: int = 0
    brand_token_in_subdomain_or_path: int = 0

    # --- TLD-based ---
    tld_length: int = 0
    is_suspicious_tld: int = 0

    # --- structural ---
    uses_https: int = 0
    has_at_symbol: int = 0
    has_port: int = 0
    has_double_slash_redirect: int = 0  # "//" appearing after the initial scheme

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def extract_features(raw_url: str) -> URLFeatures:
    """
    Convert a single raw URL string into a URLFeatures object.

    Never touches the network: parsing only. Malformed input degrades
    gracefully (missing pieces just default to 0 / empty).
    """
    url = raw_url.strip()

    # Ensure a scheme so urlsplit parses the host correctly; this does
    # NOT fetch anything, it only affects local string parsing.
    working_url = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url) else "http://" + url

    parts = urlsplit(working_url)
    host = (parts.hostname or "").lower()
    path = parts.path or ""
    query = parts.query or ""

    f = URLFeatures()

    # lexical / length
    f.url_length = len(url)
    f.hostname_length = len(host)
    f.path_length = len(path)
    f.query_length = len(query)
    f.digit_count = sum(c.isdigit() for c in url)
    f.digit_ratio = f.digit_count / max(len(url), 1)
    f.special_char_count = sum(1 for c in url if not c.isalnum() and c not in "./:-")
    f.hyphen_count = url.count("-")
    f.dot_count = url.count(".")
    f.slash_count = url.count("/")
    f.entropy = round(_shannon_entropy(url), 4)

    # host-based
    f.has_ip_address = int(_is_ip_host(host))
    f.subdomain_count = max(host.count(".") - 1, 0) if not f.has_ip_address else 0
    f.hostname_hyphen_count = host.count("-")
    f.hostname_digit_count = sum(c.isdigit() for c in host)
    registrable = _registrable_domain(host)
    f.is_known_shortener = int(registrable in SHORTENER_DOMAINS)

    # look-alike / obfuscation
    f.has_punycode = int("xn--" in host)
    # Ratio (not presence) of characters that can visually resemble other
    # letters/digits (e.g. '0' for 'o', '1' for 'l'). A single occurrence
    # in an otherwise normal domain (common -- "0" or "1" appear in many
    # legitimate hostnames) isn't meaningful on its own; a high ratio,
    # especially mixed in with letters rather than as a clean numeric
    # token, is a much stronger and rarer signal of visual spoofing.
    host_letters_digits = host.replace(".", "")
    f.homoglyph_char_ratio = (
        sum(c in HOMOGLYPH_CHARS for c in host_letters_digits) / len(host_letters_digits)
        if host_letters_digits else 0.0
    )
    f.percent_encoded_count = url.count("%")
    subdomain_and_path = host.rsplit(registrable, 1)[0] + " " + path
    f.brand_token_in_subdomain_or_path = int(
        any(brand in subdomain_and_path.lower() for brand in BRAND_TOKENS)
        and registrable.split(".")[0].lower() not in BRAND_TOKENS
    )

    # TLD-based
    tld = _get_tld(host)
    f.tld_length = len(tld)
    f.is_suspicious_tld = int(tld in SUSPICIOUS_TLDS)

    # structural
    f.uses_https = int(parts.scheme == "https")
    f.has_at_symbol = int("@" in url)
    f.has_port = int(parts.port is not None)
    # a "//" that shows up again after the scheme's own "//" is a classic
    # open-redirect / lookalike pattern, e.g. http://real.com//evil.com
    after_scheme = url.split("://", 1)[-1]
    f.has_double_slash_redirect = int("//" in after_scheme)

    return f


def extract_features_batch(urls: list[str]) -> "pandas.DataFrame":  # noqa: F821
    """Vectorised convenience wrapper for a list of URLs -> DataFrame."""
    import pandas as pd

    rows = [extract_features(u).as_dict() for u in urls]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick manual sanity check
    samples = [
        "https://www.google.com/search?q=test",
        "http://192.168.1.1/paypal/login.php",
        "http://secure-paypal-login-verify.xyz/account/update",
        "https://xn--pple-43d.com/signin",  # punycode look-alike for "apple"
        "http://bit.ly/3xample",
    ]
    for u in samples:
        print(u)
        print(extract_features(u).as_dict())
        print()
