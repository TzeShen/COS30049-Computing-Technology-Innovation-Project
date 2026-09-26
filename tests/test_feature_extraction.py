"""
tests/test_feature_extraction.py

Unit tests for the offline URL feature extractor. These target the
specific signals the assignment calls out (excessive length, IP host,
look-alike characters, unusual TLD) plus a few edge cases that broke
during development (see homoglyph_char_ratio below).

Run from the repo root:
    pip install -e ".[dev]"   # or: pip install -e . pytest
    pytest
"""

from phishing_detector.feature_extraction import extract_features


def test_ip_host_detected():
    f = extract_features("http://192.168.1.1/paypal/login.php")
    assert f.has_ip_address == 1
    assert f.subdomain_count == 0


def test_normal_domain_not_flagged_as_ip():
    f = extract_features("https://www.google.com/search?q=test")
    assert f.has_ip_address == 0


def test_suspicious_tld_detected():
    f = extract_features("http://secure-paypal-login-verify.xyz/account")
    assert f.is_suspicious_tld == 1


def test_common_tld_not_flagged():
    f = extract_features("https://www.wikipedia.org/wiki/Phishing")
    assert f.is_suspicious_tld == 0


def test_punycode_detected():
    f = extract_features("https://xn--pple-43d.com/signin")
    assert f.has_punycode == 1


def test_https_flag():
    assert extract_features("https://example.com").uses_https == 1
    assert extract_features("http://example.com").uses_https == 0


def test_url_length_matches_input():
    url = "https://www.example.com/a/b/c?x=1"
    assert extract_features(url).url_length == len(url)


def test_shortener_domain_detected():
    f = extract_features("http://bit.ly/3xample")
    assert f.is_known_shortener == 1


def test_at_symbol_flagged():
    f = extract_features("http://example.com@evil.com/login")
    assert f.has_at_symbol == 1


def test_homoglyph_ratio_is_low_for_ordinary_domains():
    # Regression test: an earlier version of this feature flagged almost
    # every domain because it checked *presence* of characters like '0',
    # '1', 'o', 'l' rather than their *ratio* in the hostname. A single
    # incidental 'o' or '1' in an otherwise normal domain should not
    # push this feature high.
    f = extract_features("https://www.wikipedia.org/wiki/Phishing")
    assert f.homoglyph_char_ratio < 0.35


def test_homoglyph_ratio_is_high_for_dense_substitution():
    f = extract_features("http://payp0l1d.com/login")  # many 0/1/l substitutions
    assert f.homoglyph_char_ratio >= 0.35


def test_malformed_url_degrades_gracefully():
    # No scheme, minimal structure -- should not raise, and should
    # produce sane defaults rather than garbage values.
    f = extract_features("not a real url at all")
    assert f.url_length == len("not a real url at all")
    assert isinstance(f.has_ip_address, int)


def test_batch_extraction_matches_single(tmp_path=None):
    from phishing_detector.feature_extraction import extract_features_batch

    urls = [
        "https://www.google.com",
        "http://192.168.1.1/login",
    ]
    df = extract_features_batch(urls)
    assert len(df) == 2
    assert df.loc[1, "has_ip_address"] == 1
