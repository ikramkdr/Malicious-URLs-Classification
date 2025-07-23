from urllib.parse import urlparse
import re

def extract_features_from_url(url: str) -> dict:
    from urllib.parse import urlparse
    import re

    known_safe_domains = {
        "google.com", "github.com", "youtube.com", "bbc.com", "wikipedia.org", "facebook.com"
    }

    suspicious_tlds = {
        "tk", "ml", "ga", "cf", "gq", "xyz", "top", "ru", "cn", "pw", "biz", "info"
    }

    # Gestion of malformed URLs
    try:
        url = url.strip()
        if not url.startswith("http"):
            url = "http://" + url
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "").strip()
        path = parsed.path.lower()
    except Exception:
        # Invalid URL → We return a vector [000...]
        return {
            "url_length": 0,
            "num_dots": 0,
            "has_ip": 0,
            "num_subdomains": 0,
            "has_https": 0,
            "has_login": 0,
            "has_admin": 0,
            "count_digits": 0,
            "count_special_char": 0,
            "is_known_safe_domain": 0,
            "has_at_symbol": 0,
            "has_double_slash_redirect": 0,
            "is_short_url": 0,
            "has_suspicious_tld": 0,
            "count_hyphens": 0
        }

    tld = domain.split('.')[-1] if '.' in domain else ""

    features = {
        "url_length": len(url),
        "num_dots": url.count('.'),
        "has_ip": 1 if re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", domain) else 0,
        "num_subdomains": domain.count('.') - 1,
        "has_https": 1 if parsed.scheme == "https" else 0,
        "has_login": 1 if "login" in url.lower() else 0,
        "has_admin": 1 if "admin" in url.lower() else 0,
        "count_digits": sum(c.isdigit() for c in url),
        "count_special_char": sum(c in "@!#$%^&*()+=[]{}|\\:;\"'<>,?" for c in url),
        "is_known_safe_domain": 1 if domain in known_safe_domains else 0,
        "has_at_symbol": 1 if "@" in url else 0,
        "has_double_slash_redirect": 1 if "//" in path else 0,
        "is_short_url": 1 if len(url) < 20 else 0,
        "has_suspicious_tld": 1 if tld in suspicious_tlds else 0,
        "count_hyphens": url.count("-")
    }

    return features

