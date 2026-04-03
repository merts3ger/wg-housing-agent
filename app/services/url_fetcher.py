from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

_SUPPORTED_HOSTS = {"wg-gesucht.de", "www.wg-gesucht.de"}
_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "aside", "button"]


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host not in _SUPPORTED_HOSTS:
        raise ValueError(
            f"Unsupported URL host '{host}'. Only WG-Gesucht listing URLs are supported."
        )

    if not parsed.path.lower().endswith(".html"):
        raise ValueError(
            f"URL does not look like a WG-Gesucht listing page: {url}"
        )


def _clean_lines(text: str) -> str:
    seen: set[str] = set()
    cleaned_lines: list[str] = []

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# Markers that indicate the start of the actual listing content.
_START_MARKERS = ["Ad ID:", "Room size", "Final rent", "Address"]

# Markers that indicate we've passed the listing and entered unrelated sections
# (tenant profile, sharing buttons, report links, site statistics, etc.).
_END_MARKERS = ["Member since", "Share", "Report", "Private User", "Statistics", "Upload here"]


def _trim_to_listing_window(text: str) -> str:
    """
    Trim cleaned page text to the likely listing-content window.

    WG-Gesucht pages embed the listing details inside a larger page that also
    contains navigation, login prompts, recommendation widgets, and footer content.
    The rule-based and LLM parsers both perform better when given only the relevant
    slice instead of the full page dump.
    """
    start_pos = None
    for marker in _START_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            start_pos = idx
            break  # use the earliest marker found

    if start_pos is None:
        # No start marker found — return the full text rather than nothing
        return text

    trimmed = text[start_pos:]

    for marker in _END_MARKERS:
        idx = trimmed.find(marker)
        if idx != -1:
            trimmed = trimmed[:idx]
            break  # stop at the first end marker

    return trimmed.strip()


def _extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    root = soup.find("main") or soup.find("body") or soup

    text = root.get_text(separator="\n", strip=True)
    return _trim_to_listing_window(_clean_lines(text))


def fetch_listing_text_from_url(url: str) -> str:
    _validate_url(url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    }

    with httpx.Client(follow_redirects=True, timeout=20.0) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()

    text = _extract_text_from_html(response.text)

    if len(text.strip()) < 100:
        raise ValueError(
            "Fetched page appears empty or too short to parse reliably."
        )

    return text