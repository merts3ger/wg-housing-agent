"""
Listing parser for WG / student-room rental offers.

Primary path:   LLM-based structured extraction via Azure OpenAI (Structured Outputs).
                The model is constrained to return a JSON object that maps directly
                onto ParsedListing, eliminating free-form hallucination.
Fallback path:  Rule-based extraction (regex + keyword matching).

To activate the LLM path, add these to your .env file:
    AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
    AZURE_OPENAI_API_KEY=<your-key>
    AZURE_OPENAI_DEPLOYMENT=<deployment-name>   # e.g. gpt-4o
    AZURE_OPENAI_API_VERSION=2024-02-01         # optional, this is the default

If any of these are missing, _llm_extract() returns None and parse_listing()
falls back to _rule_based_extract() automatically.
"""

import json
import logging
import re
from typing import Optional

from app.schemas.listing import ParsedListing
from app.services import azure_openai_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM extraction (Azure OpenAI)
# ---------------------------------------------------------------------------

# Maximum characters of listing text sent to the LLM.
# WG-Gesucht pages after trimming are typically 1–3k chars; 5000 gives headroom
# without approaching token limits on gpt-4o (128k context).
MAX_INPUT_CHARS = 5000

_SYSTEM_PROMPT = """\
You are a structured data extraction assistant specialising in German WG and student-room rental listings.

Your output must be a JSON object that exactly matches the ParsedListing schema below.
Extract only what is explicitly stated or strongly supported by the listing text.
Do NOT hallucinate values — use null for any field you are not confident about.

Schema:
{
  "title":                        string,        // see title rule below
  "warm_rent":                    integer | null, // total monthly rent incl. utilities in EUR
  "cold_rent":                    integer | null, // base monthly rent excl. utilities in EUR; see cold_rent rule
  "deposit":                      integer | null, // security deposit in EUR
  "neighborhood":                 string | null,  // district or neighbourhood name
  "address_text":                 string | null,  // see address_text rule below
  "furnished":                    boolean | null, // true=furnished, false=explicitly unfurnished, null=unknown
  "furnishing_status":            "furnished" | "partially_furnished" | "unfurnished" | "takeover_possible" | null, // see furnishing rule below
  "furniture_details":            string | null,  // short factual note on specific furniture mentioned, e.g. "Desk and pallet bed can be taken over"
  "available_from":               string | null,  // move-in date as written, e.g. "01.05.2025" or "sofort"
  "amenities":                    string[],       // amenities explicitly mentioned
  "description_summary":          string | null,  // 1-2 sentence factual summary of the listing description
  "missing_info":                 string[],       // see missing_info rule below
  "location_precision":           "exact" | "neighborhood" | "city" | null,
  "room_size_sqm":                integer | null, // room size in square metres if explicitly stated
  "flatmate_count":               integer | null, // total number of people in the flatshare if stated
  "flatshare_age_min":            integer | null, // lower bound of current flatmates' age range, if given
  "flatshare_age_max":            integer | null, // upper bound of current flatmates' age range, if given
  "flatshare_description_summary": string | null, // see flatshare_description_summary rule below
  "transit_access_mentioned":     boolean | null, // true if the listing explicitly mentions nearby public transport; null if not mentioned
  "transit_access_summary":       string | null,  // short factual note on what the listing says about public transport, e.g. "Bus stop 2 min walk, tram nearby"
  "transit_access_walk_minutes":  integer | null, // walking minutes to the nearest transit stop if explicitly stated; null if not stated
  "city":                         string | null,  // city name from the address or listing text, e.g. "Kiel" or "Hamburg"
  "missing_info":                 []              // always return an empty array — this field is computed externally
}

Rules:

structured metadata priority:
- WG-Gesucht pages contain structured metadata blocks (e.g. "Flatshare details", "Searching for",
  "Available from", "Room size", "Costs"). These are labelled key-value lines.
- Always prefer values from these structured blocks over values inferred from prose descriptions.
  Prose may paraphrase or omit detail; the structured block is authoritative.

title:
- Construct a short normalized display title. Do NOT copy the raw ad headline.
- Preferred format (use the first that applies):
  1. If flatmate_count AND neighborhood are known: "{flatmate_count}er WG in {neighborhood}" — e.g. "5er WG in Exerzierplatz"
  2. If only neighborhood is known: "WG-Zimmer in {neighborhood}"
  3. If neighborhood is unknown but another location clue is present (e.g. city, street): use a short location-based title
  4. Fallback: "WG-Zimmer"
- Keep it short and display-friendly. No long sentences, no original ad copy.

furnishing_status / furniture_details:
- "furnished": room or flat is clearly described as furnished (möbliert).
- "unfurnished": explicitly described as empty/unfurnished (unmöbliert, leer).
- "partially_furnished": only some furniture is included (e.g. wardrobe but no bed).
- "takeover_possible": furniture is not included by default but can be taken over from the previous tenant or by arrangement (Übernahme möglich, kann übernommen werden).
- null: furnishing situation is not mentioned or too ambiguous to classify.
- furniture_details: populate whenever furnishing_status is not null, with a short factual note on what is mentioned. Leave null if nothing specific is stated.
- furnished (bool): set true only for "furnished", false only for "unfurnished"; leave null for all other statuses.

cold_rent:
- Only set a numeric value if the listing explicitly states the base rent (Kaltmiete) as a meaningful figure.
- Do NOT extract 0 from a UI field that is blank, unlabelled, or shows a dash — use null instead.
- A cold_rent of 0 is only valid if the listing text explicitly and meaningfully states the base rent is zero.

address_text:
- Prefer the fullest address string available: street name + number + postal code + city is ideal.
- If only street name + number is present, use that.
- If only a neighbourhood or district name is known (no street), leave address_text null and populate neighborhood instead.
- Do not construct an address by combining unrelated fragments.

room_size_sqm / flatmate_count:
- Extract numeric values only when clearly stated. Do not infer from phrases like "large room" or "small WG".
- flatmate_count is the total number of people living there (including the new tenant if stated that way).

flatshare_age_min / flatshare_age_max:
- Extract from descriptions of the current flatmates' ages, e.g. "we are between 24 and 30".
- If a single age is given for all flatmates, set both min and max to that value.

city:
- Extract the city name if it appears in the address block, postal code line, or body text.
- Prefer structured address data (e.g. "24105 Kiel") over prose mentions.
- If address_text already contains the city, extract it here too.
- Leave null if the city cannot be determined from the listing.

transit_access_mentioned / transit_access_summary / transit_access_walk_minutes:
- Set transit_access_mentioned to true only if the listing explicitly names or describes nearby public transport (bus, tram, U-Bahn, S-Bahn, train, etc.).
- transit_access_summary: write a 1-sentence factual note on what is stated, e.g. "Bus stop Exerzierplatz (lines 11, 81) is around the corner." Only populate if transit_access_mentioned is true.
- transit_access_walk_minutes: extract an integer only if the listing explicitly states a walking time to transit, e.g. "2 min zu Fuß zur Haltestelle". Do not estimate or infer — use null if not stated.

flatshare_description_summary:
- Summarize how the flatmates describe their shared life together — NOT the room or the flat itself.
- Focus on: whether they spend time together (shared meals, film nights, etc.), privacy expectations, social atmosphere, cleanliness norms, and any house rules about guests or noise.
- Write 2-4 short factual sentences based only on what is stated.
- Do NOT use vibe labels like "communal", "quiet", "creative", or "party flat". Describe what is actually written.
- If the listing contains no information about how the people live together (only room/location facts), leave this null.

Return the JSON object only. No markdown, no explanation.
"""


def _llm_extract(text: str) -> Optional[ParsedListing]:
    """
    Extract ParsedListing fields from listing text using Azure OpenAI Structured Outputs.

    Uses response_format={"type": "json_object"} (JSON mode) which is broadly supported
    across Azure OpenAI deployments. The system prompt constrains the model to the
    ParsedListing schema, acting as schema-guided extraction without requiring the
    newer json_schema response format.

    Returns a ParsedListing on success, or None if Azure OpenAI is not configured.
    Any exception (API error, JSON parse failure, Pydantic validation error) is
    caught by the caller in parse_listing() and triggers the rule-based fallback.
    """
    if not azure_openai_client.is_configured():
        return None

    client = azure_openai_client.get_client()
    deployment = azure_openai_client.get_deployment()

    response = client.chat.completions.create(
        model=deployment,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text[:MAX_INPUT_CHARS]},
        ],
        temperature=0,
    )

    raw_json = response.choices[0].message.content
    data = json.loads(raw_json)
    data["missing_info"] = []  # always ignore model-generated missing_info; computed in parse_listing()
    return ParsedListing(**data)


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

_AMOUNT_RE = re.compile(r"(?:€|EUR)\s*(\d{2,4})|(\d{2,4})\s*(?:€|EUR)", re.IGNORECASE)


def _find_amount_near(text: str, keyword_pattern: str) -> Optional[int]:
    kw_re = re.compile(keyword_pattern, re.IGNORECASE)
    for m in kw_re.finditer(text):
        window = text[max(0, m.start() - 60) : m.end() + 60]
        amt = _AMOUNT_RE.search(window)
        if amt:
            return int(amt.group(1) or amt.group(2))
    return None


def _extract_warm_rent(text: str) -> Optional[int]:
    return _find_amount_near(text, r"warm(?:miete)?|inkl\.?\s*(?:NK|Nebenkosten)")


def _extract_cold_rent(text: str) -> Optional[int]:
    return _find_amount_near(text, r"kalt(?:miete)?|zzgl\.?\s*(?:NK|Nebenkosten)|ohne\s+NK")


def _extract_deposit(text: str) -> Optional[int]:
    return _find_amount_near(text, r"kaution|depot|anzahlung")


_FURNISHED_YES = re.compile(
    r"\b(möbliert|moebliert|furnished|mit\s+möbeln|inkl\.?\s*möbel)\b", re.IGNORECASE
)
_FURNISHED_NO = re.compile(
    r"\b(unmöbliert|unmoebliert|unfurnished|ohne\s+möbel|leer(?:stehend)?)\b", re.IGNORECASE
)


def _extract_furnished(text: str) -> Optional[bool]:
    if _FURNISHED_YES.search(text):
        return True
    if _FURNISHED_NO.search(text):
        return False
    return None


_STREET_RE = re.compile(
    r"\b([A-ZÄÖÜ][a-zäöüß\-]+(?:straße|strasse|str\.|weg|allee|gasse|platz|ring|damm|ufer|chaussee))\s*\d+\w*\b",
    re.IGNORECASE,
)
_NEIGHBORHOOD_RE = re.compile(
    r"\b(?:in|im|aus|bezirk|stadtteil|viertel|kiez)\s+([A-ZÄÖÜ][a-zäöüß\-]+(?:\s[A-ZÄÖÜ][a-zäöüß\-]+)?)\b",
    re.IGNORECASE,
)


def _extract_location(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    street = _STREET_RE.search(text)
    if street:
        return street.group(0), None, "exact"
    nbhd = _NEIGHBORHOOD_RE.search(text)
    if nbhd:
        return None, nbhd.group(1), "neighborhood"
    return None, None, "city"


# Matches "24105 Kiel", "20095 Hamburg", etc.
_CITY_FROM_PLZ_RE = re.compile(r"\b\d{5}\s+([A-ZÄÖÜ][a-zäöüß\-]+(?:\s[A-ZÄÖÜ][a-zäöüß\-]+)?)\b")


def _extract_city(text: str, address_text: Optional[str] = None) -> Optional[str]:
    # Try postal code pattern first
    for src in (address_text or "", text):
        m = _CITY_FROM_PLZ_RE.search(src)
        if m:
            return m.group(1)
    return None


_AVAIL_DATE_RE = re.compile(
    r"\b(?:ab\s+)?(\d{1,2}\.\d{1,2}\.(?:\d{4}|\d{2}))\b"
    r"|\b(sofort|ab\s+sofort|immediately|now)\b"
    r"|\b(?:ab\s+)?(januar|februar|märz|april|mai|juni|juli|august|september|oktober|november|dezember"
    r"|january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{4}\b",
    re.IGNORECASE,
)


def _extract_available_from(text: str) -> Optional[str]:
    m = _AVAIL_DATE_RE.search(text)
    return m.group(0).strip() if m else None


_AMENITY_KEYWORDS: dict[str, str] = {
    "balkon": "balcony", "balcony": "balcony",
    "terrasse": "terrace", "terrace": "terrace",
    "einbauküche": "fitted kitchen", "fitted kitchen": "fitted kitchen",
    "keller": "cellar/storage", "fahrradkeller": "bike storage", "bike storage": "bike storage",
    "waschmaschine": "washing machine", "washing machine": "washing machine",
    "geschirrspüler": "dishwasher", "dishwasher": "dishwasher",
    "parkplatz": "parking", "garage": "parking", "parking": "parking",
    "aufzug": "elevator", "fahrstuhl": "elevator", "elevator": "elevator",
    "garten": "garden", "garden": "garden",
"wlan": "wifi included", "wifi": "wifi included", "internet": "wifi included",
}


def _extract_amenities(text: str) -> list[str]:
    lower = text.lower()
    found, seen = [], set()
    for keyword, label in _AMENITY_KEYWORDS.items():
        if keyword in lower and label not in seen:
            found.append(label)
            seen.add(label)
    return found


def _rule_based_extract(text: str) -> ParsedListing:
    """Rule-based fallback parser. Prefer _llm_extract() for URL-fetched text."""
    warm_rent = _extract_warm_rent(text)
    cold_rent = _extract_cold_rent(text)
    if warm_rent is None and cold_rent is None:
        generic = _AMOUNT_RE.search(text)
        if generic:
            warm_rent = int(generic.group(1) or generic.group(2))

    deposit = _extract_deposit(text)
    furnished = _extract_furnished(text)
    address_text, neighborhood, location_precision = _extract_location(text)
    city = _extract_city(text, address_text)
    available_from = _extract_available_from(text)
    amenities = _extract_amenities(text)

    title = next((l.strip()[:120] for l in text.splitlines() if l.strip()), "Untitled listing")
    summary = next((l.strip()[:200] for l in text.splitlines() if len(l.strip()) > 30), None)

    return ParsedListing(
        title=title,
        warm_rent=warm_rent,
        cold_rent=cold_rent,
        deposit=deposit,
        neighborhood=neighborhood,
        address_text=address_text,
        city=city,
        furnished=furnished,
        available_from=available_from,
        amenities=amenities,
        description_summary=summary,
        missing_info=[],  # populated by _populate_missing_info() in parse_listing()
        location_precision=location_precision,
    )


# ---------------------------------------------------------------------------
# missing_info computation
# ---------------------------------------------------------------------------

# Fields that are considered important enough to flag when absent.
# Deliberately excludes optional social/flatshare fields that are often legitimately absent.
_IMPORTANT_FIELDS: list[tuple[str, ...]] = [
    ("warm_rent", "cold_rent"),  # either is enough
    ("deposit",),
    ("furnishing_status",),
    ("available_from",),
    ("neighborhood", "address_text"),  # either is enough
    ("room_size_sqm",),
]


def _populate_missing_info(parsed: ParsedListing) -> ParsedListing:
    """
    Compute missing_info by inspecting the parsed object.
    Returns a copy of the ParsedListing with missing_info populated.
    """
    missing: list[str] = []
    for field_group in _IMPORTANT_FIELDS:
        if all(getattr(parsed, f, None) is None for f in field_group):
            # Use the first field name as the label, or "rent" / "location" for multi-field groups
            label = field_group[0] if len(field_group) == 1 else {
                ("warm_rent", "cold_rent"): "rent",
                ("neighborhood", "address_text"): "location",
            }.get(field_group, field_group[0])
            missing.append(label)
    return parsed.model_copy(update={"missing_info": missing})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_listing(raw_listing_text: str) -> ParsedListing:
    """
    Parse a WG/student-room listing into a structured ParsedListing.

    Tries LLM-based extraction first (requires Azure OpenAI env vars).
    Falls back to the rule-based parser if LLM is unavailable or fails.
    missing_info is always computed in Python after extraction.
    """
    text = raw_listing_text.strip()

    try:
        result = _llm_extract(text)
        if result is not None:
            return _populate_missing_info(result)
    except Exception as exc:
        logger.warning("LLM extraction failed (%s: %s), falling back to rule-based parser.", type(exc).__name__, exc)

    return _populate_missing_info(_rule_based_extract(text))
