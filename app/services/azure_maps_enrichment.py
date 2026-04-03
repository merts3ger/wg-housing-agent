"""
Azure Maps service.

Provides geocoding, commute estimation, and nearby POI lookups
to convert a ParsedListing into enrichment data for EnrichedListing.

Environment variables:
    AZURE_MAPS_SUBSCRIPTION_KEY  — required
    AZURE_MAPS_BASE_URL          — optional, defaults to https://atlas.microsoft.com
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_BASE_URL = os.getenv("AZURE_MAPS_BASE_URL", "https://atlas.microsoft.com").rstrip("/")
_API_VERSION = "1.0"


# Pools use a larger radius because there is typically only one per district.
_RADIUS_SUPERMARKET_M = 1000
_RADIUS_PARK_M = 1500
_RADIUS_POOL_M = 3000

# Reference: https://learn.microsoft.com/en-us/azure/azure-maps/supported-search-categories
_CAT_PARK = "PARK_RECREATION_AREA"
_CAT_POOL = "SWIMMING_POOL"
# Note: transit stop names are not fetched from Azure Maps.
# Transit access is now extracted directly from the listing text by the parser.


def is_configured() -> bool:
    return bool(os.getenv("AZURE_MAPS_SUBSCRIPTION_KEY"))


def _subscription_key() -> str:
    key = os.getenv("AZURE_MAPS_SUBSCRIPTION_KEY")
    if not key:
        raise RuntimeError("AZURE_MAPS_SUBSCRIPTION_KEY is not set.")
    return key


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EnrichmentResult:
    """All enrichment data derived from Azure Maps for one listing."""
    geocoded_lat: Optional[float] = None
    geocoded_lon: Optional[float] = None
    geocode_query: Optional[str] = None          # the query string that was geocoded
    commute_minutes: Optional[int] = None
    commute_confidence: Optional[str] = None     # "high", "medium", "low"
    nearby_parks: list[str] = field(default_factory=list)
    nearby_swimming_pools: list[str] = field(default_factory=list)
    nearby_supermarkets: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def enrich(
    address_text: Optional[str],
    neighborhood: Optional[str],
    location_precision: Optional[str],
    commute_destination: str,
    city: Optional[str] = None,
) -> EnrichmentResult:
    """
    Geocode the listing location, then fetch commute time and nearby POIs.

    Parameters
    ----------
    address_text        Exact or partial address from the listing (preferred).
    neighborhood        Neighbourhood name as fallback.
    location_precision  "exact", "neighborhood", or "city" — drives confidence.
    commute_destination User's workplace / university address.
    city                City name extracted from the listing (e.g. "Kiel", "Hamburg").
    """
    result = EnrichmentResult()

    if not is_configured():
        result.errors.append("AZURE_MAPS_SUBSCRIPTION_KEY not configured — skipping enrichment.")
        return result

    # --- Geocode listing location ---
    query, confidence = _build_geocode_query(address_text, neighborhood, location_precision, city)
    result.geocode_query = query
    result.commute_confidence = confidence

    coords = _geocode(query, result)
    if coords is None:
        return result  # errors already recorded

    result.geocoded_lat, result.geocoded_lon = coords
    lat, lon = coords

    # --- Commute time ---
    dest_coords = _geocode(commute_destination, result, label="destination")
    if dest_coords is not None:
        minutes = _route_transit_minutes(lat, lon, dest_coords[0], dest_coords[1], result)
        if minutes is not None:
            result.commute_minutes = minutes

    # --- Nearby POIs ---
    result.nearby_parks = _search_parks(lat, lon, result)
    result.nearby_swimming_pools = _search_swimming_pools(lat, lon, result)
    result.nearby_supermarkets = _search_supermarkets(lat, lon, result)

    return result


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def _build_geocode_query(
    address_text: Optional[str],
    neighborhood: Optional[str],
    location_precision: Optional[str],
    city: Optional[str] = None,
    country: str = "Germany",
) -> tuple[str, str]:
    """
    Build the best possible geocode query string and return (query, confidence).
    Fallback order: address (+ city) → neighborhood + city → city → country.
    """
    city_suffix = f"{city}, {country}" if city else country

    if address_text:
        already_has_city = city and city.lower() in address_text.lower()
        query = address_text if already_has_city else f"{address_text}, {city_suffix}"
        confidence = "high" if location_precision == "exact" else "medium"
        return query, confidence

    if neighborhood:
        query = f"{neighborhood}, {city_suffix}"
        confidence = "medium" if location_precision == "neighborhood" else "low"
        return query, confidence

    if city:
        return f"{city}, {country}", "low"

    return country, "low"


def _geocode(query: str, result: EnrichmentResult, label: str = "listing") -> Optional[tuple[float, float]]:
    """
    Geocode a free-text address using Azure Maps Search Address API.
    Returns (lat, lon) or None on failure.
    """
    url = f"{_BASE_URL}/search/address/json"
    params = {
        "api-version": _API_VERSION,
        "subscription-key": _subscription_key(),
        "query": query,
        "limit": 1,
        "countrySet": "DE",
    }

    try:
        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            result.errors.append(f"Geocoding returned no results for {label}: {query!r}")
            return None
        pos = results[0]["position"]
        return pos["lat"], pos["lon"]
    except httpx.HTTPStatusError as exc:
        result.errors.append(f"Geocoding HTTP error for {label}: {exc.response.status_code} — {exc}")
        return None
    except Exception as exc:
        result.errors.append(f"Geocoding failed for {label}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Commute (transit routing)
# ---------------------------------------------------------------------------

def _route_transit_minutes(
    origin_lat: float, origin_lon: float,
    dest_lat: float, dest_lon: float,
    result: EnrichmentResult,
) -> Optional[int]:
    """
    Estimate public-transit commute time using Azure Maps Route Matrix / Directions API.
    Uses the walking+transit travel mode for a realistic commute estimate.
    Returns travel time in minutes, or None on failure.
    """
    url = f"{_BASE_URL}/route/directions/json"
    params = {
        "api-version": _API_VERSION,
        "subscription-key": _subscription_key(),
        "query": f"{origin_lat},{origin_lon}:{dest_lat},{dest_lon}",
        "travelMode": "pedestrian",  # Azure Maps free tier supports pedestrian; transit requires preview
        "routeType": "fastest",
    }

    try:
        response = httpx.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        routes = data.get("routes", [])
        if not routes:
            result.errors.append("Route API returned no routes.")
            return None
        seconds = routes[0]["summary"]["travelTimeInSeconds"]
        return round(seconds / 60)
    except httpx.HTTPStatusError as exc:
        result.errors.append(f"Route API HTTP error: {exc.response.status_code} — {exc}")
        return None
    except Exception as exc:
        result.errors.append(f"Route API failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# POI search helpers
# Fallback: _poi_search_by_text — plain free-text search without categorySet,
# used only when the category search returns nothing.
# ---------------------------------------------------------------------------

def _poi_search_by_category(
    lat: float,
    lon: float,
    category: str,
    radius_m: int,
    result: EnrichmentResult,
    fetch_limit: int = 15,
) -> list[dict]:
    """
    Category-constrained POI search using /search/poi/category/json.

    Preferred over numeric categorySet IDs on the general POI endpoint because:
    - the category endpoint is purpose-built for this lookup pattern
    - category name strings are self-documenting and stable
    - results are ranked within the category, reducing noise before post-filtering
    """
    url = f"{_BASE_URL}/search/poi/category/json"
    params: dict = {
        "api-version": _API_VERSION,
        "subscription-key": _subscription_key(),
        "query": category,
        "lat": lat,
        "lon": lon,
        "radius": radius_m,
        "limit": fetch_limit,
        "countrySet": "DE",
    }
    try:
        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("results", [])
    except httpx.HTTPStatusError as exc:
        result.errors.append(
            f"Category POI search ({category!r}) HTTP error: {exc.response.status_code}"
        )
        return []
    except Exception as exc:
        result.errors.append(f"Category POI search ({category!r}) failed: {exc}")
        return []


def _poi_search_by_text(
    lat: float,
    lon: float,
    query: str,
    radius_m: int,
    result: EnrichmentResult,
    fetch_limit: int = 15,
) -> list[dict]:
    """
    Plain free-text POI search, no category restriction.
    Used as a fallback when category search returns nothing.
    """
    url = f"{_BASE_URL}/search/poi/json"
    params: dict = {
        "api-version": _API_VERSION,
        "subscription-key": _subscription_key(),
        "query": query,
        "lat": lat,
        "lon": lon,
        "radius": radius_m,
        "limit": fetch_limit,
        "countrySet": "DE",
    }
    try:
        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("results", [])
    except httpx.HTTPStatusError as exc:
        result.errors.append(f"Text POI search ({query!r}) HTTP error: {exc.response.status_code}")
        return []
    except Exception as exc:
        result.errors.append(f"Text POI search ({query!r}) failed: {exc}")
        return []


def _item_name(item: dict) -> Optional[str]:
    """Extract the display name from a POI result item."""
    return item.get("poi", {}).get("name") or item.get("address", {}).get("freeformAddress")


def _filter_names(items: list[dict], block: set[str], keep: set[str], limit: int) -> list[str]:
    """
    Lightweight post-filter on POI names.
    - block: exclude any result whose name contains one of these substrings.
    - keep: if non-empty, also require at least one of these substrings (used for
      text-fallback paths where results are not pre-filtered by category).
    Returns up to `limit` deduplicated names.
    """
    names: list[str] = []
    for item in items:
        name = _item_name(item)
        if not name:
            continue
        n = name.lower()
        if any(kw in n for kw in block):
            continue
        if keep and not any(kw in n for kw in keep):
            continue
        if name not in names:
            names.append(name)
        if len(names) == limit:
            break
    return names


# ---------------------------------------------------------------------------
# Parks  (category 9362 — PARK_AND_RECREATION_AREA)
#
# ---------------------------------------------------------------------------

def _search_parks(lat: float, lon: float, result: EnrichmentResult) -> list[str]:
    items = _poi_search_by_category(lat, lon, _CAT_PARK, _RADIUS_PARK_M, result)
    block = {"parkhaus", "parkplatz", "tiefgarage", "parking", "garage",
             "klinik", "krankenhaus", "q-park", "mobilitätsstation"}
    return _filter_names(items, block=block, keep=set(), limit=3)


# ---------------------------------------------------------------------------
# Swimming pools  (category 9718007 — SWIMMING_POOL)
#
# ---------------------------------------------------------------------------

def _search_swimming_pools(lat: float, lon: float, result: EnrichmentResult) -> list[str]:
    items = _poi_search_by_category(lat, lon, _CAT_POOL, _RADIUS_POOL_M, result)
    block = {"spa", "wellness", "hotel", "sauna", "therme", "beauty"}
    return _filter_names(items, block=block, keep=set(), limit=3)


# ---------------------------------------------------------------------------
# Supermarkets  (no single reliable category ID; use chain names + text filter)
#
# ---------------------------------------------------------------------------

_GROCERY_PREFIXES = (
    "rewe", "aldi", "lidl", "edeka", "penny", "netto", "norma", "kaufland",
    "real", "globus", "tegut", "denn's", "denns", "biomarkt", "bio company",
)

_GROCERY_BLOCK = {
    "kiosk", "späti", "spätverkauf", "tabak", "drogerie", "apotheke",
    "veganski", "marktschwärmer", "reformhaus",
}

_GROCERY_GENERIC_KEEP = {"supermarkt", "lebensmittel", "markt"}


def _search_supermarkets(lat: float, lon: float, result: EnrichmentResult) -> list[str]:
    items = _poi_search_by_text(lat, lon, "supermarkt lebensmittel", _RADIUS_SUPERMARKET_M, result)
    names: list[str] = []
    for item in items:
        name = _item_name(item)
        if not name:
            continue
        n = name.lower()
        if any(kw in n for kw in _GROCERY_BLOCK):
            continue
        is_chain = n.startswith(_GROCERY_PREFIXES)
        is_generic = any(kw in n for kw in _GROCERY_GENERIC_KEEP)
        if not is_chain and not is_generic:
            continue
        if name not in names:
            names.append(name)
        if len(names) == 4:
            break
    return names


