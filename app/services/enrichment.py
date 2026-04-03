import logging

from app.schemas.profile import UserProfile
from app.schemas.listing import EnrichedListing, ParsedListing
from app.services import azure_maps_enrichment

logger = logging.getLogger(__name__)


def enrich_listing(parsed_listing: ParsedListing, user_profile: UserProfile) -> EnrichedListing:
    if not azure_maps_enrichment.is_configured():
        logger.warning("AZURE_MAPS_SUBSCRIPTION_KEY not set — skipping enrichment, commute will be null.")
        return EnrichedListing(
            **parsed_listing.model_dump(),
            commute_minutes=None,
            commute_confidence=None,
            nearby_parks=[],
            nearby_swimming_pools=[],
            nearby_transit=[],
            nearby_supermarkets=[],
        )

    try:
        result = azure_maps_enrichment.enrich(
            address_text=parsed_listing.address_text,
            neighborhood=parsed_listing.neighborhood,
            location_precision=parsed_listing.location_precision,
            commute_destination=user_profile.commute_destination,
            city=parsed_listing.city,
        )
        logger.info(
            "[enrichment] geocode_query=%r  commute=%s min (%s)  parks=%d  pools=%d  supermarkets=%d",
            result.geocode_query,
            result.commute_minutes,
            result.commute_confidence,
            len(result.nearby_parks),
            len(result.nearby_swimming_pools),
            len(result.nearby_supermarkets),
        )
    except Exception as exc:
        logger.warning("Azure Maps enrichment failed (%s: %s) — commute will be null.", type(exc).__name__, exc)
        result = None

    return EnrichedListing(
        **parsed_listing.model_dump(),
        commute_minutes=result.commute_minutes if result else None,
        commute_confidence=result.commute_confidence if result else None,
        nearby_parks=result.nearby_parks if result else [],
        nearby_swimming_pools=result.nearby_swimming_pools if result else [],
        nearby_transit=[],  # populated from listing text via transit_access_* fields
        nearby_supermarkets=result.nearby_supermarkets if result else [],
    )
