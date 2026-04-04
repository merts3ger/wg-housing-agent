"""
Orchestration layer for the MVP single-profile evaluation flow.

Usage:
    from app.services.evaluator import evaluate_url_for_default_profile
    result = evaluate_url_for_default_profile("https://www.wg-gesucht.de/...")
"""

from app.schemas.profile import UserProfile
from app.schemas.evaluation import EvaluationResponse, ListingSnapshot
from app.schemas.result import AgentAssessment
from app.graph.state import HousingGraphState
from app.graph.graph_builder import build_housing_graph
from app.services.url_fetcher import fetch_listing_text_from_url

# ---------------------------------------------------------------------------
# Default user profile
# ---------------------------------------------------------------------------

DEFAULT_PROFILE = UserProfile(
    budget_eur=700,
    max_commute_minutes=30,
    prefers_furnished=True,
    lifestyle_preference="quiet",
    wants_park_nearby=True,
    wants_swimming_pool_nearby=True,
    commute_destination="Christian-Albrechts-Platz 4, 24118 Kiel",
    preferred_flatmate_age_min=23,
    preferred_flatmate_age_max=35,
)

_graph = build_housing_graph()


def evaluate_url_for_default_profile(url: str) -> EvaluationResponse:
    """Fetch, parse, enrich, and evaluate a listing using the default profile."""
    listing_text = fetch_listing_text_from_url(url)

    initial_state: HousingGraphState = {
        "raw_listing_text": listing_text,
        "user_profile": DEFAULT_PROFILE,
        "parsed_listing": None,
        "enriched_listing": None,
        "budget_assessment": None,
        "lifestyle_assessment": None,
        "final_recommendation": None,
        "errors": [],
    }

    result = _graph.invoke(initial_state)

    rec = result.get("final_recommendation")
    if rec is None:
        raise RuntimeError("Pipeline completed without producing a recommendation.")

    # --- debug: pipeline internals ---
    enriched = result.get("enriched_listing")
    if enriched:
        print(f"[evaluator] address_text={enriched.address_text!r}  neighborhood={enriched.neighborhood!r}  location_precision={enriched.location_precision!r}")
        print(f"[evaluator] commute_minutes={enriched.commute_minutes}  commute_confidence={enriched.commute_confidence!r}")
    print(f"[evaluator] assessments in rec: {[a.agent_name for a in rec.assessments]}")
    # ---

    # Pull individual agent assessments 
    assessments_by_name: dict[str, AgentAssessment] = {a.agent_name: a for a in rec.assessments}
    budget_assessment = assessments_by_name.get(
        "budget_and_value",
        AgentAssessment(agent_name="budget_and_value", listing_title="", score=0.0, pros=[], cons=[]),
    )
    lifestyle_assessment = assessments_by_name.get(
        "lifestyle_and_daily_fit",
        AgentAssessment(agent_name="lifestyle_and_daily_fit", listing_title="", score=0.0, pros=[], cons=[]),
    )

    listing = rec.evaluated_listing
    snapshot = ListingSnapshot(
        title=listing.title if listing else "Unknown",
        warm_rent=listing.warm_rent if listing else None,
        neighborhood=listing.neighborhood if listing else None,
        room_size_sqm=listing.room_size_sqm if listing else None,
        available_from=listing.available_from if listing else None,
        furnishing_status=listing.furnishing_status if listing else None,
        commute_minutes=listing.commute_minutes if listing else None,
    )

    return EvaluationResponse(
        listing_snapshot=snapshot,
        budget_assessment=budget_assessment,
        lifestyle_assessment=lifestyle_assessment,
        summary=rec.summary,
        warnings=rec.warnings,
        questions_to_ask_host=rec.questions_to_ask_host,
    )
