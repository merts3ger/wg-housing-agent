from app.graph.state import HousingGraphState
from app.services.listing_parser import parse_listing
from app.services.enrichment import enrich_listing
from app.agents.budget_agent import assess_budget_and_value
from app.agents.lifestyle_agent import assess_lifestyle_and_daily_fit
from app.agents.moderator_agent import synthesize_recommendation


def parse_listing_node(state: HousingGraphState) -> dict:
    parsed = parse_listing(state["raw_listing_text"])
    return {"parsed_listing": parsed}


def enrich_listing_node(state: HousingGraphState) -> dict:
    enriched = enrich_listing(state["parsed_listing"], state["user_profile"])
    return {"enriched_listing": enriched}


def budget_assessment_node(state: HousingGraphState) -> dict:
    assessment = assess_budget_and_value(state["user_profile"], state["enriched_listing"])
    return {"budget_assessment": assessment}


def lifestyle_assessment_node(state: HousingGraphState) -> dict:
    assessment = assess_lifestyle_and_daily_fit(state["user_profile"], state["enriched_listing"])
    return {"lifestyle_assessment": assessment}


def moderator_node(state: HousingGraphState) -> dict:
    recommendation = synthesize_recommendation(
        budget_assessment=state["budget_assessment"],
        lifestyle_assessment=state["lifestyle_assessment"],
        enriched_listing=state["enriched_listing"],
    )
    return {"final_recommendation": recommendation}
