from app.schemas.listing import EnrichedListing
from app.schemas.result import AgentAssessment, FinalRecommendation


def synthesize_recommendation(
    budget_assessment: AgentAssessment,
    lifestyle_assessment: AgentAssessment,
    enriched_listing: EnrichedListing | None = None,
) -> FinalRecommendation:
    # TODO: Replace logic with LLM-based synthesis.

    assessments = [budget_assessment, lifestyle_assessment]
    composite_score = sum(a.score for a in assessments) / len(assessments)

    warnings: list[str] = []
    if budget_assessment.fits_budget is False:
        warnings.append("Listing exceeds the user's budget.")
    if lifestyle_assessment.fits_commute is False:
        warnings.append("Commute time exceeds the user's limit.")

    questions = _generate_host_questions(enriched_listing)

    over_budget = budget_assessment.fits_budget is False
    commute_too_long = lifestyle_assessment.fits_commute is False

    if over_budget:
        verdict = "only worth considering if your budget is flexible"
    elif commute_too_long:
        verdict = "worth considering with caveats"
    elif composite_score >= 6.0:
        verdict = "recommended"
    elif composite_score >= 4.0:
        verdict = "worth considering with caveats"
    else:
        verdict = "not recommended"

    budget_label = "OK" if budget_assessment.fits_budget else "OVER" if budget_assessment.fits_budget is False else "UNKNOWN"
    commute_label = "OK" if lifestyle_assessment.fits_commute else "TOO LONG" if lifestyle_assessment.fits_commute is False else "UNKNOWN"
    open_points = f" {len(questions)} open question(s) to clarify with host." if questions else ""

    precision = enriched_listing.location_precision if enriched_listing else None
    if precision == "city" or precision is None:
        location_note = " Location is approximate — commute and nearby-amenity conclusions may not be accurate."
    elif precision == "neighborhood":
        location_note = " Location is neighbourhood-level — commute and nearby amenities are approximate."
    else:
        location_note = ""

    summary = (
        f"Composite score: {composite_score:.1f}/10. "
        f"Overall verdict: {verdict}. "
        f"Budget: {budget_label}. "
        f"Commute: {commute_label}."
        f"{open_points}"
        f"{location_note}"
    )

    return FinalRecommendation(
        evaluated_listing=enriched_listing,
        assessments=assessments,
        summary=summary,
        warnings=warnings,
        questions_to_ask_host=questions,
    )


def _generate_host_questions(listing: EnrichedListing | None) -> list[str]:
    """
    Derive a short list of follow-up questions for genuinely unresolved points.
    Only include questions that are practically useful to ask the host.
    """
    if listing is None:
        return []

    questions: list[str] = []

    if listing.deposit is None:
        questions.append("What is the deposit amount, and when is it due?")

    if listing.furnishing_status == "takeover_possible":
        detail = f" ({listing.furniture_details})" if listing.furniture_details else ""
        questions.append(f"Is the furniture takeover{detail} confirmed, and is there an extra cost?")

    if listing.location_precision in ("city", None):
        questions.append("Could you share the exact address or street so I can check the commute?")

    if listing.available_from is None:
        questions.append("When is the room available from?")

    for field in listing.missing_info:
        if field == "rent":
            questions.append("What is the total monthly rent (warm), and what utilities are included?")

    return questions
