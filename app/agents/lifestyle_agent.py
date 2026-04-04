import json
import logging

from app.schemas.profile import UserProfile
from app.schemas.listing import EnrichedListing
from app.schemas.result import AgentAssessment
from app.services import azure_openai_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a lifestyle and daily-fit assessment specialist for WG / student-room rental listings in Germany.

You will receive structured data about a listing and the user's preferences.
Reason about how well this listing fits the user's everyday quality of life — not finances or legal risk.

Return a JSON object matching this schema exactly:

{
  "score":        float,    // 1.0-10.0; see scoring guidance below
  "fits_commute": boolean,  // true if commute_minutes <= max_commute_minutes, null if commute unknown
  "pros":         string[], // concrete positive observations, one sentence each
  "cons":         string[], // concrete negative observations, one sentence each
  "notes":        string    // 1-3 sentences summarising fit, tradeoffs, and any relevant context
}

Assess these dimensions in roughly this order:

0. Location precision — use location_precision as a confidence modifier for all location-dependent conclusions:
   - "exact": commute time and nearby amenities can be treated as reliable.
   - "neighborhood": treat commute and nearby amenities as approximate; note this when relevant.
   - "city" or null: commute and nearby amenity data are highly uncertain; flag this clearly in notes.
   Always consider location_precision before drawing conclusions from commute_minutes or nearby_* fields.
   The neighborhood and address_text fields give context for how the listing is situated in the city.

1. Commute burden — treat commute as a quality-of-life factor, not just a pass/fail.
   A short commute is a meaningful daily benefit. A long one is a real burden.
   Take both commute_confidence and location_precision into account: low confidence or imprecise location means the estimate is uncertain — say so in notes.

2. Nearby places — parks and swimming pools matter only if the user has stated they want them.
   If wanted and present, it is a clear pro. If wanted and absent, it is a clear con.
   When location_precision is not "exact", qualify nearby amenity conclusions as approximate.
   Transit access: use transit_access_mentioned, transit_access_summary, and transit_access_walk_minutes
   from the listing text. If transit is mentioned, treat it as a moderate positive and include the summary.
   If transit_access_mentioned is false or null, note its absence but do not penalise heavily.

3. Convenience amenities — things like dishwasher, washing machine, wifi, fitted kitchen, airfryer,
   large living room, balcony, and similar improve daily comfort. Mention the most relevant ones.

4. Social and household fit — reason about the household dynamic based on:
   - flatshare_description_summary (how flatmates describe their shared life)
   - flatmate_count (a larger WG — 4 or 5+ people — is inherently more social and busier)
   - flatmate age range

5. Lifestyle preference alignment — the user has stated a lifestyle_preference (e.g. "quiet", "social", "urban").
   Reason about whether what is described in the listing matches or conflicts with that preference.
   Only use what is stated — do not invent household character.

Scoring guidance — use the full range:
- 9-10: outstanding fit across most dimensions; no significant tradeoffs
- 7-8.5: good fit with minor tradeoffs or a missing signal
- 5-6.5: mixed fit; some clear positives but also meaningful gaps or tensions
- 3-4.5: weak fit; significant mismatches in commute, social atmosphere, or stated preferences
- 1-2.5: poor fit; hard lifestyle conflicts or multiple unmet important preferences
- A listing with several unresolved tradeoffs should not score above 7 by default.
- Reserve 8.5+ for listings where the household description, commute, and nearby amenities all align well.

Social WG rule — strictly enforced:
- Evaluate the actual household behaviour: shared meals, film nights, regular group activities — these are social signals regardless of what the description calls itself.
- A large WG (flatmate_count >= 4) with recurring shared activities is meaningfully social, even if it claims to respect privacy.
- For a user with lifestyle_preference = "quiet": a social WG of this kind should be listed as a con and should lower the score to the 5-6.5 range or below. "Respects privacy" softens the downside modestly — it does not eliminate it.

Each-fact-once rule — strictly enforced:
- Every observation must appear in exactly one field: pros, cons, or notes. Never describe the same fact in two fields.
- Mixed or nuanced facts belong in notes only — do not also add them to pros or cons.
  - Social-but-privacy-respecting household for a quiet-preference user → cons only (social character is the relevant signal).
  - Social-but-privacy-respecting for a social-preference user → pros only (the privacy caveat need not be repeated in notes).
  - Commute that fits the limit but is close to it → pros only, with the margin noted inline (e.g. "27 min — within 30 min limit"). Do not also put it in notes.
  - Commute that exceeds the limit but only slightly → cons only, with the actual figure.
  - Furnishing takeover possible → lifestyle agent should not assess furnishing; leave it to the budget agent.
- Reserve pros for unambiguous positives and cons for unambiguous negatives. Anything that is genuinely ambiguous belongs in notes instead of straddling both lists.

Lifestyle preference alignment rule:
- A user with lifestyle_preference = "quiet" should score lower on any listing with a social household identity, regardless of privacy disclaimers.
- Conversely, a social user should score lower on listings that emphasise strong independence and quiet norms.
- If the household description is neutral or missing, do not penalise — just note the uncertainty.

Rules:
- Do not assess budget, deposit, or financial value — those are handled by another agent.
- Do not hallucinate facts. Only reason from what is provided in the input.
- If a dimension has no data (e.g. no flatshare summary, no age info), skip it or note it briefly.
- Return JSON only. No markdown, no explanation outside the JSON.
"""



def assess_lifestyle_and_daily_fit(
    user_profile: UserProfile,
    enriched_listing: EnrichedListing,
) -> AgentAssessment:
    result = _llm_assess(user_profile, enriched_listing)
    if result is not None:
        return result

    logger.warning("LLM lifestyle assessment unavailable, falling back to rule-based.")
    return _rule_based_assess(user_profile, enriched_listing)


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

def _build_input(user_profile: UserProfile, listing: EnrichedListing) -> str:
    data = {
        "user_lifestyle_preference": user_profile.lifestyle_preference,
        "user_max_commute_minutes": user_profile.max_commute_minutes,
        "user_wants_park_nearby": user_profile.wants_park_nearby,
        "user_wants_swimming_pool_nearby": user_profile.wants_swimming_pool_nearby,
"user_preferred_flatmate_age_min": user_profile.preferred_flatmate_age_min,
        "user_preferred_flatmate_age_max": user_profile.preferred_flatmate_age_max,
        "neighborhood": listing.neighborhood,
        "address_text": listing.address_text,
        "location_precision": listing.location_precision,
        "commute_minutes": listing.commute_minutes,
        "commute_confidence": listing.commute_confidence,
        "nearby_parks": listing.nearby_parks,
        "nearby_swimming_pools": listing.nearby_swimming_pools,
        "transit_access_mentioned": listing.transit_access_mentioned,
        "transit_access_summary": listing.transit_access_summary,
        "transit_access_walk_minutes": listing.transit_access_walk_minutes,
        "amenities": listing.amenities,
        "flatmate_count": listing.flatmate_count,
        "flatshare_age_min": listing.flatshare_age_min,
        "flatshare_age_max": listing.flatshare_age_max,
        "flatshare_description_summary": listing.flatshare_description_summary,
        "description_summary": listing.description_summary,
    }
    return json.dumps(data, ensure_ascii=False)


def _llm_assess(user_profile: UserProfile, listing: EnrichedListing) -> AgentAssessment | None:
    if not azure_openai_client.is_configured():
        return None

    try:
        client = azure_openai_client.get_client()
        deployment = azure_openai_client.get_deployment()

        response = client.chat.completions.create(
            model=deployment,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_input(user_profile, listing)},
            ],
            temperature=0,
        )

        data = json.loads(response.choices[0].message.content)
        fits_commute = data.get("fits_commute")
        if fits_commute is not None:
            fits_commute = bool(fits_commute)

        return AgentAssessment(
            agent_name="lifestyle_and_daily_fit",
            listing_title=listing.title,
            score=float(data["score"]),
            fits_budget=None,
            fits_commute=fits_commute,
            pros=data.get("pros", []),
            cons=data.get("cons", []),
            notes=data.get("notes"),
        )
    except Exception as exc:
        logger.warning("LLM lifestyle assessment failed (%s: %s).", type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

_CONVENIENCE_AMENITIES = {
    "dishwasher", "washing machine", "fitted kitchen", "wifi included",
    "elevator", "bike storage", "parking",
}
_SHARED_SPACE_AMENITIES = {"balcony", "terrace", "garden", "large living room"}


def _rule_based_assess(user_profile: UserProfile, listing: EnrichedListing) -> AgentAssessment:
    pros: list[str] = []
    cons: list[str] = []
    notes_parts: list[str] = []

    fits_commute: bool | None = None
    if listing.commute_minutes is not None:
        fits_commute = listing.commute_minutes <= user_profile.max_commute_minutes
        confidence = f" ({listing.commute_confidence} confidence)" if listing.commute_confidence else ""
        if fits_commute:
            pros.append(f"Commute {listing.commute_minutes} min ≤ limit {user_profile.max_commute_minutes} min{confidence}")
        else:
            cons.append(f"Commute {listing.commute_minutes} min exceeds limit {user_profile.max_commute_minutes} min{confidence}")
    else:
        notes_parts.append("Commute could not be estimated")

    if user_profile.wants_park_nearby:
        if listing.nearby_parks:
            pros.append(f"Nearby parks: {', '.join(listing.nearby_parks)}")
        else:
            cons.append("No nearby parks found, but user wants one")

    if user_profile.wants_swimming_pool_nearby:
        if listing.nearby_swimming_pools:
            pros.append(f"Nearby pools: {', '.join(listing.nearby_swimming_pools)}")
        else:
            cons.append("No nearby swimming pools found, but user wants one")

    if listing.nearby_transit:
        pros.append(f"Transit access: {', '.join(listing.nearby_transit[:3])}")

    amenities_lower = {a.lower() for a in listing.amenities}
    convenience = [a for a in _CONVENIENCE_AMENITIES if a in amenities_lower]
    shared_space = [a for a in _SHARED_SPACE_AMENITIES if a in amenities_lower]
    if convenience:
        pros.append(f"Convenience amenities: {', '.join(convenience)}")
    if shared_space:
        pros.append(f"Shared space: {', '.join(shared_space)}")

    if listing.flatshare_age_min is not None or listing.flatshare_age_max is not None:
        notes_parts.append(f"Current flatmates aged {_fmt_age_range(listing.flatshare_age_min, listing.flatshare_age_max)}")

    if listing.flatshare_description_summary:
        notes_parts.append(f"Household: {listing.flatshare_description_summary}")

    score = round(max(1.0, min(10.0, 6.5 + len(pros) * 0.3 - len(cons) * 1.2)), 1)

    return AgentAssessment(
        agent_name="lifestyle_and_daily_fit",
        listing_title=listing.title,
        score=score,
        fits_budget=None,
        fits_commute=fits_commute,
        pros=pros,
        cons=cons,
        notes="; ".join(notes_parts) if notes_parts else None,
    )


def _fmt_age_range(min_age: int | None, max_age: int | None) -> str:
    if min_age is not None and max_age is not None:
        return f"{min_age}–{max_age}"
    if min_age is not None:
        return f"{min_age}+"
    if max_age is not None:
        return f"up to {max_age}"
    return "unknown"
