import json
import logging

from app.schemas.profile import UserProfile
from app.schemas.listing import EnrichedListing
from app.schemas.result import AgentAssessment
from app.services import azure_openai_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a budget and value assessment specialist for WG / student-room rental listings in Germany.

You will receive structured data about a listing and the user's budget. Assess how good this listing is from a financial and value-for-money perspective.

Return a JSON object matching this schema exactly:

{
  "score":       float,    // 1.0-10.0
  "fits_budget": boolean,  // true if warm_rent (or cold_rent when warm is missing) <= user_budget_eur
  "pros":        string[], // concrete positive observations, one sentence each
  "cons":        string[], // concrete negative observations, one sentence each
  "notes":       string    // 1-2 sentence summary covering only facts not already stated in pros/cons
}

Scoring anchor — apply in this order:
1. Budget fit: if warm_rent <= user_budget_eur, anchor the score based on headroom:
     - < 15% below budget  → anchor at 5.0
     - 15–39% below budget → anchor at 5.0–5.5
     - ≥ 40% below budget  → anchor at 5.5–6.0
   If over budget, anchor at 2.0–3.5.
2. Room size / EUR/m²: adjust from the anchor. Small room (< 10 m²) pulls the score down by roughly
   0.5–1.0. High EUR/m² (> 35) pulls the score down by roughly 0.5–1.0. Do not let value-for-space
   concerns override a strong budget fit — they temper it, not reverse it.
3. Furnishing: apply based on user preference if known.
     - "furnished" and user prefers furnished         = +2.0
     - "partially_furnished" and user prefers furnished = +1.0
     - "takeover_possible"                            = +0.5
     - "unfurnished" and user prefers furnished       = -1.0
     - "furnished" and user prefers unfurnished       = -0.5
     - preference unknown: apply no adjustment; do not guess.
4. Deposit: a small adjustment.
     - deposit > 3 months rent  = -0.5
     - deposit unknown          = -0.2
5. Hard cap: after all adjustments, if warm_rent > user_budget_eur, clamp the final score to a maximum of 4.0 regardless of other factors.

Output rules — strictly enforced:
- EACH FACT ONCE: every observation must appear in exactly one field — pros, cons, or notes. Never
  describe the same fact (deposit, room size, EUR/m², furnishing) in two separate fields, even with
  different wording.

- VALUE-FOR-SPACE (room size and EUR/m² treated as one composite observation — never split across
  both a pro and a con):
    - EUR/m² < 25       → pros. Mention both room size and EUR/m². Strong positive framing.
    - 25 ≤ EUR/m² ≤ 35  → pros. Mention both room size and EUR/m². Mild positive framing
                          (e.g. "fair value for space").
    - EUR/m² > 35       → cons. Mention EUR/m². Do NOT separately comment on room size as a
                          positive — doing so would contradict the con.
    - room < 10 m²      → always add a separate con for the small room size, regardless of
                          EUR/m². If EUR/m² ≤ 35, both the value-for-space pro and the small-room
                          con can coexist. If EUR/m² > 35, the single cons entry should cover both
                          the high price-per-m² and the small size together.
    - room size unknown → skip the room size component; still apply the EUR/m² rule above.

- DEPOSIT rules (pick exactly one field per deposit situation):
    - deposit_eur is null/unknown → cons only.
      Exact wording: "Deposit unknown — upfront financial commitment is unclear."
      Do not also mention it in notes.
    - deposit is known and ≤ 2 months rent → notes only. Unremarkable; do not list as pro or con.
    - deposit is known and 2–3 months rent → notes only, with the EUR amount and months figure.
    - deposit is known and > 3 months rent → cons only. Do not also mention it in notes.

- FURNISHING: "takeover_possible" → notes only (uncertain; neither a clear pro nor a clear con).
  Do not list it as a pro or a con.

- NOTES field: summarize only facts not already stated in pros or cons — typically the budget
  headroom figure, or the EUR/m² figure if it fell in the neutral 25–35 band and was already
  covered in pros. Do not restate pros or cons in different words.

- NULL / MISSING DATA:
    - If both warm_rent and cold_rent are null: set fits_budget to null, add a con stating that
      rent information is missing.
    - If room_size is null: skip the room size component of the value-for-space rule entirely;
      still apply EUR/m² if calculable.

- Use concrete numbers (EUR amounts, m², EUR/m²) wherever available.
- AMENITIES: do not list amenities (kitchen, bike storage, washing machine, wifi, etc.) as pros.
  Amenities are a lifestyle signal, not a financial one. Ignore the amenities field entirely.
- Do not describe the neighbourhood, commute, or social atmosphere — those are assessed by other
  agents.
- Do not hallucinate values — only reason about what is provided in the input.
- Return JSON only. No markdown, no explanation outside the JSON.
"""


def assess_budget_and_value(
    user_profile: UserProfile,
    enriched_listing: EnrichedListing,
) -> AgentAssessment:
    result = _llm_assess(user_profile, enriched_listing)
    if result is not None:
        return result

    logger.warning("LLM budget assessment unavailable, falling back to rule-based.")
    return _rule_based_assess(user_profile, enriched_listing)


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

def _build_input(user_profile: UserProfile, listing: EnrichedListing) -> str:
    rent = listing.warm_rent or listing.cold_rent
    price_per_sqm = round(rent / listing.room_size_sqm, 1) if rent and listing.room_size_sqm else None
    deposit_months = round(listing.deposit / rent, 1) if listing.deposit and rent else None

    data = {
        "user_budget_eur": user_profile.budget_eur,
        "user_prefers_furnished": user_profile.prefers_furnished,
        "warm_rent": listing.warm_rent,
        "cold_rent": listing.cold_rent,
        "room_size_sqm": listing.room_size_sqm,
        "price_per_sqm_eur": price_per_sqm,
        "deposit_eur": listing.deposit,
        "deposit_months": deposit_months,
        "furnishing_status": listing.furnishing_status,
        "furniture_details": listing.furniture_details,
        "amenities": listing.amenities,
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
        return AgentAssessment(
            agent_name="budget_and_value",
            listing_title=listing.title,
            score=float(data["score"]),
            fits_budget=bool(data["fits_budget"]),
            fits_commute=None,
            pros=data.get("pros", []),
            cons=data.get("cons", []),
            notes=data.get("notes"),
        )
    except Exception as exc:
        logger.warning("LLM budget assessment failed (%s: %s).", type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _rule_based_assess(user_profile: UserProfile, listing: EnrichedListing) -> AgentAssessment:
    pros: list[str] = []
    cons: list[str] = []
    notes_parts: list[str] = []

    rent = listing.warm_rent or listing.cold_rent
    fits_budget = rent is not None and rent <= user_profile.budget_eur

    budget_score = _budget_score(rent, user_profile.budget_eur, pros, cons, notes_parts)
    value_score = _value_score(rent, listing.room_size_sqm, pros, cons, notes_parts)
    furnishing_score = _furnishing_score(listing, user_profile, pros, cons, notes_parts)
    deposit_score = _deposit_score(rent, listing.deposit, notes_parts)

    score = round(max(1.0, min(10.0, budget_score + value_score + furnishing_score + deposit_score)), 1)

    return AgentAssessment(
        agent_name="budget_and_value",
        listing_title=listing.title,
        score=score,
        fits_budget=fits_budget,
        fits_commute=None,
        pros=pros,
        cons=cons,
        notes="; ".join(notes_parts),
    )


def _budget_score(rent, budget, pros, cons, notes):
    if rent is None:
        cons.append("Rent information is missing")
        notes.append("Cannot assess budget fit without rent")
        return 2.0
    if rent > budget:
        overage = rent - budget
        overage_pct = overage / budget
        cons.append(f"Rent {rent} EUR exceeds budget by {overage} EUR")
        notes.append(f"Over budget by {overage} EUR ({overage_pct:.0%})")
        return max(1.0, 2.0 - overage_pct * 2.0)
    headroom = budget - rent
    headroom_pct = headroom / budget
    pros.append(f"Rent {rent} EUR fits budget ({headroom} EUR headroom, {headroom_pct:.0%})")
    notes.append(f"Within budget with {headroom_pct:.0%} headroom")
    return 5.5 if headroom_pct >= 0.40 else 5.0 if headroom_pct >= 0.20 else 4.5


def _value_score(rent, sqm, pros, cons, notes):
    if not rent or not sqm:
        return 0.0
    ppsm = rent / sqm
    if sqm < 10:
        cons.append(f"Very small room: {sqm} m²")
        notes.append(f"Tiny room ({sqm} m²) at {ppsm:.1f} EUR/m²")
        return -0.5
    if ppsm < 15:
        pros.append(f"Excellent value: {sqm} m² at {ppsm:.1f} EUR/m²")
        return 1.5
    elif ppsm < 25:
        pros.append(f"Good value: {sqm} m² at {ppsm:.1f} EUR/m²")
        return 0.8
    elif ppsm < 35:
        pros.append(f"Fair value: {sqm} m² at {ppsm:.1f} EUR/m²")
        return 0.2
    elif ppsm < 50:
        cons.append(f"Below-average value: {sqm} m² at {ppsm:.1f} EUR/m²")
        return -0.5
    cons.append(f"Poor value: {sqm} m² at {ppsm:.1f} EUR/m²")
    return -1.0


def _furnishing_score(listing, user_profile, pros, cons, notes):
    status = listing.furnishing_status
    prefers = user_profile.prefers_furnished
    detail_s = f": {listing.furniture_details}" if listing.furniture_details else ""
    detail_p = f" ({listing.furniture_details})" if listing.furniture_details else ""
    if status == "furnished":
        pros.append("Furnished, matches preference" if prefers else "Furnished (convenient)")
        return 1.0 if prefers else 0.3
    if status == "partially_furnished":
        pros.append(f"Partially furnished{detail_s}")
        return 0.5 if prefers else 0.2
    if status == "takeover_possible":
        notes.append(f"Furniture takeover possible{detail_p} — not guaranteed")
        return 0.2
    if status == "unfurnished" and prefers:
        cons.append("Unfurnished, user prefers furnished")
        return -0.5
    return 0.0


def _deposit_score(rent, deposit, notes):
    if deposit is None:
        notes.append("Deposit unknown — financial commitment unclear")
        return -0.3
    if rent and rent > 0:
        months = deposit / rent
        if months > 3:
            notes.append(f"High deposit: {deposit} EUR ({months:.1f} months)")
            return -0.5
        if months > 2:
            notes.append(f"Above-average deposit: {deposit} EUR ({months:.1f} months)")
            return -0.2
        notes.append(f"Deposit {deposit} EUR ({months:.1f} months) — reasonable")
        return 0.3
    notes.append(f"Deposit stated: {deposit} EUR")
    return 0.1
