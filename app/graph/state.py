from __future__ import annotations

from typing import List, Optional
from typing_extensions import TypedDict

from app.schemas.profile import UserProfile
from app.schemas.listing import ParsedListing, EnrichedListing
from app.schemas.result import AgentAssessment, FinalRecommendation


class HousingGraphState(TypedDict):
    # Input
    raw_listing_text: str
    user_profile: UserProfile

    # Parsing & enrichment
    parsed_listing: Optional[ParsedListing]
    enriched_listing: Optional[EnrichedListing]

    # Agent assessments
    budget_assessment: Optional[AgentAssessment]
    lifestyle_assessment: Optional[AgentAssessment]

    # Output
    final_recommendation: Optional[FinalRecommendation]

    # Error accumulation
    errors: List[str]
