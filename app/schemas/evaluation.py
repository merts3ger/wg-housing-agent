from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.result import AgentAssessment


class ListingSnapshot(BaseModel):
    """Key fields from the listing, surfaced directly in the response."""

    title: str
    warm_rent: Optional[int] = Field(None, description="Total monthly rent incl. utilities in EUR")
    neighborhood: Optional[str] = None
    room_size_sqm: Optional[int] = None
    available_from: Optional[str] = None
    furnishing_status: Optional[str] = None
    commute_minutes: Optional[int] = Field(None, description="Estimated commute to user's destination")


class EvaluationResponse(BaseModel):
    listing_snapshot: ListingSnapshot
    budget_assessment: AgentAssessment
    lifestyle_assessment: AgentAssessment
    summary: str = Field(..., description="Moderator's final recommendation text")
    warnings: List[str] = Field(default_factory=list)
    questions_to_ask_host: List[str] = Field(default_factory=list)
