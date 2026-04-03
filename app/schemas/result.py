from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.listing import EnrichedListing


class AgentAssessment(BaseModel):
    agent_name: str = Field(..., description="Name of the specialist agent")
    listing_title: str = Field(..., description="Title of the assessed listing")
    score: float = Field(..., ge=0.0, le=10.0, description="Overall score from 0 to 10")
    fits_budget: bool | None = Field(
        None,
        description="Whether the listing fits the user's budget, if assessed by this agent",
    )
    fits_commute: bool | None = Field(
        None,
        description="Whether the commute fits the user's limit, if assessed by this agent",
    )
    pros: List[str] = Field(default_factory=list, description="List of positive aspects relative to user preferences")
    cons: List[str] = Field(default_factory=list, description="List of drawbacks relative to user preferences")
    notes: Optional[str] = Field(None, description="Additional context or caveats from the agent")


class FinalRecommendation(BaseModel):
    evaluated_listing: EnrichedListing | None = Field(
        None,
        description="The listing evaluated by the system",
    )
    assessments: List[AgentAssessment] = Field(default_factory=list, description="Assessments from all specialist agents for the current listing")
    summary: str = Field(..., description="Human-readable summary of the recommendation")
    warnings: List[str] = Field(default_factory=list, description="Any warnings or caveats the user should be aware of")
    questions_to_ask_host: List[str] = Field(default_factory=list, description="Follow-up questions the user should ask the host before deciding")
