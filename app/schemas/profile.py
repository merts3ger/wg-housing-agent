from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    budget_eur: int = Field(..., gt=0, description="Maximum monthly budget in EUR (warm rent)")
    max_commute_minutes: int = Field(..., gt=0, le=180, description="Maximum acceptable commute time in minutes")
    prefers_furnished: bool = Field(..., description="Whether the user prefers a furnished apartment")
    lifestyle_preference: str = Field(..., description="Lifestyle preference, e.g. 'quiet', 'social', 'urban'")
    wants_park_nearby: bool = Field(..., description="Whether a nearby park is important to the user")
    wants_swimming_pool_nearby: bool = Field(..., description="Whether a nearby swimming pool is important to the user")
    commute_destination: str = Field(..., description="Address or area the user commutes to (e.g. workplace)")
    preferred_flatmate_age_min: int | None = Field(None, gt=0, description="Minimum preferred flatmate age")
    preferred_flatmate_age_max: int | None = Field(None, gt=0, description="Maximum preferred flatmate age")
