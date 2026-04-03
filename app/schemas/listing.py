from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ParsedListing(BaseModel):
    title: str = Field(..., description="Listing title as shown in the source")
    warm_rent: Optional[int] = Field(None, description="Total monthly rent including utilities, in EUR")
    cold_rent: Optional[int] = Field(None, description="Base monthly rent excluding utilities, in EUR")
    deposit: Optional[int] = Field(None, description="Security deposit amount in EUR")
    neighborhood: Optional[str] = Field(None, description="Neighborhood or district name")
    address_text: Optional[str] = Field(None, description="Full or partial address as given in the listing")
    furnished: Optional[bool] = Field(None, description="Whether the apartment is furnished; None if unknown")
    furnishing_status: Optional[str] = Field(None, description="One of: 'furnished', 'partially_furnished', 'unfurnished', 'takeover_possible', or None")
    furniture_details: Optional[str] = Field(None, description="Short factual note about specific furniture mentioned, e.g. 'Desk and pallet bed can be taken over'")
    available_from: Optional[str] = Field(None, description="Move-in date as a string (e.g. '01.05.2025' or 'sofort')")
    amenities: List[str] = Field(default_factory=list, description="List of amenities mentioned in the listing")
    description_summary: Optional[str] = Field(None, description="Short summary of the listing description")
    missing_info: List[str] = Field(default_factory=list, description="Fields that could not be extracted from the listing")
    location_precision: Optional[str] = Field(
        None, description="Precision of the location data, e.g. 'exact', 'neighborhood', 'city'"
    )
    room_size_sqm: Optional[int] = Field(None, description="Room size in square meters")
    flatmate_count: Optional[int] = Field(None, description="Total number of people in the flatshare")
    flatshare_age_min: Optional[int] = Field(None, description="Minimum age of current flatmates, if stated")
    flatshare_age_max: Optional[int] = Field(None, description="Maximum age of current flatmates, if stated")
    flatshare_description_summary: Optional[str] = Field(None, description="Short factual summary of the household dynamic, social habits, and privacy boundaries as described by the flatmates")
    transit_access_mentioned: Optional[bool] = Field(None, description="True if the listing explicitly mentions nearby public transport; None if not mentioned")
    transit_access_summary: Optional[str] = Field(None, description="Short factual summary of what the listing says about public transport access, e.g. 'Bus stop 2 min walk, tram nearby'")
    transit_access_walk_minutes: Optional[int] = Field(None, description="Walking time to the nearest transit stop in minutes, if explicitly stated in the listing")
    city: Optional[str] = Field(None, description="City name extracted from the listing address or text, e.g. 'Kiel' or 'Hamburg'")


class EnrichedListing(ParsedListing):
    commute_minutes: Optional[int] = Field(None, description="Estimated commute time to the user's destination in minutes")
    commute_confidence: Optional[str] = Field(
        None, description="Confidence level of the commute estimate, e.g. 'high', 'medium', 'low'"
    )
    nearby_parks: List[str] = Field(default_factory=list, description="Names of parks within reasonable walking distance")
    nearby_swimming_pools: List[str] = Field(
        default_factory=list, description="Names of public swimming pools within reasonable distance"
    )
    nearby_transit: List[str] = Field(default_factory=list, description="Transit stops explicitly named in the listing text (not sourced from Azure Maps)")
    nearby_supermarkets: List[str] = Field(default_factory=list, description="Nearby supermarkets")
