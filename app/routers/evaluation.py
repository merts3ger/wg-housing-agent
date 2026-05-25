from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from app.schemas.profile import UserProfile
from app.schemas.result import FinalRecommendation
from app.schemas.evaluation import EvaluationResponse
from app.graph.state import HousingGraphState
from app.graph.graph_builder import build_housing_graph
from app.services.url_fetcher import fetch_listing_text_from_url
from app.services.evaluator import evaluate_url_for_default_profile

router = APIRouter()

# Transitional: single compiled graph for /evaluate.
# Step 2 will collapse this into evaluator.py.
_housing_graph = build_housing_graph()


class EvaluateRequest(BaseModel):
    raw_listing_text: Optional[str] = None
    url: Optional[str] = None
    user_profile: UserProfile

    @model_validator(mode="after")
    def require_text_or_url(self) -> "EvaluateRequest":
        if not self.raw_listing_text and not self.url:
            raise ValueError("Provide either 'raw_listing_text' or 'url'.")
        return self


class EvaluateUrlRequest(BaseModel):
    url: str


@router.post("/evaluate", response_model=FinalRecommendation, summary="Evaluate one housing listing")
def evaluate(request: EvaluateRequest) -> FinalRecommendation:
    if request.url:
        try:
            listing_text = fetch_listing_text_from_url(request.url)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch listing URL: {exc}") from exc
    else:
        listing_text = request.raw_listing_text  # type: ignore[assignment]

    initial_state: HousingGraphState = {
        "raw_listing_text": listing_text,
        "user_profile": request.user_profile,
        "parsed_listing": None,
        "enriched_listing": None,
        "budget_assessment": None,
        "lifestyle_assessment": None,
        "final_recommendation": None,
        "errors": [],
    }

    result = _housing_graph.invoke(initial_state)

    final_recommendation = result.get("final_recommendation")
    if final_recommendation is None:
        raise HTTPException(status_code=500, detail="Graph completed without producing a recommendation.")

    return final_recommendation


@router.post("/evaluate-url", response_model=EvaluationResponse, summary="Evaluate a WG-Gesucht URL using the default profile")
def evaluate_url(request: EvaluateUrlRequest) -> EvaluationResponse:
    try:
        return evaluate_url_for_default_profile(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
