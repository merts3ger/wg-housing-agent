from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from app.schemas.profile import UserProfile
from app.schemas.result import FinalRecommendation
from app.schemas.evaluation import EvaluationResponse
from app.services.url_fetcher import fetch_listing_text_from_url
from app.services.evaluator import evaluate_url_for_default_profile, evaluate_url_for_profile, run_graph

router = APIRouter()


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


class EvaluateUrlForProfileRequest(BaseModel):
    url: str
    user_profile: UserProfile


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

    try:
        return run_graph(listing_text, request.user_profile)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/evaluate-url", response_model=EvaluationResponse, summary="Evaluate a WG-Gesucht URL using the default profile")
def evaluate_url(request: EvaluateUrlRequest) -> EvaluationResponse:
    try:
        return evaluate_url_for_default_profile(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/evaluate-url-for-profile", response_model=EvaluationResponse,
             summary="Evaluate a WG-Gesucht URL against a supplied profile")
def evaluate_url_for_profile_endpoint(request: EvaluateUrlForProfileRequest) -> EvaluationResponse:
    try:
        return evaluate_url_for_profile(request.url, request.user_profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc