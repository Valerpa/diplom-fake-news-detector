from typing import List
from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.schemas.requests import VerifyRequest, GenerateQueriesRequest, RunWithQueriesRequest
from app.schemas.responses import VerifyResponse, EvidenceItem
from app.services.models.main_model import MainVerificationService

router   = APIRouter(prefix="/verify", tags=["verify"])
_service = MainVerificationService()
settings = get_settings()


class QueriesResponse(BaseModel):
    text:    str
    queries: List[str]


@router.post("/queries", response_model=QueriesResponse,
             summary="Step 1: generate search queries without running the search")
def generate_queries(req: GenerateQueriesRequest) -> QueriesResponse:

    queries = _service.generate_queries(req.text, req.num_queries)
    return QueriesResponse(text=req.text, queries=queries)


@router.post("/run", response_model=VerifyResponse,
             summary="Step 2+3: run search and scoring with a supplied query list")
def run_with_queries(req: RunWithQueriesRequest) -> VerifyResponse:

    result = _service.run_with_queries(
        news_text=req.text,
        queries=req.queries,
        num_results=req.num_results,
        threshold=settings.default_threshold,
    )
    return VerifyResponse(
        text=req.text,
        label=result["label"],
        probability=result["probability"],
        queries=result["queries"],
        evidence=[EvidenceItem(**{k: v for k, v in ev.items()
                                  if k in EvidenceItem.model_fields})
                  for ev in result["evidence"]],
        reasoning=result.get("reasoning", ""),
    )

@router.post("", response_model=VerifyResponse,
             summary="Full pipeline in one call (generate queries + search + score)")
def verify(req: VerifyRequest) -> VerifyResponse:

    result = _service.verify(
        news_text=req.text,
        num_queries=req.num_queries,
        num_results=req.num_results,
        threshold=settings.default_threshold,
    )
    return VerifyResponse(
        text=req.text,
        label=result["label"],
        probability=result["probability"],
        queries=result["queries"],
        evidence=[EvidenceItem(**{k: v for k, v in ev.items()
                                  if k in EvidenceItem.model_fields})
                  for ev in result["evidence"]],
        reasoning=result.get("reasoning", ""),
    )
