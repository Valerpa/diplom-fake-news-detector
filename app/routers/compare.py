import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException
from app.core.config import get_settings
from app.schemas.requests import CompareRequest
from app.schemas.responses import CompareResponse, MethodResult
from app.services.models.main_model import MainVerificationService
from app.services.models.baselines import (
    RuBERTService, LLMClassifierService,
    ChainOfRAGService, STEELService, NLIClassifierService, GNNService,
)
from app.services.models.analysis import inter_method_agreement

router = APIRouter(prefix="/compare", tags=["compare"])
settings = get_settings()

_VERIFIERS: dict[str, object] = {
    "main": MainVerificationService(),
    "rubert": RuBERTService(),
    "llm_zeroshot": LLMClassifierService(),
    "corag": ChainOfRAGService(),
    "steel": STEELService(),
    "nli": NLIClassifierService(),
    "gnn": GNNService(),
}

VALID_METHODS = set(_VERIFIERS.keys())


@router.post("", response_model=CompareResponse,
             summary="Run multiple methods and return inter-method agreement")
def compare(req: CompareRequest) -> CompareResponse:
    threshold = settings.default_threshold

    unknown = set(req.methods) - VALID_METHODS
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown methods: {unknown}. Valid: {VALID_METHODS}"
        )

    method_results: dict[str, dict] = {}
    for name in req.methods:
        verifier = _VERIFIERS[name]
        try:
            if name == "main":
                result = verifier.verify(
                    req.text, num_queries=req.num_queries,
                    threshold=threshold
                )
            elif name in ("corag", "steel"):
                result = verifier.verify(
                    req.text, num_results=5, threshold=threshold
                )
            elif name in ("nli", "gnn"):
                result = verifier.verify(
                    req.text, num_queries=req.num_queries,
                    num_results=5, threshold=threshold
                )
            else:
                result = verifier.verify(req.text, threshold=threshold)
        except Exception as e:
            result = {
                "label": "ОШИБКА", "probability": None,
                "reasoning": str(e), "evidence": [], "queries": [],
            }
        method_results[name] = result

    # Compute agreement
    agreement = inter_method_agreement(method_results, threshold=threshold)

    method_result_list = [
        MethodResult(
            method=name,
            label=res.get("label", "ОШИБКА"),
            probability=res.get("probability"),
            reasoning=res.get("reasoning", ""),
        )
        for name, res in method_results.items()
    ]

    return CompareResponse(
        text=req.text,
        gold_label=req.gold_label,
        results=method_result_list,
        agreement_fraction=agreement["agreement_fraction"],
        consensus_label=agreement["consensus_label"],
        disagreement=agreement["disagreement"],
    )
