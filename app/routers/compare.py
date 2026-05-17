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

_VERIFIERS = {
    "main": MainVerificationService(),
    "rubert": RuBERTService(),
    "llm_zeroshot": LLMClassifierService(),
    "corag": ChainOfRAGService(),
    "steel": STEELService(),
    "nli": NLIClassifierService(),
    "gnn": GNNService(),
}

VALID_METHODS = set(_VERIFIERS.keys())


async def _run_one(name: str, text: str, num_queries: int,
                   threshold: float) -> tuple[str, dict]:
    verifier = _VERIFIERS[name]
    try:
        if name == "main":
            result = await verifier.verify(
                text, num_queries=num_queries, threshold=threshold
            )
        elif name == "rubert":
            result = await asyncio.to_thread(
                verifier.verify, text, threshold
            )
        elif name in ("corag", "steel"):
            result = await verifier.verify(
                text, num_results=5, threshold=threshold
            )
        elif name in ("nli", "gnn"):
            result = await verifier.verify(
                text, num_queries=num_queries,
                num_results=5, threshold=threshold
            )
        else:
            result = await verifier.verify(text, threshold=threshold)
    except Exception as e:
        result = {
            "label": "ОШИБКА", "probability": None,
            "reasoning": str(e), "evidence": [], "queries": [],
        }
    return name, result


@router.post("", response_model=CompareResponse)
async def compare(req: CompareRequest):
    threshold = settings.default_threshold

    unknown = set(req.methods) - VALID_METHODS
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown methods: {unknown}. Valid: {VALID_METHODS}"
        )

    # Все методы запускаются параллельно
    tasks = [
        _run_one(name, req.text, req.num_queries, threshold)
        for name in req.methods
    ]
    results = await asyncio.gather(*tasks)

    method_results = dict(results)
    agreement = inter_method_agreement(method_results, threshold)

    return CompareResponse(
        text=req.text,
        gold_label=req.gold_label,
        results=[
            MethodResult(
                method=name,
                label=res.get("label", "ОШИБКА"),
                probability=res.get("probability"),
                reasoning=res.get("reasoning", ""),
            )
            for name, res in method_results.items()
        ],
        agreement_fraction=agreement["agreement_fraction"],
        consensus_label=agreement["consensus_label"],
        disagreement=agreement["disagreement"],
    )
