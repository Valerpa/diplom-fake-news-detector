import asyncio
from fastapi import APIRouter, HTTPException, UploadFile, File
import shutil, tempfile, os
from app.core.config import get_settings
from app.schemas.requests import BaselineRequest
from app.schemas.responses import BaselineResponse, EvidenceItem
from app.services.models.baselines import (
    RuBERTService,
    LLMClassifierService,
    ChainOfRAGService,
    NLIClassifierService,
)

router = APIRouter(prefix="/baselines", tags=["baselines"])
settings = get_settings()

_rubert = RuBERTService()
_llm = LLMClassifierService()
_corag = ChainOfRAGService()
_nli = NLIClassifierService()


def _to_response(method: str, text: str, result: dict) -> BaselineResponse:
    return BaselineResponse(
        method=method,
        text=text,
        label=result["label"],
        probability=result["probability"],
        evidence=[
            EvidenceItem(**{k: v for k, v in ev.items()
                            if k in EvidenceItem.model_fields})
            for ev in result.get("evidence", [])
        ],
        reasoning=result.get("reasoning", ""),
    )


@router.post("/rubert", response_model=BaselineResponse,
             summary="Fine-tuned RuBERT content-only classifier")
async def rubert(req: BaselineRequest) -> BaselineResponse:

    result = await asyncio.to_thread(
        _rubert.verify, req.text, threshold=settings.default_threshold
    )
    return _to_response("rubert", req.text, result)


@router.post("/rubert/train", summary="Fine-tune RuBERT on a labeled CSV dataset")
async def rubert_train(
        file: UploadFile = File(..., description="CSV with 'text' and 'label' columns"),
        text_col: str = "text",
        label_col: str = "label",
        epochs: int = 3,
):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: _rubert.train(tmp_path, text_col=text_col, label_col=label_col, epochs=epochs)
        )
    finally:
        os.unlink(tmp_path)
    return {"status": "ok", "message": "RuBERT fine-tuned successfully."}


@router.post("/llm", response_model=BaselineResponse,
             summary="GigaChat zero-shot or few-shot classifier (no retrieval)")
async def llm_classifier(req: BaselineRequest) -> BaselineResponse:

    result = await _llm.verify(
        req.text,
        few_shot_examples=req.few_shot_examples,
        threshold=settings.default_threshold,
    )
    return _to_response("llm", req.text, result)


@router.post("/corag", response_model=BaselineResponse,
             summary="Chain-of-RAG: iterative retrieval (RAGAR, ACL FEVER 2024)")
async def corag(req: BaselineRequest) -> BaselineResponse:

    result = await _corag.verify(
        req.text,
        max_rounds=req.max_rounds,
        num_results=req.num_results,
        threshold=settings.default_threshold,
    )
    return _to_response("corag", req.text, result)


@router.post("/nli", response_model=BaselineResponse,
             summary="NLI-based entailment classifier (mDeBERTa)")
async def nli(req: BaselineRequest) -> BaselineResponse:

    result = await _nli.verify(
        req.text,
        num_queries=req.num_queries,
        num_results=req.num_results,
        threshold=settings.default_threshold,
    )
    return _to_response("nli", req.text, result)
