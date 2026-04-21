from fastapi import APIRouter, HTTPException, UploadFile, File
import shutil, tempfile, os
from app.schemas.requests import BaselineRequest
from app.schemas.responses import BaselineResponse, EvidenceItem
from app.services.models.baselines import (
    RuBERTService,
    LLMClassifierService,
    ChainOfRAGService,
    STEELService,
    NLIClassifierService,
    GNNService,
)

router = APIRouter(prefix="/baselines", tags=["baselines"])

_rubert = RuBERTService()
_llm = LLMClassifierService()
_corag = ChainOfRAGService()
_steel = STEELService()
_nli = NLIClassifierService()
_gnn = GNNService()


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
def rubert(req: BaselineRequest) -> BaselineResponse:

    result = _rubert.verify(req.text, threshold=req.threshold)
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
        _rubert.train(tmp_path, text_col=text_col, label_col=label_col, epochs=epochs)
    finally:
        os.unlink(tmp_path)
    return {"status": "ok", "message": "RuBERT fine-tuned successfully."}


@router.post("/llm", response_model=BaselineResponse,
             summary="GigaChat zero-shot or few-shot classifier (no retrieval)")
def llm_classifier(req: BaselineRequest) -> BaselineResponse:

    result = _llm.verify(
        req.text,
        few_shot_examples=req.few_shot_examples,
        threshold=req.threshold,
    )
    return _to_response("llm", req.text, result)


@router.post("/corag", response_model=BaselineResponse,
             summary="Chain-of-RAG: iterative retrieval (RAGAR, ACL FEVER 2024)")
def corag(req: BaselineRequest) -> BaselineResponse:

    result = _corag.verify(
        req.text,
        max_rounds=req.max_rounds,
        num_results=req.num_results,
        threshold=req.threshold,
    )
    return _to_response("corag", req.text, result)


@router.post("/steel", response_model=BaselineResponse,
             summary="STEEL: multi-round retrieval with LLM relevance filtering")
def steel(req: BaselineRequest) -> BaselineResponse:

    result = _steel.verify(
        req.text,
        max_rounds=req.max_rounds,
        num_results=req.num_results,
        threshold=req.threshold,
    )
    return _to_response("steel", req.text, result)


@router.post("/nli", response_model=BaselineResponse,
             summary="NLI-based entailment classifier (mDeBERTa)")
def nli(req: BaselineRequest) -> BaselineResponse:

    result = _nli.verify(
        req.text,
        num_queries=req.num_queries,
        num_results=req.num_results,
        threshold=req.threshold,
    )
    return _to_response("nli", req.text, result)


@router.post("/gnn", response_model=BaselineResponse,
             summary="Graph Attention Network classifier (requires training)")
def gnn(req: BaselineRequest) -> BaselineResponse:

    result = _gnn.verify(
        req.text,
        num_queries=req.num_queries,
        num_results=req.num_results,
        threshold=req.threshold,
    )
    return _to_response("gnn", req.text, result)


@router.post("/gnn/train", summary="Train the GNN on a labeled CSV dataset")
async def gnn_train(
        file: UploadFile = File(..., description="CSV with 'text' and 'label' columns"),
        text_col: str = "text",
        label_col: str = "label",
        epochs: int = 20,
        num_queries: int = 5,
        num_results: int = 4,
):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        _gnn.train(tmp_path, text_col=text_col, label_col=label_col,
                   epochs=epochs, num_queries=num_queries, num_results=num_results)
    finally:
        os.unlink(tmp_path)
    return {"status": "ok", "message": "GNN trained successfully."}
