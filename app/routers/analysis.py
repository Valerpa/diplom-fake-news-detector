import asyncio
from fastapi import APIRouter
from app.schemas.requests import AnalysisRequest
from app.schemas.responses import (
    AttributionItem,
    SpanAnalysisResponse,
    HeatmapResponse, HeatmapCell,
    SensitivityResponse, SensitivityTrial,
    CredibilityResponse,
    EvidenceItem,
)
from app.services.models.main_model import MainVerificationService
from app.services.models.analysis import (
    evidence_attribution,
    SpanHighlightingService,
    ContradictionHeatmapService,
    QuerySensitivityService,
    source_credibility,
    FakeSignsService
)

router = APIRouter(prefix="/analysis", tags=["analysis"])
_main = MainVerificationService()
_span_svc = SpanHighlightingService()
_heat_svc = ContradictionHeatmapService()
_sens_svc = QuerySensitivityService()
_signs_svc = FakeSignsService()


async def _get_result(req: AnalysisRequest) -> dict:
    if req.result:
        return req.result
    return await _main.verify(req.text)


@router.post("/attribution", response_model=list[AttributionItem],
             summary="Rank evidence by contribution to the verdict")
async def attribution(req: AnalysisRequest) -> list[AttributionItem]:

    result = await _get_result(req)
    rows = evidence_attribution(result, top_k=req.top_k_docs * 3)
    return [
        AttributionItem(
            domain=r["domain"],
            title=r["title"],
            score=r["score"],
            prob_true=r["prob_true"],
            contribution=r["contribution"],
            direction=r["direction"],
            url=r["url"],
        )
        for r in rows
    ]


@router.post("/spans", response_model=SpanAnalysisResponse,
             summary="Extract specific contradicting spans per sub-claim")
async def spans(req: AnalysisRequest) -> SpanAnalysisResponse:

    result = await _get_result(req)
    analysis = await _span_svc.analyse(req.text, result, top_k_docs=req.top_k_docs)
    return SpanAnalysisResponse(**analysis)


@router.post("/heatmap", response_model=HeatmapResponse,
             summary="NLI sentence-level contradiction heatmap")
async def heatmap(req: AnalysisRequest) -> HeatmapResponse:

    result = await _get_result(req)
    evs = sorted(
        [ev for ev in result.get("evidence", []) if "score" in ev],
        key=lambda e: e["score"]
    )
    if not evs:
        return HeatmapResponse(
            claim_sentences=[], evidence_sentences=[],
            cells=[], domain="", title=""
        )
    data = await asyncio.to_thread(_heat_svc.build, req.text, evs[0])
    cells = [HeatmapCell(**c) for c in data["cells"]]
    return HeatmapResponse(
        claim_sentences=data["claim_sentences"],
        evidence_sentences=data["evidence_sentences"],
        cells=cells,
        domain=data["domain"],
        title=data["title"],
    )


@router.post("/sensitivity", response_model=SensitivityResponse,
             summary="Measure verdict stability across different query sets")
async def sensitivity(req: AnalysisRequest) -> SensitivityResponse:
    data = await _sens_svc.run(
        req.text,
        n_trials=req.sensitivity_trials,
    )
    return SensitivityResponse(
        news=data["news"],
        trials=[SensitivityTrial(**t) for t in data["trials"]],
        mean_prob=data["mean_prob"],
        std_prob=data["std_prob"],
        min_prob=data["min_prob"],
        max_prob=data["max_prob"],
        verdict_stable=data["verdict_stable"],
    )

@router.post("/signs",
             summary="Detect typical signs of fake news in the text")
async def signs(req: AnalysisRequest):
    data = await _signs_svc.analyse(req.text)
    return data