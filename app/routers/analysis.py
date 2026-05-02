import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Body
from app.schemas.requests import AnalysisRequest
from app.schemas.responses import (
    AttributionItem,
    SpanAnalysisResponse,
    HeatmapResponse, HeatmapCell,
    SensitivityResponse, SensitivityTrial,
    CredibilityResponse,
    TemporalResponse,
    ErrorAnalysisResponse, ErrorRecord,
)
from app.services.models.main_model import MainVerificationService
from app.services.models.analysis import (
    evidence_attribution,
    SpanHighlightingService,
    ContradictionHeatmapService,
    QuerySensitivityService,
    source_credibility,
    temporal_analysis,
    error_analysis,
    inter_method_agreement,
)

router = APIRouter(prefix="/analysis", tags=["analysis"])
_main = MainVerificationService()
_span_svc = SpanHighlightingService()
_heat_svc = ContradictionHeatmapService()
_sens_svc = QuerySensitivityService()


def _get_result(req: AnalysisRequest) -> dict:

    if req.result:
        return req.result
    return _main.verify(req.text)


@router.post("/attribution", response_model=list[AttributionItem],
             summary="Rank evidence by contribution to the verdict")
def attribution(req: AnalysisRequest) -> list[AttributionItem]:

    result = _get_result(req)
    rows = evidence_attribution(result, top_k=req.top_k_docs * 3)
    return [
        AttributionItem(
            domain=r["domain"],
            title=r["title"],
            score=r["score"],
            **{"P(true)": r["prob_true"]},
            contribution=r["contribution"],
            direction=r["direction"],
            url=r["url"],
        )
        for r in rows
    ]


@router.post("/spans", response_model=SpanAnalysisResponse,
             summary="Extract specific contradicting spans per sub-claim")
def spans(req: AnalysisRequest) -> SpanAnalysisResponse:

    result = _get_result(req)
    analysis = _span_svc.analyse(req.text, result, top_k_docs=req.top_k_docs)
    return SpanAnalysisResponse(**analysis)


@router.post("/heatmap", response_model=HeatmapResponse,
             summary="NLI sentence-level contradiction heatmap")
def heatmap(req: AnalysisRequest) -> HeatmapResponse:

    result = _get_result(req)
    evs = sorted(
        [ev for ev in result.get("evidence", []) if "score" in ev],
        key=lambda e: e["score"]  # most contradicting first
    )
    if not evs:
        return HeatmapResponse(
            claim_sentences=[], evidence_sentences=[],
            cells=[], domain="", title=""
        )
    data = _heat_svc.build(req.text, evs[0])
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
def sensitivity(req: AnalysisRequest) -> SensitivityResponse:
    """
    Runs the retrieval pipeline `sensitivity_trials` times with freshly
    generated query sets and measures variance in P(true).
    std < 0.10 → verdict_stable = True.
    """
    data = _sens_svc.run(
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


@router.post("/credibility", response_model=CredibilityResponse,
             summary="Re-weight evidence by domain credibility")
def credibility(req: AnalysisRequest) -> CredibilityResponse:

    result = _get_result(req)
    weighted = source_credibility(result, req.credibility_overrides)

    from app.schemas.responses import EvidenceItem
    return CredibilityResponse(
        original_label=result.get("label", ""),
        original_probability=result.get("probability"),
        weighted_label=weighted.get("label_weighted", ""),
        weighted_probability=weighted.get("probability_weighted"),
        verdict_changed=weighted.get("verdict_changed", False),
        evidence=[
            EvidenceItem(**{k: v for k, v in ev.items()
                            if k in EvidenceItem.model_fields})
            for ev in weighted.get("evidence", [])
        ],
    )


@router.post("/temporal", response_model=TemporalResponse,
             summary="Recency-weighted analysis using evidence publication dates")
def temporal(req: AnalysisRequest) -> TemporalResponse:

    result = _get_result(req)
    claim_dt: datetime | None = None
    if req.claim_date:
        try:
            from dateutil import parser as dp
            claim_dt = dp.parse(req.claim_date)
            if claim_dt.tzinfo is None:
                claim_dt = claim_dt.replace(tzinfo=timezone.utc)
        except Exception:
            claim_dt = None

    enriched = temporal_analysis(result, claim_date=claim_dt)

    from app.schemas.responses import EvidenceItem
    return TemporalResponse(
        dated_count=enriched["dated_count"],
        undated_count=enriched["undated_count"],
        prob_temporal=enriched["prob_temporal"],
        label_temporal=enriched["label_temporal"],
        verdict_changed=enriched["verdict_changed"],
        evidence=[
            EvidenceItem(**{k: v for k, v in ev.items()
                            if k in EvidenceItem.model_fields})
            for ev in enriched.get("evidence", [])
        ],
    )


@router.post("/errors", response_model=ErrorAnalysisResponse,
             summary="Error taxonomy on a batch of labeled results")
def errors(
        items: list = Body(
            ...,
            example=[
                {"text": "...", "gold": 0, "pred": 1,
                 "probability": 0.62, "evidence": []}
            ],
            description=(
                    "List of objects with keys: text, gold (int), pred (int), "
                    "probability (float), evidence (list)"
            ),
        )
) -> ErrorAnalysisResponse:

    analysis = error_analysis(items)
    return ErrorAnalysisResponse(
        accuracy=analysis["accuracy"],
        f1_weighted=analysis["f1_weighted"],
        records=[ErrorRecord(**r) for r in analysis["records"]
                 if set(r.keys()) >= {"text", "gold", "pred", "probability",
                                      "n_evidence", "category", "correct", "top_domain"}],
        category_counts=analysis["category_counts"],
    )
