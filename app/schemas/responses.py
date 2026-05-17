from typing import Any
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    title: str
    content: str
    domain: str
    url: str
    score: float | None = None
    credibility: float | None = None
    score_weighted: float | None = None
    recency_weight: float | None = None
    days_from_claim: int| None = None
    nli_entailment: float | None = None
    nli_contradiction: float | None = None
    nli_neutral: float | None = None
    relevant: bool | None = None


class VerifyResponse(BaseModel):
    text: str
    label: str = Field(..., description="'ПРАВДИВАЯ' | 'ЛОЖНАЯ' | 'НЕДОСТАТОЧНО ДАННЫХ'")
    probability: float | None
    queries: list[str]
    evidence: list[EvidenceItem]
    reasoning: str = ""


class BaselineResponse(BaseModel):
    method: str
    text: str
    label: str
    probability: float | None
    evidence: list[EvidenceItem] = []
    reasoning: str = ""


class AttributionItem(BaseModel):
    domain: str
    title: str
    score: float
    prob_true: float = Field(..., alias="P(правда)")
    contribution: float
    direction: str
    url: str

    model_config = {"populate_by_name": True}


class SubClaimSpan(BaseModel):
    sub_claim: str
    evidence_span: str
    confidence: float


class EvidenceSpanAnalysis(BaseModel):
    domain: str
    title: str
    score: float
    direction: str
    sub_claims: list[SubClaimSpan]


class SpanAnalysisResponse(BaseModel):
    news: str
    sub_claims: list[str]
    evidence: list[EvidenceSpanAnalysis]


class HeatmapCell(BaseModel):
    claim_sentence: str
    evidence_sentence: str
    entailment: float
    neutral: float
    contradiction: float


class HeatmapResponse(BaseModel):
    claim_sentences: list[str]
    evidence_sentences: list[str]
    cells: list[HeatmapCell]  # flattened matrix
    domain: str
    title: str


class SensitivityTrial(BaseModel):
    trial: int
    probability: float
    label: str
    n_evidences: int
    queries: list[dict[str, Any]]


class SensitivityResponse(BaseModel):
    news: str
    trials: list[SensitivityTrial]
    mean_prob: float
    std_prob: float
    min_prob: float
    max_prob: float
    verdict_stable: bool


class CredibilityResponse(BaseModel):
    original_label: str
    original_probability: float
    weighted_label: str
    weighted_probability: float | None
    verdict_changed: bool
    evidence: list[EvidenceItem]


class TemporalResponse(BaseModel):
    dated_count: int
    undated_count: int
    prob_temporal: float
    label_temporal: str
    verdict_changed: bool
    evidence: list[EvidenceItem]


class ErrorRecord(BaseModel):
    text: str
    gold: int
    pred: int
    probability: float | None
    n_evidence: int
    category: str
    correct: int
    top_domain: str


class ErrorAnalysisResponse(BaseModel):
    accuracy: float
    f1_weighted: float
    records: list[ErrorRecord]
    category_counts: dict[str, int]


class MethodResult(BaseModel):
    method: str
    label: str
    probability: float | None
    reasoning: str = ""


class CompareResponse(BaseModel):
    text: str
    gold_label: int | None
    results: list[MethodResult]
    agreement_fraction: float = Field(
        ...,
        description="Доля методов, согласных с большинством"
    )
    consensus_label: str
    disagreement: bool


class HealthResponse(BaseModel):
    status: str
    device: str
    loaded_models: list[str]
