from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class VerifyRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Текст новости для верификации")
    num_queries: int = Field(5, ge=1, le=10)
    num_results: int = Field(5, ge=1, le=20)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class BaselineRequest(BaseModel):
    text: str = Field(..., min_length=10)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    few_shot_examples: Optional[List[Tuple[str, str]]] = Field(
        None,
        description="Список примеров для few-shot классификации. Каждый пример — это кортеж (текст новости, метка), "
                    "Класс должен быть 'ПРАВДИВАЯ' или 'ФЕЙКОВАЯ'."
    )
    # CoRAG / STEEL round limits
    max_rounds: int = Field(4, ge=1, le=8)
    num_queries: int = Field(5, ge=1, le=10)
    num_results: int = Field(5, ge=1, le=20)


class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=10)
    # Pass a pre-existing result to avoid re-running the main model
    # If omitted the main model is run first
    result: Optional[dict] = Field(
        None,
        description="Pre-computed verification result dict. "
                    "If omitted the main model is invoked first."
    )
    # Module-specific options
    top_k_docs: int = Field(3, ge=1, le=10)
    sensitivity_trials: int = Field(3, ge=2, le=10)
    claim_date: Optional[str] = Field(
        None,
        description="ISO-8601 date string of the news article (for temporal analysis)"
    )
    credibility_overrides: Optional[dict] = Field(
        None,
        description="Dict of domain → credibility score overrides, e.g. {'rbc.ru': 0.8}"
    )


class CompareRequest(BaseModel):
    text: str = Field(..., min_length=10)
    methods: List[str] = Field(
        default=["main", "single_rag", "llm_zeroshot", "nli"],
        description="Which methods to include. Options: main, rubert, llm_zeroshot, "
                    "llm_fewshot, single_rag, corag, steel, nli, gnn"
    )
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    num_queries: int = Field(5, ge=1, le=10)
    gold_label: Optional[int] = Field(
        None,
        description="Ground-truth label (0=fake, 1=real) for accuracy reporting"
    )


class GenerateQueriesRequest(BaseModel):
    text: str = Field(..., min_length=10)
    num_queries: int = Field(5, ge=1, le=10)
    method: str = Field("main", description=(
        "Which method's query generator to use. "
        "Options: main, single_rag, corag, steel, nli, gnn"
    ))


class RunWithQueriesRequest(BaseModel):
    text: str = Field(..., min_length=10)
    queries: List[str] = Field(..., min_items=1, description="Изменённый пользователем список поисковых запросов")
    num_results: int = Field(5, ge=1, le=20)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_rounds: int = Field(4, ge=1, le=8)
    method: str = Field("main", description=(
        "Какой из методов запустить. "
        "Варианты: main, single_rag, corag, steel, nli, gnn"
    ))
