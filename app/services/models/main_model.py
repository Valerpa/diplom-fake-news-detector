import logging
from typing import Dict, List
import numpy as np
from app.core.registry import registry
from app.services.search_llm import GigaChatService, YandexSearchService

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class MainVerificationService:

    def __init__(self):
        self.llm    = GigaChatService()
        self.search = YandexSearchService()

    def generate_queries(self, news_text: str, num_queries: int = 5) -> List[str]:
        queries = self.llm.generate_queries(news_text, num_queries)
        logger.info("Generated %d queries", len(queries))
        return queries

    def run_with_queries(
        self,
        news_text:   str,
        queries:     List[str],
        num_results: int   = 5,
        threshold:   float = 0.5,
    ) -> Dict:
        evidences = self.search.multi_search(queries, n_per_query=num_results)
        logger.info("Retrieved %d evidence documents", len(evidences))

        if not evidences:
            return {
                "label":     "НЕДОСТАТОЧНО ДАННЫХ",
                "probability": None,
                "queries":   queries,
                "evidence":  [],
                "reasoning": "Search returned no results.",
            }

        ce     = registry.cross_encoder
        pairs  = [[news_text, f"{ev['title']}. {ev['content']}"] for ev in evidences]
        scores = ce.predict(pairs)

        for ev, sc in zip(evidences, scores):
            ev["score"] = float(sc)

        mean_score  = float(np.mean(scores))
        probability = _sigmoid(mean_score)
        label       = "ПРАВДИВАЯ" if probability >= threshold else "ФЕЙКОВАЯ"

        return {
            "label":       label,
            "probability": probability,
            "queries":     queries,
            "evidence":    evidences,
            "reasoning":   f"mean_ce_score={mean_score:.3f}",
        }

    def verify(
        self,
        news_text:   str,
        num_queries: int   = 5,
        num_results: int   = 5,
        threshold:   float = 0.5,
    ) -> Dict:
        queries = self.generate_queries(news_text, num_queries)
        return self.run_with_queries(news_text, queries, num_results, threshold)
