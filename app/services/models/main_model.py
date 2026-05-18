import logging
import asyncio
import numpy as np
from app.core.registry import registry
from app.services.search_llm import GigaChatService, YandexSearchService
from app.services.models.utils import _sigmoid

logger = logging.getLogger(__name__)


class MainVerificationService:

    def __init__(self):
        self.llm = GigaChatService()
        self.search = YandexSearchService()

    async def generate_queries(self, news_text: str, num_queries: int = 5) -> list[str]:
        queries = await self.llm.generate_queries(news_text, num_queries)
        logger.info("Generated %d queries", len(queries))
        return queries

    async def run_with_queries(
            self,
            news_text: str,
            queries: list[str],
            num_results: int = 5,
            threshold: float = 0.5,
    ) -> dict:
        evidences = await self.search.multi_search(queries, n_per_query=num_results)
        logger.info("Retrieved %d evidence documents", len(evidences))

        if not evidences:
            return {
                "label": "НЕДОСТАТОЧНО ДАННЫХ",
                "probability": None,
                "queries": queries,
                "evidence": [],
                "reasoning": "Search returned no results.",
            }

        ce = registry.cross_encoder
        pairs = [[news_text, f"{ev['title']}. {ev['content'][:1000]}"] for ev in evidences]
        scores = await asyncio.to_thread(ce.predict,pairs)

        for ev, sc in zip(evidences, scores):
            ev["score"] = float(sc)

        mean_score = float(np.mean(scores))
        probability = _sigmoid(mean_score)
        label = "ПРАВДИВАЯ" if probability >= threshold else "ЛОЖНАЯ"

        return {
            "label": label,
            "probability": probability,
            "queries": queries,
            "evidence": evidences,
            "reasoning": f"mean_ce_score={mean_score:.3f}",
        }

    async def verify(
            self,
            news_text: str,
            num_queries: int = 5,
            num_results: int = 5,
            threshold: float = 0.5,
    ) -> dict:
        queries = await self.generate_queries(news_text, num_queries)
        return await self.run_with_queries(news_text, queries, num_results, threshold)
