import json
import logging
import re
import asyncio
import numpy as np
from app.core.registry import registry
from app.services.search_llm import GigaChatService, YandexSearchService
from app.services.models.utils import _sigmoid

logger = logging.getLogger(__name__)

_FALLBACK_SYSTEM = (
    "Ты — эксперт по верификации новостей. "
    "Тебе дана новость и список источников, найденных в интернете. "
    "Определи, подтверждают ли источники новость по существу. "
    "Не оценивай стиль или грамматику — только фактическое содержание. "
    "Если ни один источник не подтверждает ключевое утверждение новости, "
    "значит новость не подтверждена.\n\n"
    "Ответь строго JSON без markdown-разметки:\n"
    '{"supported": true | false, "confidence": 0.0-1.0, '
    '"reasoning": "краткое обоснование"}'
)


class MainVerificationService:

    def __init__(self):
        self.llm = GigaChatService()
        self.search = YandexSearchService()

    async def generate_queries(self, news_text: str,
                               num_queries: int = 5) -> list[str]:
        queries = await self.llm.generate_queries(news_text, num_queries)
        logger.info("Generated %d queries", len(queries))
        return queries

    async def _llm_fallback(self, news_text: str,
                            evidences: list[dict]) -> dict | None:
        """Спрашиваем GigaChat, подтверждают ли источники новость."""
        snippets = "\n".join(
            f"- [{ev.get('domain', '')}] {ev.get('title', '')}: "
            f"{ev.get('content', '')[:200]}"
            for ev in evidences[:7]
        )
        raw = await self.llm.complete(
            _FALLBACK_SYSTEM,
            f"Новость: {news_text}\n\nИсточники:\n{snippets}",
        )
        try:
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            logger.warning("LLM fallback returned unparseable response")
            return None

    async def run_with_queries(
            self,
            news_text: str,
            queries: list[str],
            num_results: int = 5,
            threshold: float = 0.5,
    ) -> dict:
        evidences = await self.search.multi_search(
            queries, n_per_query=num_results
        )
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
        pairs = [
            [news_text, f"{ev['title']}. {ev['content'][:1000]}"]
            for ev in evidences
        ]
        scores = await asyncio.to_thread(ce.predict, pairs)

        for ev, sc in zip(evidences, scores):
            ev["score"] = float(sc)

        mean_score = float(np.mean(scores))
        probability = _sigmoid(mean_score)
        label = "ПРАВДИВАЯ" if probability >= threshold else "ЛОЖНАЯ"
        reasoning = f"mean_ce_score={mean_score:.3f}"

        if 0.35 < probability < 0.65:
            logger.info(
                "Low confidence (P=%.3f), running LLM fallback",
                probability,
            )
            parsed = await self._llm_fallback(news_text, evidences)

            if parsed is not None:
                supported = parsed.get("supported", True)
                llm_conf = float(parsed.get("confidence", 0.5))
                llm_reason = parsed.get("reasoning", "")

                if not supported:
                    probability = min(probability, 1.0 - llm_conf)
                    label = "ЛОЖНАЯ"
                    reasoning += (
                        f" → LLM fallback: NOT supported "
                        f"(conf={llm_conf:.2f}): {llm_reason}"
                    )
                else:
                    # LLM подтверждает — немного поднимаем уверенность
                    probability = max(probability, llm_conf)
                    label = "ПРАВДИВАЯ" if probability >= threshold else "ЛОЖНАЯ"
                    reasoning += (
                        f" → LLM fallback: supported "
                        f"(conf={llm_conf:.2f}): {llm_reason}"
                    )

                logger.info("After LLM fallback: P=%.3f, label=%s",
                            probability, label)

        return {
            "label": label,
            "probability": probability,
            "queries": queries,
            "evidence": evidences,
            "reasoning": reasoning,
        }

    async def verify(
            self,
            news_text: str,
            num_queries: int = 5,
            num_results: int = 5,
            threshold: float = 0.5,
    ) -> dict:
        queries = await self.generate_queries(news_text, num_queries)
        return await self.run_with_queries(
            news_text, queries, num_results, threshold
        )
