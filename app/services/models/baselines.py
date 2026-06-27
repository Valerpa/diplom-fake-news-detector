import asyncio
import json
import logging
import re
import time
import numpy as np
import torch
from app.core.config import get_settings
from app.core.registry import registry, DEVICE
from app.services.search_llm import GigaChatService, YandexSearchService
from app.services.models.utils import _label

logger = logging.getLogger(__name__)
settings = get_settings()


def _parse_json(raw: str) -> dict | None:
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        return None


class RuBERTService:

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        try:
            self._tok = AutoTokenizer.from_pretrained(settings.rubert_save_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                settings.rubert_save_path
            ).to(DEVICE)
            self._model.eval()
            self._ready = True
            logger.info("RuBERT loaded from %s", settings.rubert_save_path)
        except Exception as e:
            logger.warning(
                f"RuBERT not loaded ({e}). Call /baselines/rubert/train first."
            )
            self._ready = False

    def train(self, csv_path: str,
              text_col: str = "text",
              label_col: str = "label",
              epochs: int = 3,
              batch_size: int = 8):
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer
        )
        from datasets import Dataset
        from sklearn.model_selection import train_test_split
        import pandas as pd

        df = pd.read_csv(csv_path)[[text_col, label_col]].dropna()
        df.columns = ["text", "label"]
        df["label"] = df["label"].astype(int)

        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df["label"], random_state=42
        )
        tok = AutoTokenizer.from_pretrained(settings.rubert_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            settings.rubert_model, num_labels=2
        )

        def tokenize(batch):
            return tok(batch["text"], truncation=True,
                       max_length=512, padding="max_length")

        def to_ds(d):
            ds = Dataset.from_pandas(d.reset_index(drop=True))
            ds = ds.map(tokenize, batched=True)
            ds = ds.rename_column("label", "labels")
            ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
            return ds

        args = TrainingArguments(
            output_dir=settings.rubert_save_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=(DEVICE == "cuda"),
        )
        trainer = Trainer(model=model, args=args,
                          train_dataset=to_ds(train_df),
                          eval_dataset=to_ds(val_df))
        trainer.train()
        trainer.save_model(settings.rubert_save_path)
        tok.save_pretrained(settings.rubert_save_path)
        self._tok = tok
        self._model = model.to(DEVICE)
        self._model.eval()
        self._ready = True
        logger.info("RuBERT fine-tuned and saved.")

    def verify(self, news_text: str, threshold: float = 0.5) -> dict:
        if not self._ready:
            return {
                "label": "ОШИБКА", "probability": None, "queries": [],
                "evidence": [],
                "reasoning": "RuBERT model not loaded. POST /baselines/rubert/train first.",
            }
        inputs = self._tok(
            news_text, return_tensors="pt",
            truncation=True, max_length=512, padding=True
        ).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(
                self._model(**inputs).logits, dim=1
            ).cpu().numpy()[0]
        prob = float(probs[1])
        return {
            "label": _label(prob, threshold),
            "probability": prob,
            "queries": [],
            "evidence": [],
            "reasoning": f"RuBERT: P(fake)={probs[0]:.3f} P(true)={probs[1]:.3f}",
        }


_CLASSIFY_SYSTEM = """Ты — эксперт по верификации новостей на русском языке.
Определи, является ли новость правдивой или ложной.
Отвечай строго в формате JSON:
{"label": "ПРАВДИВАЯ" | "ЛОЖНАЯ", "confidence": 0.0-1.0, "reasoning": "..."}
Не добавляй ничего кроме JSON."""


class LLMClassifierService:

    def __init__(self):
        self.llm = GigaChatService()

    async def verify(
            self,
            news_text: str,
            few_shot_examples: list[tuple[str, str]] | None = None,
            threshold: float = 0.5,
    ) -> dict:
        if few_shot_examples:
            messages = [{"role": "system", "content": _CLASSIFY_SYSTEM}]
            for ex_text, ex_label in few_shot_examples:
                messages.append({"role": "user",
                                 "content": f"Новость: {ex_text}"})
                messages.append({"role": "assistant",
                                 "content": json.dumps(
                                     {"label": ex_label, "confidence": 0.9,
                                      "reasoning": "Пример из датасета."},
                                     ensure_ascii=False
                                 )})
            messages.append({"role": "user",
                             "content": f"Новость: {news_text}"})
            raw = await self.llm.complete_messages(messages)
        else:
            raw = await self.llm.complete(
                _CLASSIFY_SYSTEM, f"Новость: {news_text}"
            )

        parsed = _parse_json(raw)
        if parsed:
            label = parsed.get("label", "")
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", "")
            prob = confidence if label == "ПРАВДИВАЯ" else 1.0 - confidence
        else:
            label = "ПРАВДИВАЯ" if "ПРАВДИВАЯ" in raw.upper() else "ЛОЖНАЯ"
            prob = 0.7 if label == "ПРАВДИВАЯ" else 0.3
            reasoning = raw[:300]

        return {
            "label": label, "probability": prob,
            "queries": [], "evidence": [], "reasoning": reasoning,
        }


_CORAG_SYSTEM = """Ты — эксперт по верификации новостей. Ты проверяешь новости итеративно.
На каждом шаге отвечай строго JSON одного из двух форматов:

Если нужен поиск:
{{"action": "search", "query": "поисковый запрос"}}

Если достаточно доказательств:
{{"action": "verdict", "label": "ПРАВДИВАЯ" | "ЛОЖНАЯ", "confidence": 0.0-1.0, "reasoning": "..."}}

Делай не более {max_rounds} итераций."""

_CORAG_EVIDENCE_TPL = (
    '\n--- Результаты по запросу "{query}" ---\n{snippets}\n---\n'
    'Достаточно ли доказательств? Если нет — следующий запрос.'
)


class ChainOfRAGService:

    def __init__(self):
        self.llm = GigaChatService()
        self.search = YandexSearchService()

    async def verify(self, news_text: str,
                     max_rounds: int = 4,
                     num_results: int = 5,
                     threshold: float = 0.5) -> dict:

        all_evidences: list[dict] = []
        seen_urls: set = set()
        conversation = [
            {"role": "system",
             "content": _CORAG_SYSTEM.format(max_rounds=max_rounds)},
            {"role": "user",
             "content": f"Проверь новость:\n\n{news_text}"},
        ]

        for round_num in range(max_rounds):
            raw = await self.llm.complete_messages(conversation)
            parsed = _parse_json(raw)
            conversation.append({"role": "assistant", "content": raw})

            if parsed is None:
                break

            if parsed.get("action") == "verdict":
                label = parsed.get("label", "ЛОЖНАЯ")
                confidence = float(parsed.get("confidence", 0.5))
                reasoning = parsed.get("reasoning", "")
                prob = confidence if label == "ПРАВДИВАЯ" else 1.0 - confidence
                return {
                    "label": label, "probability": prob,
                    "queries": [m["content"] for m in conversation
                                if m["role"] == "user"
                                and "запрос" not in m["content"]],
                    "evidence": all_evidences,
                    "reasoning": f"[CoRAG {round_num + 1} rounds] {reasoning}",
                }

            query = parsed.get("query", news_text[:100])
            evidences = await self.search.search(
                query, n=num_results, seen_urls=seen_urls
            )
            all_evidences.extend(evidences)

            snippets = "\n".join(
                f"{i + 1}. [{ev['domain']}] {ev['title']} — {ev['content'][:120]}"
                for i, ev in enumerate(evidences)
            ) or "Результатов не найдено."
            conversation.append({
                "role": "user",
                "content": _CORAG_EVIDENCE_TPL.format(
                    query=query, snippets=snippets
                ),
            })

        conversation.append({
            "role": "user",
            "content": (
                "Лимит поиска исчерпан. Вынеси финальный вердикт: "
                '{"action":"verdict","label":"...","confidence":0.0-1.0,"reasoning":"..."}'
            ),
        })
        raw = await self.llm.complete_messages(conversation)
        parsed = _parse_json(raw)
        if parsed and parsed.get("action") == "verdict":
            label = parsed.get("label", "ЛОЖНАЯ")
            confidence = float(parsed.get("confidence", 0.5))
            prob = confidence if label == "ПРАВДИВАЯ" else 1.0 - confidence
            return {
                "label": label, "probability": prob,
                "queries": [], "evidence": all_evidences,
                "reasoning": f"[CoRAG forced verdict] {parsed.get('reasoning', '')}",
            }
        return {
            "label": "НЕДОСТАТОЧНО ДАННЫХ", "probability": 0.5,
            "queries": [], "evidence": all_evidences,
            "reasoning": "CoRAG could not reach a verdict.",
        }


_NLI_LABELS = ["entailment", "neutral", "contradiction"]


class NLIClassifierService:

    def __init__(self):
        self.llm = GigaChatService()
        self.search = YandexSearchService()

    def _run_nli(self, news_text: str, evidences: list[dict]) -> list[dict]:
        for ev in evidences:
            premise = f"{ev['title']}. {ev['content'][:800]}"
            scores = registry.nli_scores(
                premise=premise[:512],
                hypothesis=news_text[:512],
            )
            ev["nli_entailment"] = scores["entailment"]
            ev["nli_neutral"] = scores["neutral"]
            ev["nli_contradiction"] = scores["contradiction"]
        return evidences

    async def verify(self, news_text: str,
                     num_queries: int = 5,
                     num_results: int = 5,
                     threshold: float = 0.5) -> dict:

        queries = await self.llm.generate_queries(news_text, num_queries)
        evidences = await self.search.multi_search(
            queries, n_per_query=num_results
        )

        if not evidences:
            return {
                "label": "НЕДОСТАТОЧНО ДАННЫХ", "probability": None,
                "queries": queries, "evidence": [],
                "reasoning": "Search returned no results.",
            }

        evidences = await asyncio.to_thread(
            self._run_nli, news_text, evidences
        )

        e_sum = sum(ev["nli_entailment"] for ev in evidences)
        n_sum = sum(ev["nli_neutral"] for ev in evidences)
        c_sum = sum(ev["nli_contradiction"] for ev in evidences)

        denom = e_sum + c_sum
        prob = (e_sum / denom) if denom > 1e-9 else 0.5
        total = e_sum + n_sum + c_sum
        nw = (n_sum / total) if total > 0 else 0
        prob = prob * (1 - nw) + 0.5 * nw

        return {
            "label": _label(prob, threshold),
            "probability": float(prob),
            "queries": queries,
            "evidence": evidences,
            "reasoning": (
                f"NLI: entail={e_sum:.2f} neutral={n_sum:.2f} "
                f"contra={c_sum:.2f} P(true)={prob:.3f}"
            ),
        }


