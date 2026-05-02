import json
import logging
import re
import time
import numpy as np
import torch
from app.core.config import get_settings
from app.core.registry import registry, DEVICE
from app.services.search_llm import GigaChatService, YandexSearchService

logger = logging.getLogger(__name__)
settings = get_settings()


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _label(prob: float, threshold: float = 0.5) -> str:
    return "ПРАВДИВАЯ" if prob >= threshold else "ФЕЙКОВАЯ"


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
            logger.warning("RuBERT not loaded (%s). Call /baselines/rubert/train first.", e)
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
Определи, является ли новость правдивой или фейковой.
Отвечай строго в формате JSON:
{"label": "ПРАВДИВАЯ" | "ФЕЙКОВАЯ", "confidence": 0.0-1.0, "reasoning": "..."}
Не добавляй ничего кроме JSON."""


class LLMClassifierService:

    def __init__(self):
        self.llm = GigaChatService()

    def verify(
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
            raw = self.llm.complete_messages(messages)
        else:
            raw = self.llm.complete(_CLASSIFY_SYSTEM, f"Новость: {news_text}")

        parsed = _parse_json(raw)
        if parsed:
            label = parsed.get("label", "ФЕЙКОВАЯ")
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", "")
            prob = confidence if label == "ПРАВДИВАЯ" else 1.0 - confidence
        else:
            label = "ПРАВДИВАЯ" if "ПРАВДИВАЯ" in raw.upper() else "ФЕЙКОВАЯ"
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
{{"action": "verdict", "label": "ПРАВДИВАЯ" | "ФЕЙКОВАЯ", "confidence": 0.0-1.0, "reasoning": "..."}}

Делай не более {max_rounds} итераций."""

_CORAG_EVIDENCE_TPL = (
    '\n--- Результаты по запросу "{query}" ---\n{snippets}\n---\n'
    'Достаточно ли доказательств? Если нет — следующий запрос.'
)


class ChainOfRAGService:

    def __init__(self):
        self.llm = GigaChatService()
        self.search = YandexSearchService()

    def verify(self, news_text: str,
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
            raw = self.llm.complete_messages(conversation)
            parsed = _parse_json(raw)
            conversation.append({"role": "assistant", "content": raw})

            if parsed is None:
                break

            if parsed.get("action") == "verdict":
                label = parsed.get("label", "ФЕЙКОВАЯ")
                confidence = float(parsed.get("confidence", 0.5))
                reasoning = parsed.get("reasoning", "")
                prob = confidence if label == "ПРАВДИВАЯ" else 1.0 - confidence
                return {
                    "label": label, "probability": prob,
                    "queries": [m["content"] for m in conversation
                                if m["role"] == "user" and "запрос" not in m["content"]],
                    "evidence": all_evidences,
                    "reasoning": f"[CoRAG {round_num + 1} rounds] {reasoning}",
                }

            query = parsed.get("query", news_text[:100])
            evidences = self.search.search(query, n=num_results, seen_urls=seen_urls)
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

        # Force final verdict
        conversation.append({
            "role": "user",
            "content": (
                "Лимит поиска исчерпан. Вынеси финальный вердикт: "
                '{"action":"verdict","label":"...","confidence":0.0-1.0,"reasoning":"..."}'
            ),
        })
        raw = self.llm.complete_messages(conversation)
        parsed = _parse_json(raw)
        if parsed and parsed.get("action") == "verdict":
            label = parsed.get("label", "ФЕЙКОВАЯ")
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


_STEEL_FILTER_SYS = (
    "Оцени релевантность документа для верификации новости. "
    'Ответь строго JSON: {"relevant": true | false, "reason": "..."}'
)
_STEEL_REQUERY_SYS = (
    "Текущих доказательств недостаточно. Сформулируй новый поисковый запрос. "
    'Ответь строго JSON: {"query": "..."}'
)
_STEEL_VERdict_SYS = (
    "На основании собранных доказательств вынеси вердикт. "
    'Ответь строго JSON: {"label":"ПРАВДИВАЯ"|"ФЕЙКОВАЯ","confidence":0.0-1.0,"reasoning":"..."}'
)


class STEELService:
    def __init__(self):
        self.llm = GigaChatService()
        self.search = YandexSearchService()

    def verify(self, news_text: str,
               max_rounds: int = 5,
               num_results: int = 5,
               min_relevant: int = 3,
               threshold: float = 0.5) -> dict:

        relevant: list[dict] = []
        all_evidences: list[dict] = []
        seen_urls: set = set()
        prev_queries: list[str] = []
        current_query = news_text[:150].strip()

        for round_num in range(max_rounds):
            logger.info("[STEEL] Round %d: %s", round_num + 1, current_query[:60])
            docs = self.search.search(current_query, n=num_results,
                                      seen_urls=seen_urls)
            all_evidences.extend(docs)
            prev_queries.append(current_query)

            for doc in docs:
                snippet = f"{doc['title']}. {doc['content'][:200]}"
                raw = self.llm.complete(
                    _STEEL_FILTER_SYS,
                    f"Новость: {news_text}\n\nДокумент: {snippet}"
                )
                parsed = _parse_json(raw)
                doc["relevant"] = bool(parsed.get("relevant", False)) if parsed else False
                if doc["relevant"]:
                    relevant.append(doc)

            if len(relevant) >= min_relevant:
                break

            if round_num < max_rounds - 1:
                raw = self.llm.complete(
                    _STEEL_REQUERY_SYS,
                    f"Новость: {news_text}\n"
                    f"Предыдущие запросы: {json.dumps(prev_queries, ensure_ascii=False)}\n"
                    f"Релевантных документов: {len(relevant)} (нужно {min_relevant})"
                )
                parsed = _parse_json(raw)
                current_query = (parsed.get("query", news_text[:150])
                                 if parsed else news_text[:150])

        if not relevant:
            return {
                "label": "НЕДОСТАТОЧНО ДАННЫХ", "probability": None,
                "queries": prev_queries, "evidence": all_evidences,
                "reasoning": "STEEL: no relevant documents found.",
            }

        summary = "\n".join(
            f"{i + 1}. [{ev['domain']}] {ev['title']} — {ev['content'][:120]}"
            for i, ev in enumerate(relevant[:10])
        )
        raw = self.llm.complete(
            _STEEL_VERdict_SYS,
            f"Новость: {news_text}\n\nДоказательства:\n{summary}"
        )
        parsed = _parse_json(raw)
        if parsed:
            label = parsed.get("label", "ФЕЙКОВАЯ")
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", "")
            prob = confidence if label == "ПРАВДИВАЯ" else 1.0 - confidence
        else:
            label, prob, reasoning = "ФЕЙКОВАЯ", 0.3, raw[:200]

        return {
            "label": label, "probability": prob,
            "queries": prev_queries, "evidence": relevant,
            "reasoning": f"[STEEL {round_num + 1} rounds, {len(relevant)} relevant] {reasoning}",
        }


_NLI_LABELS = ["entailment", "neutral", "contradiction"]


class NLIClassifierService:

    def __init__(self):
        self.llm = GigaChatService()
        self.search = YandexSearchService()

    def verify(self, news_text: str,
               num_queries: int = 5,
               num_results: int = 5,
               threshold: float = 0.5) -> dict:

        queries = self.llm.generate_queries(news_text, num_queries)
        evidences = self.search.multi_search(queries, n_per_query=num_results)

        if not evidences:
            return {
                "label": "НЕДОСТАТОЧНО ДАННЫХ", "probability": None,
                "queries": queries, "evidence": [],
                "reasoning": "Search returned no results.",
            }

        nli = registry.nli
        e_sum = n_sum = c_sum = 0.0

        for ev in evidences:
            premise = f"{ev['title']}. {ev['content'][:300]}"
            out = nli(
                sequences=premise[:512],
                candidate_labels=[news_text[:512]],
                hypothesis_template="{}",
            )
            scores = dict(zip(out["labels"], out["scores"]))
            ev["nli_entailment"] = float(scores.get("entailment", 0.0))
            ev["nli_neutral"] = float(scores.get("neutral", 0.0))
            ev["nli_contradiction"] = float(scores.get("contradiction", 0.0))
            e_sum += ev["nli_entailment"]
            n_sum += ev["nli_neutral"]
            c_sum += ev["nli_contradiction"]

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


class GNNService:

    def __init__(self):
        self.llm = GigaChatService()
        self.search = YandexSearchService()
        self._model = None
        self._ready = False
        self._embed_dim: int = 0

        try:
            import torch_geometric  # noqa: F401
            self._tg_available = True
        except ImportError:
            self._tg_available = False
            logger.warning(
                "torch-geometric not installed. GNN service unavailable. "
                "Install with: pip install torch-geometric"
            )

    def _build_graph(self, news_text: str, evidences: list[dict]):
        from torch_geometric.data import Data

        sbert = registry.sbert
        texts = [news_text] + [
            f"{ev['title']}. {ev['content'][:200]}" for ev in evidences
        ]
        embs = sbert.encode(
            texts, convert_to_tensor=True, device=DEVICE, show_progress_bar=False
        )
        norms = embs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sim = (embs / norms @ (embs / norms).T).cpu()

        src, dst = [], []
        n = len(texts)
        for i in range(n):
            for j in range(n):
                if i != j and float(sim[i, j]) >= 0.5:
                    src.append(i);
                    dst.append(j)
        for j in range(1, n):
            src += [0, j];
            dst += [j, 0]

        return Data(
            x=embs,
            edge_index=torch.tensor([src, dst], dtype=torch.long)
        )

    def train(self, csv_path: str,
              text_col: str = "text", label_col: str = "label",
              epochs: int = 20, num_queries: int = 5, num_results: int = 4):
        if not self._tg_available:
            raise RuntimeError("torch-geometric not installed.")

        from torch_geometric.nn import GATConv
        import pandas as pd

        sbert = registry.sbert
        self._embed_dim = sbert.get_sentence_embedding_dimension()

        class _GAT(torch.nn.Module):
            def __init__(self, in_dim, hidden=128, heads=4):
                super().__init__()
                self.g1 = GATConv(in_dim, hidden, heads=heads, dropout=0.3)
                self.g2 = GATConv(hidden * heads, hidden, heads=1, dropout=0.3)
                self.clf = torch.nn.Linear(hidden, 2)
                self.drop = torch.nn.Dropout(0.3)

            def forward(self, x, ei):
                x = torch.relu(self.g1(x, ei))
                x = self.drop(x)
                x = torch.relu(self.g2(x, ei))
                return self.clf(x[0].unsqueeze(0))

        model = _GAT(self._embed_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        df = pd.read_csv(csv_path)[[text_col, label_col]].dropna()
        df.columns = ["text", "label"]

        model.train()
        for epoch in range(epochs):
            total_loss = correct = total = 0
            for _, row in df.iterrows():
                try:
                    queries = self.llm.generate_queries(row["text"], num_queries)
                    evidences = self.search.multi_search(queries, n_per_query=num_results)
                    graph = self._build_graph(row["text"], evidences).to(DEVICE)
                    label_t = torch.tensor([int(row["label"])],
                                           dtype=torch.long).to(DEVICE)
                    optimizer.zero_grad()
                    logits = model(graph.x, graph.edge_index)
                    loss = criterion(logits, label_t)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    correct += int(logits.argmax() == int(row["label"]))
                    total += 1
                    time.sleep(0.3)
                except Exception as e:
                    logger.warning("[GNN train] skipping example: %s", e)
            logger.info(
                "GNN Epoch %d/%d | Loss=%.4f | Acc=%.3f",
                epoch + 1, epochs, total_loss / max(total, 1),
                correct / max(total, 1)
            )

        self._model = model
        self._ready = True
        torch.save(model.state_dict(), settings.rubert_save_path + "_gnn.pt")
        logger.info("GNN saved.")

    def verify(self, news_text: str,
               num_queries: int = 5, num_results: int = 4,
               threshold: float = 0.5) -> dict:
        if not self._tg_available:
            return {
                "label": "ОШИБКА", "probability": None, "queries": [], "evidence": [],
                "reasoning": "torch-geometric not installed.",
            }
        if not self._ready:
            return {
                "label": "ОШИБКА", "probability": None, "queries": [], "evidence": [],
                "reasoning": "GNN not trained. POST /baselines/gnn/train first.",
            }

        queries = self.llm.generate_queries(news_text, num_queries)
        evidences = self.search.multi_search(queries, n_per_query=num_results)

        if not evidences:
            return {
                "label": "НЕДОСТАТОЧНО ДАННЫХ", "probability": None,
                "queries": queries, "evidence": [],
                "reasoning": "Search returned no results.",
            }

        graph = self._build_graph(news_text, evidences).to(DEVICE)
        self._model.eval()
        with torch.no_grad():
            probs = torch.softmax(
                self._model(graph.x, graph.edge_index), dim=1
            ).cpu().numpy()[0]

        prob = float(probs[1])
        return {
            "label": _label(prob, threshold),
            "probability": prob,
            "queries": queries,
            "evidence": evidences,
            "reasoning": (
                f"GAT | nodes={len(evidences) + 1} "
                f"| edges={graph.edge_index.shape[1]} "
                f"| P(fake)={probs[0]:.3f} P(true)={probs[1]:.3f}"
            ),
        }
