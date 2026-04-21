import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
from dateutil import parser as dateparser
from app.core.config import get_settings
from app.core.registry import registry
from app.services.search_llm import GigaChatService, YandexSearchService

logger = logging.getLogger(__name__)
settings = get_settings()

_NLI_LABELS = ["entailment", "neutral", "contradiction"]

DEFAULT_CREDIBILITY: dict[str, float] = {
    "ria.ru": 0.60, "tass.ru": 0.60,
    "interfax.ru": 0.70, "rbc.ru": 0.75,
    "kommersant.ru": 0.80, "vedomosti.ru": 0.80,
    "novayagazeta.ru": 0.75, "meduza.io": 0.75,
    "bbc.com": 0.90, "reuters.com": 0.92,
    "ap.org": 0.92, "factcheck.org": 0.95,
    "kremlin.ru": 0.50, "government.ru": 0.55,
    "mos.ru": 0.65, "life.ru": 0.30,
    "pravda.ru": 0.25, "riafan.ru": 0.20,
}
CREDIBILITY_FALLBACK = 0.50


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _label(prob: float, thr: float = 0.5) -> str:
    return "ПРАВДИВАЯ" if prob >= thr else "ФЕЙКОВАЯ"


def _get_credibility(domain: str,
                     overrides: dict[str, float] | None = None) -> float:
    src = {**DEFAULT_CREDIBILITY, **(overrides or {})}
    for key, val in src.items():
        if key in domain:
            return val
    return CREDIBILITY_FALLBACK


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip())
            if len(s.strip()) > 10]


def evidence_attribution(result: dict, top_k: int = 10) -> list[dict]:

    evs = [ev for ev in result.get("evidence", []) if "score" in ev]
    if not evs:
        return []

    scores = np.array([ev["score"] for ev in evs])
    probs = np.array([_sigmoid(s) for s in scores])
    softmax_w = np.exp(probs) / np.exp(probs).sum()

    rows = []
    for ev, sc, pr, w in zip(evs, scores, probs, softmax_w):
        rows.append({
            **ev,
            "prob_true": round(float(pr), 4),
            "contribution": round(float(w), 4),
            "direction": "support" if pr >= 0.5 else "contradict",
        })

    return sorted(rows, key=lambda r: r["score"], reverse=True)[:top_k]


class SpanHighlightingService:

    def __init__(self):
        self.llm = GigaChatService()

    def decompose_claim(self, news_text: str) -> list[str]:
        system = (
            "Разбей новостное утверждение на атомарные проверяемые факты. "
            "Каждый факт — одно конкретное утверждение. "
            "Выведи нумерованный список, ничего кроме списка."
        )
        raw = self.llm.complete(system, f"Новость: {news_text}")
        claims = []
        for line in raw.split("\n"):
            line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if len(line) > 5:
                claims.append(line)
        return claims

    def extract_span(self, question: str, context: str) -> dict:
        qa = registry.qa
        try:
            out = qa(question=question, context=context[:1500])
            return {
                "answer": out["answer"],
                "confidence": round(float(out["score"]), 4),
            }
        except Exception:
            return {"answer": "", "confidence": 0.0}

    def analyse(self, news_text: str, result: dict,
                top_k_docs: int = 3) -> dict:
        sub_claims = self.decompose_claim(news_text)

        evs = sorted(
            [ev for ev in result.get("evidence", []) if "score" in ev],
            key=lambda e: abs(e["score"]), reverse=True
        )[:top_k_docs]

        evidence_analysis = []
        for ev in evs:
            context = f"{ev['title']}. {ev['content']}"
            sc_results = []
            for sc in sub_claims:
                span = self.extract_span(sc, context)
                sc_results.append({
                    "sub_claim": sc,
                    "evidence_span": span["answer"],
                    "confidence": span["confidence"],
                })
            evidence_analysis.append({
                "domain": ev["domain"],
                "title": ev["title"],
                "score": ev["score"],
                "direction": "support" if ev["score"] > 0 else "contradict",
                "sub_claims": sc_results,
            })

        return {
            "news": news_text,
            "sub_claims": sub_claims,
            "evidence": evidence_analysis,
        }


class ContradictionHeatmapService:

    def build(self, news_text: str, evidence: dict) -> dict:
        claim_sents = _split_sentences(news_text)
        ev_text = f"{evidence.get('title', '')}. {evidence.get('content', '')}"
        ev_sents = _split_sentences(ev_text)

        if not claim_sents or not ev_sents:
            return {
                "claim_sentences": claim_sents,
                "evidence_sentences": ev_sents,
                "cells": [],
                "domain": evidence.get("domain", ""),
                "title": evidence.get("title", ""),
            }

        nli = registry.nli
        cells = []

        for cs in claim_sents:
            for es in ev_sents:
                out = nli(
                    sequences=cs[:512],
                    candidate_labels=_NLI_LABELS,
                    hypothesis_template="{}",
                )
                scores = dict(zip(out["labels"], out["scores"]))
                cells.append({
                    "claim_sentence": cs,
                    "evidence_sentence": es,
                    "entailment": round(scores.get("entailment", 0.0), 4),
                    "neutral": round(scores.get("neutral", 0.0), 4),
                    "contradiction": round(scores.get("contradiction", 0.0), 4),
                })

        return {
            "claim_sentences": claim_sents,
            "evidence_sentences": ev_sents,
            "cells": cells,
            "domain": evidence.get("domain", ""),
            "title": evidence.get("title", ""),
        }


class QuerySensitivityService:

    def __init__(self):
        self.llm = GigaChatService()
        self.search = YandexSearchService()

    def run(self, news_text: str,
            n_trials: int = 3,
            n_queries: int = 5,
            n_results: int = 5) -> dict:

        trial_results = []

        for trial in range(n_trials):
            logger.info("[Sensitivity] Trial %d/%d", trial + 1, n_trials)
            queries = self.llm.generate_queries(news_text, n_queries)
            evidences = self.search.multi_search(queries, n_per_query=n_results)

            query_scores = []
            ce = registry.cross_encoder

            for q, evs in zip(queries, self._group_by_query(queries, evidences, n_results)):
                if evs:
                    pairs = [[news_text, f"{ev['title']}. {ev['content']}"] for ev in evs]
                    scores = ce.predict(pairs).tolist()
                    for ev, sc in zip(evs, scores):
                        ev["score"] = float(sc)
                    query_scores.append({
                        "query": q,
                        "n_results": len(evs),
                        "mean_score": float(np.mean(scores)),
                        "max_score": float(np.max(scores)),
                    })

            all_scores = [ev["score"] for ev in evidences if "score" in ev]
            probability = _sigmoid(float(np.mean(all_scores))) if all_scores else 0.5

            trial_results.append({
                "trial": trial + 1,
                "probability": probability,
                "label": _label(probability),
                "n_evidences": len(evidences),
                "queries": query_scores,
            })
            time.sleep(1)

        probs = [t["probability"] for t in trial_results]
        return {
            "news": news_text,
            "trials": trial_results,
            "mean_prob": float(np.mean(probs)),
            "std_prob": float(np.std(probs)),
            "min_prob": float(np.min(probs)),
            "max_prob": float(np.max(probs)),
            "verdict_stable": bool(np.std(probs) < 0.1),
        }

    @staticmethod
    def _group_by_query(queries: list[str],
                        evidences: list[dict],
                        n_per: int) -> list[list[dict]]:
        """Split flat evidence list back into per-query groups."""
        groups = []
        idx = 0
        for _ in queries:
            groups.append(evidences[idx: idx + n_per])
            idx += n_per
        return groups


def source_credibility(result: dict,
                       overrides: dict[str, float] | None = None) -> dict:

    import copy
    weighted = copy.deepcopy(result)

    for ev in weighted.get("evidence", []):
        cred = _get_credibility(ev.get("domain", ""), overrides)
        ev["credibility"] = cred
        ev["score_weighted"] = ev.get("score", 0.0) * cred

    w_scores = [ev["score_weighted"] for ev in weighted["evidence"]
                if "score_weighted" in ev]
    if w_scores:
        weighted["probability_weighted"] = _sigmoid(float(np.mean(w_scores)))
        weighted["label_weighted"] = _label(weighted["probability_weighted"])
    else:
        weighted["probability_weighted"] = None
        weighted["label_weighted"] = "НЕДОСТАТОЧНО ДАННЫХ"

    weighted["verdict_changed"] = (
            result.get("label") != weighted["label_weighted"]
    )
    return weighted


def temporal_analysis(result: dict,
                      claim_date: datetime | None = None,
                      decay_days: float = 90.0) -> dict:
    """
    Attach parsed dates and recency weights to evidence.
    Recompute probability with recency-weighted scores.
    """
    import copy
    if claim_date is None:
        claim_date = datetime.now(tz=timezone.utc)

    enriched = copy.deepcopy(result)
    dated_count = undated_count = 0

    for ev in enriched.get("evidence", []):
        dt = _parse_date(ev.get("date_str", ""))
        if dt:
            days = abs((claim_date - dt).days)
            ev["parsed_date"] = dt.isoformat()
            ev["days_from_claim"] = days
            ev["recency_weight"] = float(np.exp(-days / decay_days))
            dated_count += 1
        else:
            ev["parsed_date"] = None
            ev["days_from_claim"] = None
            ev["recency_weight"] = 0.5
            undated_count += 1

    w_scores = [
        ev.get("score", 0.0) * ev["recency_weight"]
        for ev in enriched["evidence"] if "score" in ev
    ]
    prob_temporal = _sigmoid(float(np.mean(w_scores))) if w_scores else 0.5

    enriched["dated_count"] = dated_count
    enriched["undated_count"] = undated_count
    enriched["prob_temporal"] = prob_temporal
    enriched["label_temporal"] = _label(prob_temporal)
    enriched["verdict_changed"] = (result.get("label") != enriched["label_temporal"])
    return enriched


def _parse_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    try:
        dt = dateparser.parse(date_str, fuzzy=True)
        if dt and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


FAILURE_CATEGORIES = {
    "no_evidence": "Search returned fewer than 2 documents",
    "low_confidence": "P(true) ∈ [0.4, 0.6] — model uncertain",
    "source_dominated": "One source accounts for >80% of softmax weight",
    "all_neutral": "All evidence scores ≈ 0",
    "correct": "Correct classification",
    "false_positive": "Fake classified as real",
    "false_negative": "Real classified as fake",
}


def _categorise_error(prob: float, pred: int, gold: int,
                      n_evidence: int, scores: list[float]) -> str:
    if n_evidence < 2:
        return "no_evidence"
    if abs(prob - 0.5) < 0.1:
        return "low_confidence"
    if scores:
        sig = np.array([_sigmoid(s) for s in scores])
        sm = np.exp(sig) / np.exp(sig).sum()
        if sm.max() > 0.8:
            return "source_dominated"
    if scores and all(abs(s) < 0.5 for s in scores):
        return "all_neutral"
    if pred == gold:
        return "correct"
    return "false_positive" if pred == 1 else "false_negative"


def error_analysis(results_with_labels: list[dict]) -> dict:
    """
    Compute error taxonomy over a list of:
      {"text": str, "gold": int, "pred": int, "probability": float,
       "evidence": [...], "category": str (computed here)}
    """
    from sklearn.metrics import accuracy_score, f1_score

    records = []
    for item in results_with_labels:
        scores = [ev.get("score", 0.0) for ev in item.get("evidence", [])]
        cat = _categorise_error(
            item.get("probability", 0.5),
            item.get("pred", 0),
            item.get("gold", 0),
            len(item.get("evidence", [])),
            scores,
        )
        records.append({**item, "category": cat})

    valid = [r for r in records if r.get("pred", -1) != -1]
    if not valid:
        return {"records": records, "accuracy": 0.0, "f1_weighted": 0.0,
                "category_counts": {}}

    golds = [r["gold"] for r in valid]
    preds = [r["pred"] for r in valid]
    cats = defaultdict(int)
    for r in valid:
        cats[r["category"]] += 1

    return {
        "records": records,
        "accuracy": float(accuracy_score(golds, preds)),
        "f1_weighted": float(f1_score(golds, preds, average="weighted",
                                      zero_division=0)),
        "category_counts": dict(cats),
    }


def inter_method_agreement(method_results: dict[str, dict],
                           threshold: float = 0.5) -> dict:
    """
    method_results: {"method_name": verify_result_dict, ...}
    Returns agreement metrics and per-method verdicts.
    """
    verdicts = {}
    for name, res in method_results.items():
        prob = res.get("probability")
        lbl = res.get("label", "ОШИБКА")
        verdicts[name] = {
            "label": lbl,
            "probability": prob,
            "pred": 1 if lbl == "ПРАВДИВАЯ" else 0,
        }

    preds = [v["pred"] for v in verdicts.values()
             if verdicts[name]["label"] not in ("ОШИБКА", "НЕДОСТАТОЧНО ДАННЫХ")]
    agree_frac = float(np.mean(preds)) if preds else 0.5
    consensus = _label(agree_frac, threshold)
    disagreement = len(set(preds)) > 1

    kappa = None
    if len(preds) >= 2:
        arr = np.array(preds)
        n, k = 1, len(arr)
        p_e = ((arr.sum() / k) ** 2 + ((k - arr.sum()) / k) ** 2)
        P_i = (arr.sum() * (arr.sum() - 1) + (k - arr.sum()) * (k - arr.sum() - 1)) / (k * (k - 1)) \
            if k > 1 else 1.0
        kappa = float((P_i - p_e) / (1 - p_e)) if abs(1 - p_e) > 1e-9 else 1.0

    return {
        "verdicts": verdicts,
        "agreement_fraction": agree_frac,
        "consensus_label": consensus,
        "disagreement": disagreement,
        "fleiss_kappa": kappa,
    }
