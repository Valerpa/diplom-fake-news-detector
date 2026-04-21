import time
from typing import Optional
import httpx
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fake news detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
API_BASE = os.getenv("API_BASE_URL", "http://api:8000")


STRINGS = {
    "en": {
        "page_title":        "Fake news detector",
        "tab_verify":        "Verify",
        "tab_compare":       "Compare methods",
        "tab_analysis":      "Analysis",
        "tab_errors":        "Error analysis",
        # sidebar
        "sidebar_config":    "Configuration",
        "sidebar_method":    "Verification method",
        "sidebar_queries":   "Number of queries",
        "sidebar_results":   "Results per query",
        "sidebar_threshold": "Threshold",
        "sidebar_rounds":    "Max rounds",
        # verify tab
        "news_text_label":   "News text",
        "news_text_ph":      "Paste a Russian-language news article or headline",
        "btn_verify":        "▶ Verify",
        "confidence":        "Confidence",
        "evidence_docs":     "Evidence docs",
        "queries_label":     "Queries",
        "mean_ce":           "Mean CE score",
        "gen_queries":       "Generated queries",
        "evidence_ranked":   "Evidence — ranked by score",
        "supports":          "supports",
        "contradicts":       "contradicts",
        # compare tab
        "cmp_title":         "##### Compare multiple methods on the same news item",
        "cmp_methods":       "Methods to run",
        "cmp_gold":          "Ground-truth label (optional)",
        "cmp_gold_opts":     ["— unknown —", "ФЕЙКОВАЯ (0)", "ПРАВДИВАЯ (1)"],
        "btn_compare":       "▶ Run comparison",
        "cmp_consensus":     "Consensus (majority vote)",
        "cmp_agreement":     "Agreement fraction",
        "cmp_ambiguous":     "⚠ Methods disagree — this news item is genuinely ambiguous.",
        "cmp_col_method":    "Method",
        "cmp_col_prob":      "P(true)",
        "cmp_col_verdict":   "Verdict",
        "cmp_col_agrees":    "Agrees?",
        "cmp_agrees":        "✓",
        "cmp_differs":       "✗ differs",
        # analysis tab
        "an_title":          "##### Run analysis modules on any news text",
        "an_module":         "Module",
        "an_modules":        ["Attribution", "Span highlighting", "NLI heatmap",
                              "Sensitivity", "Credibility", "Temporal"],
        "an_claim_date":     "Claim date (ISO, e.g. 2025-01-15)",
        "an_claim_ph":       "Leave blank to use today",
        "btn_analysis":      "▶ Run analysis",
        "an_attr_title":     "Evidence ranked by contribution to the verdict",
        # errors tab
        "err_title":         "##### Upload a labeled dataset to run error taxonomy analysis",
        "err_caption":       "CSV with two columns: `text` (news text) and `label` (0 = fake, 1 = real)",
        "err_upload":        "Dataset CSV",
        "err_text_col":      "Text column name",
        "err_label_col":     "Label column name",
        "btn_errors":        "▶ Run error analysis",
        "err_accuracy":      "Accuracy",
        "err_f1":            "F1 weighted",
        "err_total":         "Total items",
        "err_errors":        "Errors",
        "err_taxonomy":      "Failure taxonomy",
        "err_records":       "All records",
        "err_gold":          "Gold",
        "err_pred":          "Pred",
        "err_correct":       "Correct",
        "err_category":      "Category",
        # misc
        "api_ok":            "API ✓  ·  device:",
        "api_loaded":        "Loaded:",
        "api_unreachable":   "⚠ API unreachable",
        "enter_text":        "Please enter a news text first.",
        "enter_text2":       "Please enter a news text.",
        "select_method":     "Select at least one method.",
        "api_error":         "API error:",
        "p_true":            "P(true) =",
        "running":           "Running",
        "spinning_attr":     "Running attribution…",
        "spinning_spans":    "Running span highlighting (QA model)…",
        "spinning_heatmap":  "Running NLI heatmap (mDeBERTa)…",
        "spinning_sens":     "Running sensitivity analysis (3 trials)…",
        "spinning_cred":     "Running credibility weighting…",
        "spinning_temp":     "Running temporal analysis…",
        "attr_expander":     "Attribution — evidence contributions",
        "spans_expander":    "Span highlighting — sub-claims",
        "heatmap_expander":  "NLI heatmap — contradiction matrix",
        "sens_expander":     "Sensitivity — verdict stability",
        "cred_expander":     "Credibility — source-weighted verdict",
        "temp_expander":     "Temporal — recency-weighted verdict",
        "cred_orig":         "Original verdict",
        "cred_weighted":     "Weighted verdict",
        "cred_changed":      "⚠ Verdict changed after credibility weighting — low-credibility sources were influencing the result.",
        "temp_dated":        "Dated docs",
        "temp_undated":      "Undated docs",
        "temp_verdict":      "Temporal verdict",
        "temp_changed":      "⚠ Temporal weighting changed the verdict.",
        "sens_std":          "Std deviation",
        "sens_stable":       "✓ stable",
        "sens_unstable":     "⚠ unstable",
        "sens_mean":         "Mean P(true)",
        "sens_stable_q":     "Stable?",
        "sens_yes":          "Yes ✓",
        "sens_no":           "No ⚠",
        "sub_claims":        "Atomic sub-claims:",
        "no_evidence":       "No evidence documents retrieved.",
        "verifying_batch":   "Running verification on each item…",
        "verified":          "Verified",
        "computing_errors":  "Computing error taxonomy…",
        "col_not_found":     "not found in CSV.",
        "lang_toggle":       "🇷🇺 RU",
    },
    "ru": {
        "page_title":        "Детектор фейков",
        "tab_verify":        "Проверка",
        "tab_compare":       "Сравнение методов",
        "tab_analysis":      "Анализ",
        "tab_errors":        "Анализ ошибок",
        # sidebar
        "sidebar_config":    "Настройки",
        "sidebar_method":    "Метод верификации",
        "sidebar_queries":   "Количество запросов",
        "sidebar_results":   "Результатов на запрос",
        "sidebar_threshold": "Порог",
        "sidebar_rounds":    "Макс. раундов",
        # verify tab
        "news_text_label":   "Текст новости",
        "news_text_ph":      "Вставьте текст новости или заголовок",
        "btn_verify":        "▶ Проверить",
        "confidence":        "Уверенность",
        "evidence_docs":     "Документов",
        "queries_label":     "Запросов",
        "mean_ce":           "Средний CE score",
        "gen_queries":       "Сгенерированные запросы",
        "evidence_ranked":   "Доказательства — по убыванию score",
        "supports":          "подтверждает",
        "contradicts":       "опровергает",
        # compare tab
        "cmp_title":         "##### Сравнить несколько методов на одной новости",
        "cmp_methods":       "Методы",
        "cmp_gold":          "Истинная метка (необязательно)",
        "cmp_gold_opts":     ["— неизвестно —", "ФЕЙКОВАЯ (0)", "ПРАВДИВАЯ (1)"],
        "cmp_consensus":     "Консенсус (большинство голосов)",
        "cmp_agreement":     "Доля согласия",
        "cmp_ambiguous":     "⚠ Методы расходятся — новость неоднозначна.",
        "btn_compare":       "▶ Запустить сравнение",
        "cmp_col_method":    "Метод",
        "cmp_col_prob":      "P(правда)",
        "cmp_col_verdict":   "Вердикт",
        "cmp_col_agrees":    "Согласен?",
        "cmp_agrees":        "✓",
        "cmp_differs":       "✗ расходится",
        # analysis tab
        "an_title":          "##### Запустить модуль анализа",
        "an_module":         "Модуль",
        "an_modules":        ["Attribution", "Выделение фрагментов", "NLI тепловая карта",
                              "Чувствительность", "Достоверность источников", "Временной анализ"],
        "an_claim_date":     "Дата публикации (ISO, напр. 2025-01-15)",
        "an_claim_ph":       "Оставьте пустым для текущей даты",
        "btn_analysis":      "▶ Запустить анализ",
        "an_attr_title":     "Доказательства по вкладу в вердикт",
        # errors tab
        "err_title":         "##### Загрузите размеченный датасет для анализа ошибок",
        "err_caption":       "CSV с колонками: `text` (текст новости) и `label` (0 = фейк, 1 = правда)",
        "err_upload":        "CSV файл",
        "err_text_col":      "Название колонки с текстом",
        "err_label_col":     "Название колонки с меткой",
        "btn_errors":        "▶ Запустить анализ ошибок",
        "err_accuracy":      "Точность",
        "err_f1":            "F1 взвеш.",
        "err_total":         "Всего",
        "err_errors":        "Ошибок",
        "err_taxonomy":      "Таксономия ошибок",
        "err_records":       "Все записи",
        "err_gold":          "Истина",
        "err_pred":          "Предсказание",
        "err_correct":       "Верно",
        "err_category":      "Категория",
        # misc
        "api_ok":            "API ✓  ·  устройство:",
        "api_loaded":        "Загружено:",
        "api_unreachable":   "⚠ API недоступен",
        "enter_text":        "Пожалуйста, введите текст новости.",
        "enter_text2":       "Пожалуйста, введите текст новости.",
        "select_method":     "Выберите хотя бы один метод.",
        "api_error":         "Ошибка API:",
        "p_true":            "P(правда) =",
        "running":           "Запуск",
        "spinning_attr":     "Запуск attribution…",
        "spinning_spans":    "Запуск выделения фрагментов (QA модель)…",
        "spinning_heatmap":  "Запуск NLI тепловой карты (mDeBERTa)…",
        "spinning_sens":     "Запуск анализа чувствительности (3 прогона)…",
        "spinning_cred":     "Запуск взвешивания по достоверности…",
        "spinning_temp":     "Запуск временного анализа…",
        "attr_expander":     "Attribution — вклад доказательств",
        "spans_expander":    "Выделение фрагментов — атомарные утверждения",
        "heatmap_expander":  "NLI тепловая карта — матрица противоречий",
        "sens_expander":     "Чувствительность — стабильность вердикта",
        "cred_expander":     "Достоверность — взвешенный вердикт",
        "temp_expander":     "Временной анализ — взвешенный вердикт",
        "cred_orig":         "Исходный вердикт",
        "cred_weighted":     "Взвешенный вердикт",
        "cred_changed":      "⚠ Вердикт изменился после взвешивания — низкодостоверные источники влияли на результат.",
        "temp_dated":        "С датой",
        "temp_undated":      "Без даты",
        "temp_verdict":      "Временной вердикт",
        "temp_changed":      "⚠ Временное взвешивание изменило вердикт.",
        "sens_std":          "Стд. отклонение",
        "sens_stable":       "✓ стабильный",
        "sens_unstable":     "⚠ нестабильный",
        "sens_mean":         "Среднее P(правда)",
        "sens_stable_q":     "Стабилен?",
        "sens_yes":          "Да ✓",
        "sens_no":           "Нет ⚠",
        "sub_claims":        "Атомарные утверждения:",
        "no_evidence":       "Документы-доказательства не найдены.",
        "verifying_batch":   "Верификация каждого элемента…",
        "verified":          "Проверено",
        "computing_errors":  "Вычисление таксономии ошибок…",
        "col_not_found":     "не найдена в CSV.",
        "lang_toggle":       "🇬🇧 EN",
    },
}


if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

def T(key: str) -> str:
    return STRINGS[st.session_state["lang"]].get(key, key)


st.markdown("""
<style>
/* Base font size bump — all rem values scale from this */
html, body, [class*="css"] {
    font-size: 16px !important;
}

[data-testid="stAppViewContainer"] > .main { padding-top: 1.2rem; }
[data-testid="stHeader"] { background: transparent; }

[data-testid="stSidebar"] {
    background: #f8f7f4;
    border-right: 1px solid #e5e3dc;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label {
    font-size: 0.90rem !important;
    color: #555 !important;
}
[data-testid="stSidebar"] h3 {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: #999 !important;
    margin-top: 1.2rem;
    margin-bottom: 0.2rem;
}

/* Streamlit default text elements */
p, li, .stMarkdown, .stCaption, label { font-size: 0.95rem !important; }
.stTextArea textarea { font-size: 0.95rem !important; }
.stTextInput input  { font-size: 0.95rem !important; }
.stSelectbox div[data-baseweb="select"] { font-size: 0.95rem !important; }
.stMultiSelect span { font-size: 0.92rem !important; }
.stDataFrame        { font-size: 0.90rem !important; }
.stExpander summary { font-size: 0.95rem !important; }
[data-testid="stMetricLabel"]  { font-size: 0.88rem !important; }
[data-testid="stMetricValue"]  { font-size: 1.5rem  !important; }

/* Verdict badges */
.verdict-fake {
    display: inline-flex; align-items: center; gap: 8px;
    background: #fcebeb; color: #a32d2d;
    padding: 7px 18px; border-radius: 8px;
    font-size: 1.05rem; font-weight: 600;
}
.verdict-real {
    display: inline-flex; align-items: center; gap: 8px;
    background: #eaf3de; color: #3b6d11;
    padding: 7px 18px; border-radius: 8px;
    font-size: 1.05rem; font-weight: 600;
}
.verdict-dot-fake {
    width: 10px; height: 10px; border-radius: 50%;
    background: #e24b4a; display: inline-block;
}
.verdict-dot-real {
    width: 10px; height: 10px; border-radius: 50%;
    background: #639922; display: inline-block;
}
.verdict-unknown {
    display: inline-flex; align-items: center; gap: 8px;
    background: #f1efe8; color: #888780;
    padding: 7px 18px; border-radius: 8px;
    font-size: 1.05rem; font-weight: 600;
}

/* Evidence cards */
.ev-card {
    border: 0.5px solid #e5e3dc;
    border-radius: 10px;
    padding: 11px 15px;
    margin-bottom: 9px;
    background: #fff;
}
.ev-domain  { font-weight: 600; font-size: 0.92rem; color: #1a1a1a; }
.ev-snippet { font-size: 0.86rem; color: #555; line-height: 1.6; margin-top: 4px; }
.tag-contra {
    display: inline-block; font-size: 0.75rem; padding: 2px 9px;
    border-radius: 999px; background: #fcebeb; color: #a32d2d; margin-left: 7px;
}
.tag-support {
    display: inline-block; font-size: 0.75rem; padding: 2px 9px;
    border-radius: 999px; background: #eaf3de; color: #3b6d11; margin-left: 7px;
}

/* Query pills */
.pill {
    display: inline-block; font-size: 0.78rem;
    background: #f1efe8; color: #555;
    border: 0.5px solid #d3d1c7;
    padding: 4px 12px; border-radius: 999px;
    margin: 2px 4px 2px 0;
}

/* Section divider */
.section-rule {
    border: none;
    border-top: 0.5px solid #e5e3dc;
    margin: 1.3rem 0 1.1rem;
}

/* Confidence bar */
.conf-wrap   { position: relative; margin: 5px 0 3px; }
.conf-track  {
    height: 11px; background: #f1efe8;
    border-radius: 6px; overflow: visible;
    border: 0.5px solid #d3d1c7; position: relative;
}
.conf-fill-fake { height: 100%; border-radius: 6px; background: #e24b4a; transition: width 0.4s; }
.conf-fill-real { height: 100%; border-radius: 6px; background: #639922; transition: width 0.4s; }
.conf-marker {
    position: absolute; top: -4px;
    width: 2px; height: 19px; background: #888780;
}
.conf-labels {
    display: flex; justify-content: space-between;
    font-size: 0.75rem; color: #999; margin-top: 4px;
}

/* Metric card */
.met-card   { background: #f8f7f4; border-radius: 8px; padding: 11px 15px; }
.met-label  { font-size: 0.78rem; color: #999; margin-bottom: 4px; }
.met-value  { font-size: 1.5rem; font-weight: 500; color: #1a1a1a; }

/* Warning box */
.warn-box {
    background: #faeeda; color: #633806;
    border-radius: 8px; padding: 9px 13px;
    font-size: 0.88rem; margin-bottom: 9px;
}

/* Language toggle button */
.lang-btn button {
    font-size: 0.85rem !important;
    padding: 4px 14px !important;
    border-radius: 999px !important;
}
</style>
""", unsafe_allow_html=True)


def _get(path: str, timeout: float = 10.0):
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _post(path: str, body: dict, timeout: float = 120.0):
    try:
        r = httpx.post(f"{API_BASE}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def render_verdict_badge(label: str):
    if label == "ПРАВДИВАЯ":
        st.markdown('<span class="verdict-real"><span class="verdict-dot-real"></span> ПРАВДИВАЯ</span>',
                    unsafe_allow_html=True)
    elif label == "ФЕЙКОВАЯ":
        st.markdown('<span class="verdict-fake"><span class="verdict-dot-fake"></span> ФЕЙКОВАЯ</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="verdict-unknown">{label}</span>', unsafe_allow_html=True)


def render_confidence_bar(prob: Optional[float], threshold: float = 0.5):
    if prob is None:
        return
    pct = int(prob * 100)
    thr = int(threshold * 100)
    fill = "conf-fill-real" if prob >= threshold else "conf-fill-fake"
    st.markdown(f"""
<div class="conf-wrap">
  <div class="conf-track">
    <div class="{fill}" style="width:{pct}%"></div>
    <div class="conf-marker" style="left:{thr}%"></div>
  </div>
  <div class="conf-labels">
    <span>ФЕЙКОВАЯ</span>
    <span>threshold {threshold:.2f}</span>
    <span>ПРАВДИВАЯ</span>
  </div>
</div>""", unsafe_allow_html=True)


def render_evidence_list(evidences: list, max_show: int = 8):
    if not evidences:
        st.caption(T("no_evidence"))
        return
    shown = sorted([e for e in evidences if e.get("score") is not None],
                   key=lambda e: e["score"])
    for ev in shown[:max_show]:
        score = ev.get("score", 0)
        tag   = f'<span class="tag-support">{T("supports")}</span>' if score > 0 \
                else f'<span class="tag-contra">{T("contradicts")}</span>'
        score_str = f"+{score:.2f}" if score > 0 else f"{score:.2f}"
        st.markdown(f"""
<div class="ev-card">
  <div>
    <span class="ev-domain">{ev.get('domain','')}</span>{tag}
    <span style="float:right;font-size:0.80rem;color:#999;font-family:monospace">{score_str}</span>
  </div>
  <div class="ev-snippet">{ev.get('title','')} — {ev.get('content','')[:160]}</div>
</div>""", unsafe_allow_html=True)


def render_query_pills(queries: list):
    if not queries:
        return
    pills = "".join(f'<span class="pill">{q}</span>' for q in queries)
    st.markdown(f'<div style="margin:5px 0 9px">{pills}</div>', unsafe_allow_html=True)


def render_metrics_row(items: list):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(
            f'<div class="met-card"><div class="met-label">{label}</div>'
            f'<div class="met-value">{value}</div></div>',
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.markdown(f"### {T('sidebar_config')}")

    method = st.selectbox(
        T("sidebar_method"),
        options=["Main model", "CoRAG", "STEEL",
                 "NLI classifier", "LLM zero-shot", "RuBERT", "GNN"],
        index=0,
    )

    METHOD_MAP = {
        "Main model":     ("main",   "/verify"),
        "CoRAG":          ("corag",  "/baselines/corag"),
        "STEEL":          ("steel",  "/baselines/steel"),
        "NLI classifier": ("nli",    "/baselines/nli"),
        "LLM zero-shot":  ("llm",    "/baselines/llm"),
        "RuBERT":         ("rubert", "/baselines/rubert"),
        "GNN":            ("gnn",    "/baselines/gnn"),
    }

    num_queries = st.slider(T("sidebar_queries"), 1, 10, 5, step=1)
    num_results = st.slider(T("sidebar_results"), 1, 20, 5, step=1)
    threshold   = st.slider(T("sidebar_threshold"), 0.0, 1.0, 0.5, step=0.01)

    if method in ("CoRAG", "STEEL"):
        max_rounds = st.slider(T("sidebar_rounds"), 1, 8, 4, step=1)
    else:
        max_rounds = 4

    st.markdown("---")
    health = _get("/health", timeout=3.0)
    if "error" not in health:
        st.caption(f"{T('api_ok')} {health.get('device','?')}")
        loaded = health.get("loaded_models", [])
        if loaded:
            st.caption(f"{T('api_loaded')} " + ", ".join(loaded))
    else:
        st.caption(T("api_unreachable"))


title_col, badge_col, lang_col = st.columns([5, 2, 1])

with title_col:
    st.markdown(f"## {T('page_title')}")


with lang_col:

    st.markdown('<div class="lang-btn">', unsafe_allow_html=True)
    if st.button(T("lang_toggle"), key="lang_btn"):
        st.session_state["lang"] = "ru" if st.session_state["lang"] == "en" else "en"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


tab_verify, tab_compare, tab_analysis, tab_errors = st.tabs([
    T("tab_verify"), T("tab_compare"), T("tab_analysis"), T("tab_errors")
])


with tab_verify:
    news_text = st.text_area(
        T("news_text_label"),
        placeholder=T("news_text_ph"),
        height=110,
    )

    verify_btn = st.button(T("btn_verify"), type="primary", key="btn_verify")

    if verify_btn:
        if not news_text.strip():
            st.warning(T("enter_text"))
        else:
            method_key, endpoint = METHOD_MAP[method]
            body = {
                "text":        news_text,
                "threshold":   threshold,
                "num_queries": num_queries,
                "num_results": num_results,
                "max_rounds":  max_rounds,
            }
            with st.spinner(f"{T('running')} {method}…"):
                result = _post(endpoint, body)
            st.session_state["last_result"] = result
            st.session_state["last_text"]   = news_text

    result = st.session_state.get("last_result")
    if result:
        if "error" in result:
            st.error(f"{T('api_error')} {result['error']}")
        else:
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

            v_col, p_col = st.columns([3, 1])
            with v_col:
                render_verdict_badge(result.get("label", "?"))
            with p_col:
                prob  = result.get("probability")
                p_str = f"{prob:.3f}" if prob is not None else "N/A"
                st.markdown(
                    f'<div style="text-align:right;padding-top:7px">'
                    f'<span style="font-size:0.90rem;color:#999">{T("p_true")} </span>'
                    f'<span style="font-size:1.15rem;font-weight:500">{p_str}</span></div>',
                    unsafe_allow_html=True,
                )

            st.caption(T("confidence"))
            render_confidence_bar(result.get("probability"), threshold)

            st.markdown("")
            n_ev = len(result.get("evidence", []))
            n_q  = len(result.get("queries",  []))
            import re as _re
            ce_match = _re.search(r"mean_ce_score=([-\d.]+)", result.get("reasoning", ""))
            ce_str   = ce_match.group(1) if ce_match else "—"
            render_metrics_row([
                (T("evidence_docs"), n_ev),
                (T("queries_label"), n_q),
                (T("mean_ce"),       ce_str),
            ])

            queries = result.get("queries", [])
            if queries:
                st.markdown("")
                st.caption(T("gen_queries"))
                render_query_pills(queries)

            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.caption(T("evidence_ranked"))
            render_evidence_list(result.get("evidence", []))


with tab_compare:
    st.markdown(T("cmp_title"))

    cmp_text = st.text_area(
        T("news_text_label"),
        value=st.session_state.get("last_text", ""),
        height=90,
        key="cmp_text",
    )

    method_options = {
        "Main model":    "main",
        "LLM zero-shot": "llm_zeroshot",
        "CoRAG":         "corag",
        "STEEL":         "steel",
        "NLI":           "nli",
        "RuBERT":        "rubert",
        "GNN":           "gnn",
    }
    selected_methods = st.multiselect(
        T("cmp_methods"),
        options=list(method_options.keys()),
        default=["Main model", "LLM zero-shot"],
    )

    gold_col, _ = st.columns([2, 6])
    with gold_col:
        gold_opts  = T("cmp_gold_opts")
        gold_input = st.selectbox(T("cmp_gold"), options=gold_opts)

    gold_label = None
    if "ФЕЙКОВАЯ" in gold_input:  gold_label = 0
    elif "ПРАВДИВАЯ" in gold_input: gold_label = 1

    if st.button(T("btn_compare"), type="primary", key="btn_compare"):
        if not cmp_text.strip():
            st.warning(T("enter_text2"))
        elif not selected_methods:
            st.warning(T("select_method"))
        else:
            body = {
                "text":        cmp_text,
                "methods":     [method_options[m] for m in selected_methods],
                "threshold":   threshold,
                "num_queries": num_queries,
                "gold_label":  gold_label,
            }
            with st.spinner(T("btn_compare") + "…"):
                cmp_result = _post("/compare", body, timeout=300.0)
            st.session_state["cmp_result"] = cmp_result

    cmp_result = st.session_state.get("cmp_result")
    if cmp_result:
        if "error" in cmp_result:
            st.error(cmp_result["error"])
        else:
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

            cons_label = cmp_result.get("consensus_label", "?")
            agree_frac = cmp_result.get("agreement_fraction", 0.5)
            disagree   = cmp_result.get("disagreement", False)

            cons_col, agree_col = st.columns([2, 3])
            with cons_col:
                st.caption(T("cmp_consensus"))
                render_verdict_badge(cons_label)
            with agree_col:
                st.caption(f"{T('cmp_agreement')}: {agree_frac:.0%}")
                st.progress(agree_frac)
                if disagree:
                    st.markdown(f'<div class="warn-box">{T("cmp_ambiguous")}</div>',
                                unsafe_allow_html=True)

            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            h1, h2, h3, h4 = st.columns([2, 4, 1.5, 1.5])
            h1.caption(T("cmp_col_method"))
            h2.caption(T("cmp_col_prob"))
            h3.caption(T("cmp_col_verdict"))
            h4.caption(T("cmp_col_agrees"))

            for r in cmp_result.get("results", []):
                prob   = r.get("probability")
                label  = r.get("label", "?")
                agrees = label == cons_label
                c1, c2, c3, c4 = st.columns([2, 4, 1.5, 1.5])
                c1.markdown(f'<div style="font-size:0.90rem;font-weight:500;padding-top:6px">{r["method"]}</div>',
                            unsafe_allow_html=True)
                if prob is not None: c2.progress(prob)
                else: c2.caption("—")
                vc = "#a32d2d" if label == "ФЕЙКОВАЯ" else "#3b6d11" if label == "ПРАВДИВАЯ" else "#888"
                c3.markdown(f'<div style="font-size:0.90rem;font-weight:500;color:{vc};padding-top:6px">{label}</div>',
                            unsafe_allow_html=True)
                ac = "color:#3b6d11" if agrees else "color:#a32d2d"
                sym = T("cmp_agrees") if agrees else T("cmp_differs")
                c4.markdown(f'<div style="font-size:0.88rem;{ac};padding-top:6px">{sym}</div>',
                            unsafe_allow_html=True)


with tab_analysis:
    st.markdown(T("an_title"))

    an_text = st.text_area(
        T("news_text_label"),
        value=st.session_state.get("last_text", ""),
        height=90,
        key="an_text",
    )

    an_module_options = T("an_modules")
    an_module_key_map = {
        "Attribution":                     "/analysis/attribution",
        "Span highlighting":               "/analysis/spans",
        "Выделение фрагментов":            "/analysis/spans",
        "NLI heatmap":                     "/analysis/heatmap",
        "NLI тепловая карта":              "/analysis/heatmap",
        "Sensitivity":                     "/analysis/sensitivity",
        "Чувствительность":               "/analysis/sensitivity",
        "Credibility":                     "/analysis/credibility",
        "Достоверность источников":        "/analysis/credibility",
        "Temporal":                        "/analysis/temporal",
        "Временной анализ":                "/analysis/temporal",
    }
    an_timeout_map = {
        "/analysis/attribution":  120,
        "/analysis/spans":        240,
        "/analysis/heatmap":      240,
        "/analysis/sensitivity":  360,
        "/analysis/credibility":  120,
        "/analysis/temporal":     120,
    }

    an_module = st.selectbox(T("an_module"), options=an_module_options, key="an_module")

    claim_date = None
    if an_module in ("Temporal", "Временной анализ"):
        claim_date = st.text_input(
            T("an_claim_date"),
            placeholder=T("an_claim_ph"),
            key="claim_date",
        )

    if st.button(T("btn_analysis"), type="primary", key="btn_analysis"):
        if not an_text.strip():
            st.warning(T("enter_text2"))
        else:
            endpoint = an_module_key_map.get(an_module, "/analysis/attribution")
            timeout  = an_timeout_map.get(endpoint, 120)
            body = {
                "text":               an_text,
                "result":             st.session_state.get("last_result"),
                "top_k_docs":         3,
                "sensitivity_trials": 3,
            }
            if claim_date:
                body["claim_date"] = claim_date

            with st.spinner(f"{T('running')} {an_module}…"):
                an_result = _post(endpoint, body, timeout=float(timeout))

            st.session_state["an_result"]     = an_result
            st.session_state["an_module_ran"] = endpoint

    an_result     = st.session_state.get("an_result")
    an_module_ran = st.session_state.get("an_module_ran", "")

    if an_result:
        if "error" in an_result:
            st.error(f"{T('api_error')} {an_result['error']}")
        else:
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

            if an_module_ran == "/analysis/attribution" and isinstance(an_result, list):
                st.caption(T("an_attr_title"))
                df_attr = pd.DataFrame(an_result)[
                    ["domain", "title", "score", "contribution", "direction"]
                ].rename(columns={
                    "domain":       "Domain",
                    "title":        "Title",
                    "score":        "CE score",
                    "contribution": "Contribution",
                    "direction":    "Direction",
                })
                df_attr["Title"] = df_attr["Title"].str[:60]
                st.dataframe(df_attr, use_container_width=True, hide_index=True)

            elif an_module_ran == "/analysis/spans":
                with st.expander(T("spans_expander"), expanded=True):
                    sub_claims = an_result.get("sub_claims", [])
                    if sub_claims:
                        st.caption(T("sub_claims"))
                        for i, sc in enumerate(sub_claims, 1):
                            st.markdown(f"**{i}.** {sc}")
                    for ev_data in an_result.get("evidence", []):
                        st.markdown(
                            f'<div style="font-size:0.90rem;font-weight:500;margin:10px 0 4px">'
                            f'[{ev_data["domain"]}] {ev_data["title"][:60]}</div>',
                            unsafe_allow_html=True)
                        for sc_data in ev_data.get("sub_claims", []):
                            span_text = sc_data.get("evidence_span") or "—"
                            conf      = sc_data.get("confidence", 0)
                            st.markdown(
                                f'<div style="font-size:0.84rem;padding:4px 0;border-bottom:0.5px solid #f1efe8">'
                                f'<span style="color:#999">{sc_data["sub_claim"][:50]}</span>'
                                f' → <em>«{span_text[:80]}»</em>'
                                f' <span style="color:#bbb;font-size:0.76rem">conf={conf:.2f}</span></div>',
                                unsafe_allow_html=True)

            elif an_module_ran == "/analysis/heatmap":
                cells   = an_result.get("cells", [])
                c_sents = an_result.get("claim_sentences", [])
                e_sents = an_result.get("evidence_sentences", [])
                if cells and c_sents and e_sents:
                    with st.expander(T("heatmap_expander"), expanded=True):
                        st.caption(f"[{an_result.get('domain','')}] {an_result.get('title','')[:60]}")
                        import numpy as np
                        mat = np.zeros((len(c_sents), len(e_sents)))
                        for cell in cells:
                            ci = c_sents.index(cell["claim_sentence"])
                            ei = e_sents.index(cell["evidence_sentence"])
                            mat[ci, ei] = cell["contradiction"]
                        df_hm = pd.DataFrame(
                            mat,
                            index=[f"C{i+1}" for i in range(len(c_sents))],
                            columns=[f"E{j+1}" for j in range(len(e_sents))],
                        )
                        st.dataframe(
                            df_hm.style.background_gradient(cmap="Reds", vmin=0, vmax=1).format("{:.2f}"),
                            use_container_width=True,
                        )
                        st.caption("  ".join(f"E{j+1}: {s[:60]}" for j, s in enumerate(e_sents)))

            elif an_module_ran == "/analysis/sensitivity":
                with st.expander(T("sens_expander"), expanded=True):
                    trials = an_result.get("trials", [])
                    std    = an_result.get("std_prob", 0)
                    stable = an_result.get("verdict_stable", False)
                    st.metric(T("sens_std"), f"{std:.3f}",
                              delta=T("sens_stable") if stable else T("sens_unstable"),
                              delta_color="normal" if stable else "inverse")
                    df_s = pd.DataFrame([{
                        "Trial":   t["trial"],
                        "P(true)": round(t["probability"], 3),
                        "Verdict": t["label"],
                        "Docs":    t["n_evidences"],
                    } for t in trials])
                    st.dataframe(df_s, use_container_width=True, hide_index=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric(T("sens_mean"),    f"{an_result.get('mean_prob',0):.3f}")
                    c2.metric(T("sens_std"),      f"{an_result.get('std_prob',0):.3f}")
                    c3.metric(T("sens_stable_q"), T("sens_yes") if stable else T("sens_no"))

            elif an_module_ran == "/analysis/credibility":
                with st.expander(T("cred_expander"), expanded=True):
                    changed = an_result.get("verdict_changed", False)
                    c1, c2  = st.columns(2)
                    c1.metric(T("cred_orig"),    an_result.get("original_label","?"),
                              f"P = {an_result.get('original_probability',0):.3f}")
                    c2.metric(T("cred_weighted"), an_result.get("weighted_label","?"),
                              f"P = {an_result.get('weighted_probability',0):.3f}",
                              delta_color="inverse" if changed else "normal")
                    if changed:
                        st.markdown(f'<div class="warn-box">{T("cred_changed")}</div>',
                                    unsafe_allow_html=True)

            elif an_module_ran == "/analysis/temporal":
                with st.expander(T("temp_expander"), expanded=True):
                    tc1, tc2, tc3 = st.columns(3)
                    tc1.metric(T("temp_dated"),   an_result.get("dated_count",   0))
                    tc2.metric(T("temp_undated"),  an_result.get("undated_count", 0))
                    tc3.metric(T("temp_verdict"),  an_result.get("label_temporal","?"),
                               f"P = {an_result.get('prob_temporal',0):.3f}")
                    if an_result.get("verdict_changed"):
                        st.markdown(f'<div class="warn-box">{T("temp_changed")}</div>',
                                    unsafe_allow_html=True)
            else:
                st.json(an_result)


with tab_errors:
    st.markdown(T("err_title"))
    st.caption(T("err_caption"))

    uploaded     = st.file_uploader(T("err_upload"), type=["csv"])
    ea_text_col  = st.text_input(T("err_text_col"),  value="text",  key="ea_tc")
    ea_label_col = st.text_input(T("err_label_col"), value="label", key="ea_lc")

    if uploaded and st.button(T("btn_errors"), type="primary", key="btn_errors"):
        df_up = pd.read_csv(uploaded)
        if ea_text_col not in df_up.columns or ea_label_col not in df_up.columns:
            st.error(f"'{ea_text_col}' / '{ea_label_col}' {T('col_not_found')}")
        else:
            items    = df_up[[ea_text_col, ea_label_col]].dropna().rename(
                columns={ea_text_col: "text", ea_label_col: "label"}
            ).to_dict("records")
            progress = st.progress(0, text=T("verifying_batch"))
            batch    = []
            for idx, item in enumerate(items):
                res  = _post("/verify", {
                    "text": item["text"], "num_queries": num_queries,
                    "num_results": num_results, "threshold": threshold,
                }, timeout=120.0)
                pred = 1 if res.get("label") == "ПРАВДИВАЯ" else 0
                batch.append({
                    "text": item["text"], "gold": int(item["label"]),
                    "pred": pred, "probability": res.get("probability"),
                    "evidence": res.get("evidence", []),
                })
                progress.progress(
                    (idx + 1) / len(items),
                    text=f"{T('verified')} {idx+1}/{len(items)}…",
                )
                time.sleep(0.5)
            with st.spinner(T("computing_errors")):
                ea_result = _post("/analysis/errors", batch)
            st.session_state["ea_result"] = ea_result

    ea_result = st.session_state.get("ea_result")
    if ea_result and "error" not in ea_result:
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        render_metrics_row([
            (T("err_accuracy"), f"{ea_result.get('accuracy',0):.3f}"),
            (T("err_f1"),       f"{ea_result.get('f1_weighted',0):.3f}"),
            (T("err_total"),    len(ea_result.get("records",[]))),
            (T("err_errors"),   sum(1 for r in ea_result.get("records",[])
                                    if r.get("category") != "correct")),
        ])

        cat_counts = ea_result.get("category_counts", {})
        if cat_counts:
            st.markdown("")
            st.caption(T("err_taxonomy"))
            CAT_COLORS = {
                "correct":          "#eaf3de",
                "false_positive":   "#fcebeb",
                "false_negative":   "#faeeda",
                "no_evidence":      "#f1efe8",
                "low_confidence":   "#e6f1fb",
                "source_dominated": "#eeedfe",
                "all_neutral":      "#e1f5ee",
            }
            total_cats = sum(cat_counts.values())
            for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
                pct   = count / total_cats if total_cats else 0
                color = CAT_COLORS.get(cat, "#f1efe8")
                st.markdown(
                    f'<div style="display:grid;grid-template-columns:160px 1fr 60px;'
                    f'gap:8px;align-items:center;padding:6px 9px;margin-bottom:4px;'
                    f'background:{color};border-radius:6px;font-size:0.86rem">'
                    f'<span style="color:#444">{cat}</span>'
                    f'<div style="height:5px;background:rgba(0,0,0,0.08);border-radius:3px;overflow:hidden">'
                    f'<div style="width:{int(pct*100)}%;height:100%;background:rgba(0,0,0,0.2)"></div></div>'
                    f'<span style="text-align:right;color:#666">{count} ({pct:.0%})</span></div>',
                    unsafe_allow_html=True,
                )

        records = ea_result.get("records", [])
        if records:
            st.markdown("")
            st.caption(T("err_records"))
            st.dataframe(pd.DataFrame([{
                "Text":            r.get("text","")[:70]+"…",
                T("err_gold"):     "ПРАВДА" if r.get("gold")==1 else "ФЕЙК",
                T("err_pred"):     "ПРАВДА" if r.get("pred")==1 else "ФЕЙК",
                "P(true)":         round(r.get("probability") or 0, 3),
                T("err_category"): r.get("category",""),
                T("err_correct"):  "✓" if r.get("correct") else "✗",
            } for r in records]), use_container_width=True, hide_index=True)
