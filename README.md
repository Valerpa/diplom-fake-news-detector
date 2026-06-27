# Fake News Detection API

REST API for Russian-language fake news detection, built with FastAPI.

---

## Project structure

```
fake-news-detector/
├── app/
│   ├── main.py                        # FastAPI app, router wiring, /health
│   ├── core/
│   │   ├── config.py                  # Pydantic-settings (reads .env)
│   │   └── registry.py                # Lazy singleton model loader
│   ├── services/
│   │   ├── search_llm.py              # YandexSearchService, GigaChatService
│   │   └── models/
│   │       ├── main_model.py          # Main verification pipeline
│   │       ├── baselines.py           # 4 baseline methods
│   │       ├── analysis.py            # Explainability modules
│   │       └── utils.py               # Shared utilities
│   ├── routers/
│   │   ├── verify.py                  # POST /verify, /verify/queries, /verify/run
│   │   ├── baselines.py               # POST /baselines/{method}
│   │   ├── analysis.py                # POST /analysis/{module}
│   │   └── compare.py                 # POST /compare
│   └── schemas/
│       ├── requests.py                # Pydantic request models
│       └── responses.py               # Pydantic response models
├── ui/
│   └── streamlit_app.py               # Streamlit UI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick start

### 1. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in:
#   GIGACHAT_CREDENTIALS, GIGACHAT_SCOPE
#   YANDEX_FOLDER_ID, YANDEX_AUTH
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

| Service | URL | Description |
|---|---|---|
| Streamlit UI | `http://localhost:8501` | Visual interface |
| FastAPI backend | `http://localhost:8000` | REST API |
| API docs (Swagger) | `http://localhost:8000/docs` | Auto-generated docs |

The UI starts only after the API passes its health check.

### 3. Run locally (without Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API reference

### `GET /health`
Returns service status and list of currently loaded models.

```json
{
  "status": "ok",
  "device": "cpu",
  "cuda_available": false,
  "loaded_models": ["cross_encoder"]
}
```

---

### `POST /verify` — Main model (full pipeline)

```json
{
  "text": "С 2025 года проезды для пенсионеров в Москве станут бесплатными",
  "num_queries": 5,
  "num_results": 5
}
```

**Response:**
```json
{
  "text": "...",
  "label": "ЛОЖНАЯ",
  "probability": 0.34,
  "queries": ["пенсионеры Москва бесплатный проезд 2025", "..."],
  "evidence": [
    {
      "title": "...", "content": "...",
      "domain": "rbc.ru", "url": "...", "score": -1.23
    }
  ],
  "reasoning": "mean_ce_score=-0.672"
}
```

Labels: `ПРАВДИВАЯ`, `ЛОЖНАЯ`, `НЕДОСТАТОЧНО ДАННЫХ`. Default threshold: `0.6`.

#### Two-step verification

You can split the pipeline into two steps to inspect or edit generated queries:

1. **`POST /verify/queries`** — generate search queries without running the search
2. **`POST /verify/run`** — run search and scoring with a supplied query list

---

### `POST /baselines/{method}`

| Method | Endpoint | Description |
|---|---|---|
| RuBERT | `/baselines/rubert` | Fine-tuned content-only classifier |
| LLM | `/baselines/llm` | GigaChat zero-shot / few-shot |
| CoRAG | `/baselines/corag` | Iterative retrieval (RAGAR, ACL 2024) |
| NLI | `/baselines/nli` | mDeBERTa entailment/contradiction scoring |

All baselines accept the same request body as `/verify` plus method-specific fields (see `/docs`).

#### Training endpoint

```bash
# Fine-tune RuBERT
curl -X POST http://localhost:8000/baselines/rubert/train \
  -F "file=@dataset.csv" -F "text_col=text" -F "label_col=label" -F "epochs=3"
```

---

### `POST /analysis/{module}`

All analysis endpoints accept:
```json
{
  "text": "news text",
  "result": { ... },   // optional: pre-computed /verify result
  "top_k_docs": 3,
  "sensitivity_trials": 3,
  "claim_date": "2025-01-15",
  "credibility_overrides": {"rbc.ru": 0.9}
}
```
If `result` is omitted, the main model runs first automatically.

| Module | Endpoint | Returns |
|---|---|---|
| Attribution | `/analysis/attribution` | Evidence ranked by contribution to verdict |
| Spans | `/analysis/spans` | Sub-claims + contradicting passage per evidence doc |
| Heatmap | `/analysis/heatmap` | NLI sentence×sentence matrix (entailment/contradiction) |
| Sensitivity | `/analysis/sensitivity` | P(true) variance across N independent query sets |
| Signs | `/analysis/signs` | Detection of typical fake news indicators in the text |

---

### `POST /compare` — Multi-method comparison

```json
{
  "text": "...",
  "methods": ["main", "llm_zeroshot", "nli"],
  "threshold": 0.5,
  "gold_label": 0
}
```

**Response:**
```json
{
  "text": "...",
  "gold_label": 0,
  "results": [
    {"method": "main",        "label": "ЛОЖНАЯ",    "probability": 0.34},
    {"method": "llm_zeroshot","label": "ЛОЖНАЯ",    "probability": 0.28},
    {"method": "nli",         "label": "ПРАВДИВАЯ", "probability": 0.55}
  ],
  "agreement_fraction": 0.25,
  "consensus_label": "ЛОЖНАЯ",
  "disagreement": true
}
```

Available method names: `main`, `rubert`, `llm_zeroshot`, `corag`, `nli`

> **Tip:** `corag` in `/compare` adds 30–120s due to iterative retrieval.

---

## Model notes

| Model | When loaded | Approx. size |
|---|---|---|
| `DiTy/cross-encoder-russian-msmarco` | First `/verify` call | ~110 MB |
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | First `/analysis/heatmap` or `/baselines/nli` | ~280 MB |
| `deepset/xlm-roberta-large-squad2` | First `/analysis/spans` | ~1.1 GB |

You can preload models via `POST /models/preload` with a list of model names (`cross_encoder`, `nli`, `qa`).

All models are downloaded from HuggingFace Hub on first use and cached in the `hf-cache` Docker volume.

---

## Dataset format for training

CSV file with at minimum two columns:

| text | label |
|---|---|
| Центробанк повысил ставку до 21% | 1 |
| Земля плоская — учёные признали | 0 |

`label`: `1` = real, `0` = fake.
