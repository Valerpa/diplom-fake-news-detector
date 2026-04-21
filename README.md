# Fake News Detection API

REST API for Russian-language fake news detection, built with FastAPI.

---

## Project structure

```
fakenews/
├── app/
│   ├── main.py                        # FastAPI app, router wiring, /health
│   ├── core/
│   │   ├── config.py                  # Pydantic-settings (reads .env)
│   │   └── registry.py                # Lazy singleton model loader
│   ├── services/
│   │   ├── search_llm.py              # YandexSearchService, GigaChatService
│   │   └── models/
│   │       ├── main_model.py          # Original pipeline
│   │       ├── baselines.py           # 7 baseline methods
│   │       └── analysis.py            # 8 explainability modules
│   ├── routers/
│   │   ├── verify.py                  # POST /verify
│   │   ├── baselines.py               # POST /baselines/{method}
│   │   ├── analysis.py                # POST /analysis/{module}
│   │   └── compare.py                 # POST /compare
│   └── schemas/
│       ├── requests.py                # Pydantic request models
│       └── responses.py               # Pydantic response models
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

### `POST /verify` — Main model

```json
{
  "text": "С 2025 года проезды для пенсионеров в Москве станут бесплатными",
  "num_queries": 5,
  "num_results": 5,
  "threshold": 0.5
}
```

**Response:**
```json
{
  "text": "...",
  "label": "ФЕЙКОВАЯ",
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

---

### `POST /baselines/{method}`

| Method | Endpoint | Description |
|---|---|---|
| RuBERT | `/baselines/rubert` | Fine-tuned content-only classifier |
| LLM | `/baselines/llm` | GigaChat zero-shot / few-shot |
| CoRAG | `/baselines/corag` | Iterative retrieval (RAGAR, ACL 2024) |
| STEEL | `/baselines/steel` | Multi-round + LLM relevance filtering |
| NLI | `/baselines/nli` | mDeBERTa entailment/contradiction scoring |
| GNN | `/baselines/gnn` | Graph Attention Network (needs training) |

All baselines accept the same request body as `/verify` plus method-specific fields (see `/docs`).

#### Training endpoints (supervised methods)

```bash
# Fine-tune RuBERT
curl -X POST http://localhost:8000/baselines/rubert/train \
  -F "file=@dataset.csv" -F "text_col=text" -F "label_col=label" -F "epochs=3"

# Train GNN
curl -X POST http://localhost:8000/baselines/gnn/train \
  -F "file=@dataset.csv" -F "epochs=20"
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
| Credibility | `/analysis/credibility` | Credibility-weighted vs. original verdict |
| Temporal | `/analysis/temporal` | Recency-weighted verdict using publication dates |
| Errors | `/analysis/errors` | Failure taxonomy on a batch of labeled results |

---

### `POST /compare` — Multi-method comparison

```json
{
  "text": "...",
  "methods": ["main", "single_rag", "llm_zeroshot", "nli"],
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
    {"method": "main",       "label": "ФЕЙКОВАЯ", "probability": 0.34},
    {"method": "single_rag", "label": "ФЕЙКОВАЯ", "probability": 0.41},
    {"method": "llm_zeroshot","label": "ФЕЙКОВАЯ","probability": 0.28},
    {"method": "nli",        "label": "ПРАВДИВАЯ","probability": 0.55}
  ],
  "agreement_fraction": 0.25,
  "consensus_label": "ФЕЙКОВАЯ",
  "disagreement": true
}
```

Available method names: `main`, `rubert`, `llm_zeroshot`, `corag`, `steel`, `nli`, `gnn`

> **Tip:** Avoid `corag`, `steel`, `gnn` in `/compare` unless you have time — each adds 30–120s.

---

## Model notes

| Model | When loaded | Approx. size |
|---|---|---|
| `DiTy/cross-encoder-russian-msmarco` | First `/verify` call | ~110 MB |
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | First `/analysis/heatmap` or `/baselines/nli` | ~280 MB |
| `deepset/xlm-roberta-large-squad2` | First `/analysis/spans` | ~1.1 GB |
| `paraphrase-multilingual-mpnet-base-v2` | First `/baselines/gnn` | ~280 MB |

All models are downloaded from HuggingFace Hub on first use and cached in the `hf-cache` Docker volume.

---

## Dataset format for training

CSV file with at minimum two columns:

| text | label |
|---|---|
| Центробанк повысил ставку до 21% | 1 |
| Земля плоская — учёные признали | 0 |

`label`: `1` = real, `0` = fake.
