# Fake News Detection API

REST API для определения ложных новостей на русском языке, построенный на FastAPI.

---

## Структура проекта

```
fakenews/
├── app/
│   ├── main.py                        # FastAPI приложение, подключение роутеров, /health
│   ├── core/
│   │   ├── config.py                  # Pydantic-settings (читает .env)
│   │   └── registry.py                # Ленивый singleton загрузчик моделей
│   ├── services/
│   │   ├── search_llm.py              # YandexSearchService, GigaChatService
│   │   └── models/
│   │       ├── main_model.py          # Основной пайплайн
│   │       ├── baselines.py           # 7 базовых методов
│   │       └── analysis.py            # 8 модулей интерпретации
│   ├── routers/
│   │   ├── verify.py                  # POST /verify
│   │   ├── baselines.py               # POST /baselines/{method}
│   │   ├── analysis.py                # POST /analysis/{module}
│   │   └── compare.py                 # POST /compare
│   └── schemas/
│       ├── requests.py                # Pydantic модели запросов
│       └── responses.py               # Pydantic модели ответов
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Быстрый старт

### 1. Настройка переменных окружения

```bash
cp .env.example .env
# Заполните:
#   GIGACHAT_CREDENTIALS, GIGACHAT_SCOPE
#   YANDEX_FOLDER_ID, YANDEX_AUTH
```

### 2. Запуск через Docker Compose

```bash
docker compose up --build
```

| Сервис               | URL | Описание                          |
|----------------------|---|-----------------------------------|
| Streamlit UI         | `http://localhost:8501` | Визуальный интерфейс              |
| FastAPI backend      | `http://localhost:8000` | REST API                          |
| Swagger документация | `http://localhost:8000/docs` | Автоматически сгенерированный API |

UI запускается только после успешного health-check API.

### 3. Локальный запуск (без докера)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API

### `GET /health`
Возвращает статус сервиса и загруженные модели.

```json
{
  "status": "ok",
  "device": "cpu",
  "cuda_available": false,
  "loaded_models": ["cross_encoder"]
}
```

---

### `POST /verify` – Основная модель

```json
{
  "text": "С 2025 года проезды для пенсионеров в Москве станут бесплатными",
  "num_queries": 5,
  "num_results": 5,
  "threshold": 0.5
}
```

**Ответ:**
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

| Модель | Эндпоинт            | Описание                                                       |
|--------|---------------------|----------------------------------------------------------------|
| RuBERT | `/baselines/rubert` | Fine-tuned классификатор, основанный исключительно на контенте |
| LLM    | `/baselines/llm`    | GigaChat zero-shot / few-shot                                  |
| CoRAG  | `/baselines/corag`  | Итеративный поиск (RAGAR, ACL 2024)                            |
| STEEL  | `/baselines/steel`  | Многоэтапная фильтрация + фильтрация по релевантности с LLM    |
| NLI    | `/baselines/nli`    | Оценка следствий/противоречий в mDeBERTa                       |
| GNN    | `/baselines/gnn`    | Graph Attention Network (требуется обучение)                   |

Все бейзлайны принимают то же тело запроса, что и `/verify`, а также поля, специфичные для метода (см. `/docs`).

#### Эндпоинты для обучения

```bash
# Дообучить RuBERT
curl -X POST http://localhost:8000/baselines/rubert/train \
  -F "file=@dataset.csv" -F "text_col=text" -F "label_col=label" -F "epochs=3"

# Обучить GNN
curl -X POST http://localhost:8000/baselines/gnn/train \
  -F "file=@dataset.csv" -F "epochs=20"
```

---

### `POST /analysis/{module}`

Все эндпоинты для анализа принимают:
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
Если `result` не указан, сначала автоматически запускается основная модель.

| Модуль      | Эндпоинт                | Что возвращает                                                   |
|-------------|-------------------------|------------------------------------------------------------------|
| Attribution | `/analysis/attribution` | Доказательства, ранжированные по степени влияния                 |
| Spans       | `/analysis/spans`       | Подпункты + противоречащий отрывок по каждому доказательству     |
| Heatmap     | `/analysis/heatmap`     | Матрица «предложение × предложение» NLI (следствие/противоречие) |
| Sensitivity | `/analysis/sensitivity` | Разброс P(true) по N независимым наборам запросов                |
| Credibility | `/analysis/credibility` | Вердикт, взвешенный с учетом достоверности, по сравнению с первоначальным вердиктом                        |
| Temporal    | `/analysis/temporal`    | Взвешенный по дате публикации вердикт                 |
| Errors      | `/analysis/errors`      | Таксономия ошибок на основе набора помеченных результатов                   |

---

### `POST /compare` — Сравнение различных методов

```json
{
  "text": "...",
  "methods": ["main", "single_rag", "llm_zeroshot", "nli"],
  "threshold": 0.5,
  "gold_label": 0
}
```

**Ответ:**
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

Допустимые названия методов: `main`, `rubert`, `llm_zeroshot`, `corag`, `steel`, `nli`, `gnn`

> **Совет:** Старайтесь не использовать `corag`, `steel` и `gnn` в `/compare`, если у вас нет времени — каждое из них добавляет 30–120 секунд.

---

## Примечания к модели

| Модель                                    | Когда загружается                               | Прибл. размер |
|-------------------------------------------|-------------------------------------------------|---------------|
| `DiTy/cross-encoder-russian-msmarco`      | Первый `/verify` вызов                          | ~110 МБ       |
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | Первый `/analysis/heatmap` или `/baselines/nli` | ~280 МБ       |
| `deepset/xlm-roberta-large-squad2`        | Первый `/analysis/spans`                        | ~1.1 ГБ       |
| `paraphrase-multilingual-mpnet-base-v2`   | Первый `/baselines/gnn`                         | ~280 МБ       |

При первом запуске все модели загружаются с HuggingFaceHub и сохраняются в кэше `hf-cache` Docker volume.

---

## Формат датасета для обучения

Файл CSV, содержащий как минимум два столбца:

| text | label |
|---|---|
| Центробанк повысил ставку до 21% | 1 |
| Земля плоская — учёные признали | 0 |

`label`: `1` = правда, `0` = ложь.
