# Fake News Detection API

REST API для определения ложных новостей на русском языке, построенный на FastAPI.

---

## Структура проекта

```
fake-news-detector/
├── app/
│   ├── main.py                        # FastAPI приложение, подключение роутеров, /health
│   ├── core/
│   │   ├── config.py                  # Pydantic-settings (читает .env)
│   │   └── registry.py                # Ленивый singleton загрузчик моделей
│   ├── services/
│   │   ├── search_llm.py              # YandexSearchService, GigaChatService
│   │   └── models/
│   │       ├── main_model.py          # Основной пайплайн верификации
│   │       ├── baselines.py           # 4 базовых метода
│   │       ├── analysis.py            # Модули интерпретации
│   │       └── utils.py               # Общие утилиты
│   ├── routers/
│   │   ├── verify.py                  # POST /verify, /verify/queries, /verify/run
│   │   ├── baselines.py               # POST /baselines/{method}
│   │   ├── analysis.py                # POST /analysis/{module}
│   │   └── compare.py                 # POST /compare
│   └── schemas/
│       ├── requests.py                # Pydantic модели запросов
│       └── responses.py               # Pydantic модели ответов
├── ui/
│   └── streamlit_app.py               # Streamlit UI
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

### `POST /verify` – Основная модель (полный пайплайн)

```json
{
  "text": "С 2025 года проезды для пенсионеров в Москве станут бесплатными",
  "num_queries": 5,
  "num_results": 5
}
```

**Ответ:**
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

Метки: `ПРАВДИВАЯ`, `ЛОЖНАЯ`, `НЕДОСТАТОЧНО ДАННЫХ`. Порог по умолчанию: `0.6`.

#### Двухэтапная верификация

Можно разделить пайплайн на два шага, чтобы просмотреть или отредактировать сгенерированные запросы:

1. **`POST /verify/queries`** — генерация поисковых запросов без запуска поиска
2. **`POST /verify/run`** — поиск и оценка с переданным списком запросов

---

### `POST /baselines/{method}`

| Метод  | Эндпоинт            | Описание                                                       |
|--------|---------------------|----------------------------------------------------------------|
| RuBERT | `/baselines/rubert` | Fine-tuned классификатор, основанный исключительно на контенте |
| LLM    | `/baselines/llm`    | GigaChat zero-shot / few-shot                                  |
| CoRAG  | `/baselines/corag`  | Итеративный поиск (RAGAR, ACL 2024)                            |
| NLI    | `/baselines/nli`    | Оценка следствий/противоречий в mDeBERTa                       |

Все бейзлайны принимают то же тело запроса, что и `/verify`, а также поля, специфичные для метода (см. `/docs`).

#### Эндпоинт для обучения

```bash
# Дообучить RuBERT
curl -X POST http://localhost:8000/baselines/rubert/train \
  -F "file=@dataset.csv" -F "text_col=text" -F "label_col=label" -F "epochs=3"
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
| Signs       | `/analysis/signs`       | Обнаружение типичных признаков фейковой новости в тексте         |

---

### `POST /compare` — Сравнение различных методов

```json
{
  "text": "...",
  "methods": ["main", "llm_zeroshot", "nli"],
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
    {"method": "main",        "label": "ЛОЖНАЯ",    "probability": 0.34},
    {"method": "llm_zeroshot","label": "ЛОЖНАЯ",    "probability": 0.28},
    {"method": "nli",         "label": "ПРАВДИВАЯ", "probability": 0.55}
  ],
  "agreement_fraction": 0.25,
  "consensus_label": "ЛОЖНАЯ",
  "disagreement": true
}
```

Допустимые названия методов: `main`, `rubert`, `llm_zeroshot`, `corag`, `nli`

> **Совет:** `corag` в `/compare` добавляет 30–120 секунд из-за итеративного поиска.

---

## Примечания к модели

| Модель                                    | Когда загружается                               | Прибл. размер |
|-------------------------------------------|-------------------------------------------------|---------------|
| `DiTy/cross-encoder-russian-msmarco`      | Первый `/verify` вызов                          | ~110 МБ       |
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | Первый `/analysis/heatmap` или `/baselines/nli` | ~280 МБ       |
| `deepset/xlm-roberta-large-squad2`        | Первый `/analysis/spans`                        | ~1.1 ГБ       |

Модели можно предзагрузить через `POST /models/preload` со списком имён (`cross_encoder`, `nli`, `qa`).

При первом запуске все модели загружаются с HuggingFace Hub и сохраняются в кэше `hf-cache` Docker volume.

---

## Формат датасета для обучения

Файл CSV, содержащий как минимум два столбца:

| text | label |
|---|---|
| Центробанк повысил ставку до 21% | 1 |
| Земля плоская — учёные признали | 0 |

`label`: `1` = правда, `0` = ложь.
