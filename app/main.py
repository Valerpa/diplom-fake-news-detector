import logging
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.core.registry import registry, DEVICE
from app.routers import verify, baselines, analysis, compare

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="Fake News Detection API",
    description=(
        "REST API for Russian-language fake news detection.\n\n"
        "**Endpoints:**\n"
        "- `POST /verify` — main model (LLM queries → Yandex Search → Cross-Encoder)\n"
        "- `POST /baselines/{method}` — 7 alternative methods\n"
        "- `POST /analysis/{module}` — 8 explainability modules\n"
        "- `POST /compare` — run multiple methods and compare results\n"
        "- `GET  /health` — service health and loaded models\n"
    ),
    version="1.0.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(verify.router)
app.include_router(baselines.router)
app.include_router(analysis.router)
app.include_router(compare.router)


@app.get("/health", tags=["health"], summary="Service health and loaded model list")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": registry.loaded_models(),
        "threshold": settings.default_threshold,
        "settings": {
            "cross_encoder_model": settings.cross_encoder_model,
            "nli_model": settings.nli_model,
            "qa_model": settings.qa_model,
        },
    }


VALID_MODELS = {"cross_encoder", "nli", "qa", "sbert"}


@app.post("/models/preload", tags=["health"],
          summary="Pre-load one or more models into memory")
def preload_models(models: list[str]):
    loaded = []
    errors = {}
    for name in models:
        if name not in VALID_MODELS:
            errors[name] = f"Unknown model. Valid: {VALID_MODELS}"
            continue
        try:
            getattr(registry, name)
            loaded.append(name)
        except Exception as e:
            errors[name] = str(e)
    return {
        "loaded": loaded,
        "errors": errors,
        "all_loaded": registry.loaded_models(),
    }


@app.get("/", tags=["health"], include_in_schema=False)
def root():
    return {"message": "Fake News Detection API. Visit /docs for full documentation."}
