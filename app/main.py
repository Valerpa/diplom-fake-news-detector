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


@app.get("/", tags=["health"], include_in_schema=False)
def root():
    return {"message": "Fake News Detection API. Visit /docs for full documentation."}
