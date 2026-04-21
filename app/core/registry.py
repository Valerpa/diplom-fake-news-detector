import logging
from threading import Lock
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import pipeline as hf_pipeline
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _Registry:

    def __init__(self):
        self._lock = Lock()
        self._ce: CrossEncoder | None = None
        self._nli: object | None = None
        self._qa: object | None = None
        self._sbert: SentenceTransformer | None = None


    @property
    def cross_encoder(self) -> CrossEncoder:
        if self._ce is None:
            with self._lock:
                if self._ce is None:
                    logger.info("Loading Cross-Encoder: %s", settings.cross_encoder_model)
                    self._ce = CrossEncoder(settings.cross_encoder_model, device=DEVICE)
                    logger.info("Cross-Encoder loaded.")
        return self._ce

    @property
    def nli(self):
        if self._nli is None:
            with self._lock:
                if self._nli is None:
                    logger.info("Loading NLI model: %s", settings.nli_model)
                    self._nli = hf_pipeline(
                        "zero-shot-classification",
                        model=settings.nli_model,
                        device=0 if DEVICE == "cuda" else -1,
                    )
                    logger.info("NLI model loaded.")
        return self._nli

    @property
    def qa(self):
        if self._qa is None:
            with self._lock:
                if self._qa is None:
                    logger.info("Loading QA model: %s", settings.qa_model)
                    self._qa = hf_pipeline(
                        "question-answering",
                        model=settings.qa_model,
                        device=0 if DEVICE == "cuda" else -1,
                    )
                    logger.info("QA model loaded.")
        return self._qa

    @property
    def sbert(self) -> SentenceTransformer:
        if self._sbert is None:
            with self._lock:
                if self._sbert is None:
                    logger.info("Loading SBERT: %s", settings.sbert_model)
                    self._sbert = SentenceTransformer(
                        settings.sbert_model, device=DEVICE
                    )
                    logger.info("SBERT loaded.")
        return self._sbert

    def loaded_models(self) -> list[str]:
        loaded = []
        if self._ce is not None: loaded.append("cross_encoder")
        if self._nli is not None: loaded.append("nli")
        if self._qa is not None: loaded.append("qa")
        if self._sbert is not None: loaded.append("sbert")
        return loaded


registry = _Registry()
