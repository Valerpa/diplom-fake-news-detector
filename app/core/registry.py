import logging
from threading import Lock
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import (
    pipeline as hf_pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Порядок меток в mDeBERTa-v3-base-mnli-xnli:
# индекс 0 → contradiction, 1 → neutral, 2 → entailment
_NLI_ID2LABEL = {0: "contradiction", 1: "neutral", 2: "entailment"}


class _Registry:

    def __init__(self):
        self._lock = Lock()
        self._ce: CrossEncoder | None = None
        self._nli_tok = None
        self._nli_model = None
        self._qa: object | None = None
        self._sbert: SentenceTransformer | None = None

    @property
    def cross_encoder(self) -> CrossEncoder:
        if self._ce is None:
            with self._lock:
                if self._ce is None:
                    logger.info("Loading Cross-Encoder: %s",
                                settings.cross_encoder_model)
                    self._ce = CrossEncoder(
                        settings.cross_encoder_model, device=DEVICE
                    )
                    logger.info("Cross-Encoder loaded.")
        return self._ce

    def _ensure_nli(self):
        if self._nli_model is None:
            with self._lock:
                if self._nli_model is None:
                    logger.info("Loading NLI model: %s", settings.nli_model)
                    self._nli_tok = AutoTokenizer.from_pretrained(
                        settings.nli_model
                    )
                    self._nli_model = AutoModelForSequenceClassification \
                        .from_pretrained(settings.nli_model).to(DEVICE)
                    self._nli_model.eval()
                    logger.info("NLI model loaded.")

    def nli_scores(self, premise: str, hypothesis: str) -> dict:
        """Прямой NLI-инференс. Возвращает
        {"entailment": float, "neutral": float, "contradiction": float}."""
        self._ensure_nli()
        inputs = self._nli_tok(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=512
        ).to(DEVICE)
        with torch.no_grad():
            logits = self._nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return {
            _NLI_ID2LABEL[i]: float(probs[i])
            for i in range(len(probs))
        }

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
        if self._ce is not None:
            loaded.append("cross_encoder")
        if self._nli_model is not None:
            loaded.append("nli")
        if self._qa is not None:
            loaded.append("qa")
        if self._sbert is not None:
            loaded.append("sbert")
        return loaded


registry = _Registry()