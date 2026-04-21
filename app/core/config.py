from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    gigachat_credentials: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_model: str = "GigaChat-2-Max"

    yandex_folder_id: str = ""
    yandex_auth: str = ""

    cross_encoder_model: str = "DiTy/cross-encoder-russian-msmarco"
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    qa_model: str = "deepset/xlm-roberta-large-squad2"
    rubert_model: str = "DeepPavlov/rubert-base-cased"
    sbert_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    rubert_save_path: str = "./rubert_fakenews"

    default_num_queries: int = 5
    default_threshold: float = 0.5
    default_num_results: int = 5

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"


@lru_cache
def get_settings() -> Settings:
    return Settings()
