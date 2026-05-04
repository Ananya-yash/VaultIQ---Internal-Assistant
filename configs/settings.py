from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

BLOCKED_TOPICS: list[str] = [
    "weather",
    "stock price",
    "cryptocurrency",
    "sports score",
    "write a poem",
    "tell me a joke",
    "recipe",
    "personal advice",
    "news",
    "social media",
]

class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # --- LLM CONFIGURATION ---
    groq_api_key: str = Field(alias="groq_api_key")
    groq_model_name: str = Field(
        default="llama-3.1-8b-instant",
        alias="groq_model_name",
    )

    # --- VECTOR DATABASE ---
    # Using lowercase names here with case_sensitive=False in model_config
    qdrant_mode: str = Field(default="local", alias="QDRANT_MODE")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_local_path: str = Field(default="./qdrant_storage", alias="QDRANT_LOCAL_PATH")

    # --- SECURITY & AUTH (RBAC) ---
    jwt_secret: str = Field(alias="jwt_secret")
    algorithm: str = Field(default="HS256", alias="algorithm")
    access_token_expire_minutes: int = Field(default=60)
    LOG_LEVEL: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # Important: allows QDRANT_URL in .env to map to qdrant_url
        extra="ignore",
    )

settings = Settings()