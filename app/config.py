from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str = ""
    groq_api_keys: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_provider: str = "groq"
    # backend_base_url: str = "http://host.docker.internal:8087/elif"
    backend_base_url: str = "http://localhost:8087/elif"
    backend_community_prefix: str = "/api/community"

    http_timeout_seconds: int = 30
    max_agent_actions: int = 6


settings = Settings()
