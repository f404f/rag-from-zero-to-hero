import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API基础配置
    PROJECT_NAME: str = "RAG问答系统"
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    API_PATH: str = "/api/v1"

    # uvicorn配置
    UVICORN_WORKERS: int = 1
    UVICORN_RELOAD: bool = True

    # Milvus数据库配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    COLLECTION_NAME: str = "rag_documents"

    # Ollama配置
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:4b"

    # 检索配置
    DEFAULT_TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.6

    # 直接在 model_config 中根据环境变量设置 env_file
    model_config = SettingsConfigDict(
        env_file={
            "dev": ".env.dev",
            "prod": ".env.prod"
        }.get(os.getenv("ENV_STATE", "dev")),
        extra="ignore"
    )


settings = Settings()
