from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    question: str
    top_k: int = 3
    temperature: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "question": "什么是RAG系统？",
                "top_k": 3,
                "temperature": 0.0
            }
        }