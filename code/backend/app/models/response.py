from pydantic import BaseModel
from typing import List


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    similarity_scores: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "RAG是检索增强生成框架...",
                "sources": ["doc1.txt", "doc2.txt"],
                "similarity_scores": [0.85, 0.78]
            }
        }