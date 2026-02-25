from fastapi import APIRouter
from app.api.routes.chat import chat_router
from app.api.routes.api_doc import router as api_doc_router

master_router = APIRouter()

master_router.include_router(chat_router)
master_router.include_router(api_doc_router)
