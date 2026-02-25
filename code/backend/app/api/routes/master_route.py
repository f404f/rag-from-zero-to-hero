from fastapi import APIRouter
from app.api.routes.chat import router

master_router = APIRouter()

master_router.include_router(router)
