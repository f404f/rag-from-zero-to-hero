import uvicorn

if __name__ == "__main__":
    from app.core.config import settings

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.UVICORN_RELOAD,  # 开发模式下自动重载
        workers=settings.UVICORN_WORKERS,
    )
