from fastapi import APIRouter

router = APIRouter(prefix="", tags=["API Docs"])


@router.get("/", summary="根路径")
async def root():
    """API根路径"""
    return {
        "message": "欢迎使用RAG问答系统API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@router.get("/health", summary="健康检查")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "service": "RAG API"}
