from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.master_route import master_router
from app.core.config import settings

# 创建FastAPI应用实例
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="基于RAG的智能问答系统API",
    openapi_url=f"{settings.API_PATH}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件（允许跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(master_router, prefix=settings.API_PATH)
