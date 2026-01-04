"""
FastAPI 应用

创建和配置 FastAPI 应用
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from knowledge_assistant import __version__
from knowledge_assistant.config import get_settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期"""
    settings = get_settings()
    
    # 启动时
    logger.info(
        "application_starting",
        version=__version__,
        env=settings.app_env,
    )
    
    # 确保目录存在
    settings.ensure_directories()
    
    yield
    
    # 关闭时
    logger.info("application_shutdown")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    settings = get_settings()
    
    app = FastAPI(
        title="知识库助手 API",
        description="企业知识库问答助手 - RAG 检索增强生成",
        version=__version__,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    from knowledge_assistant.api.routers import auth, ingest, query, health
    
    app.include_router(health.router, tags=["健康检查"])
    app.include_router(auth.router, prefix=f"{settings.api_prefix}/auth", tags=["认证"])
    app.include_router(ingest.router, prefix=f"{settings.api_prefix}/ingest", tags=["文档摄取"])
    app.include_router(query.router, prefix=f"{settings.api_prefix}/query", tags=["问答查询"])
    
    return app


