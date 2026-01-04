"""
FastAPI 应用入口

功能:
- 应用配置
- 路由注册
- 中间件配置
- 异常处理器注册
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.exceptions import register_exception_handlers
from api.middleware.logging import RequestLoggingMiddleware
from api.middleware.trace import TraceMiddleware
from api.routers import auth, items, users

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    yield
    # 关闭时执行
    print(f"👋 Shutting down {settings.app_name}")


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="FastAPI 服务学习项目 - 生产级 API 示例",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 配置 CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # 自定义中间件（按顺序执行，后添加的先执行）
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(TraceMiddleware)

    # 注册异常处理器
    register_exception_handlers(app)

    # 注册路由
    app.include_router(auth.router, prefix=f"{settings.api_prefix}/auth", tags=["认证"])
    app.include_router(users.router, prefix=f"{settings.api_prefix}/users", tags=["用户"])
    app.include_router(items.router, prefix=f"{settings.api_prefix}/items", tags=["商品"])

    return app


# 创建应用实例
app = create_app()


# 根路由
@app.get("/", tags=["健康检查"])
async def root():
    """根路由 - 健康检查"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
    }


@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "app": settings.app_name}

