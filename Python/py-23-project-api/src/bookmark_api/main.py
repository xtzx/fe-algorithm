"""
FastAPI 应用入口
"""

import logging
import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bookmark_api.config import get_settings
from bookmark_api.db.session import init_db
from bookmark_api.routers import (
    auth_router,
    bookmarks_router,
    categories_router,
    tags_router,
    users_router,
)

settings = get_settings()


# ==================== 日志配置 ====================


def configure_logging():
    """配置结构化日志"""
    processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.contextvars.merge_contextvars,
        structlog.processors.format_exc_info,
    ]

    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


configure_logging()
logger = structlog.get_logger()


# ==================== 应用生命周期 ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        env=settings.app_env,
    )

    # 初始化数据库
    init_db()

    yield

    logger.info("application_shutting_down")


# ==================== 创建应用 ====================


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="书签管理 API - 综合项目 4",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)


# ==================== 中间件 ====================


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求日志
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求日志"""
    import time
    import uuid

    request_id = str(uuid.uuid4())[:8]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    start_time = time.perf_counter()

    logger.info(
        "request_started",
        method=request.method,
        path=request.url.path,
    )

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )

    response.headers["X-Request-ID"] = request_id
    return response


# ==================== 异常处理 ====================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.exception("unhandled_exception", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ==================== 路由注册 ====================


API_V1_PREFIX = "/api/v1"

app.include_router(auth_router, prefix=API_V1_PREFIX)
app.include_router(users_router, prefix=API_V1_PREFIX)
app.include_router(bookmarks_router, prefix=API_V1_PREFIX)
app.include_router(categories_router, prefix=API_V1_PREFIX)
app.include_router(tags_router, prefix=API_V1_PREFIX)


# ==================== 健康检查 ====================


@app.get("/health", tags=["健康检查"])
async def health():
    """存活检查"""
    return {"status": "healthy", "version": settings.app_version}


@app.get("/health/ready", tags=["健康检查"])
async def readiness():
    """就绪检查"""
    checks = {"database": "ok"}

    # 检查 Redis
    from bookmark_api.cache import get_cache_client

    cache = get_cache_client()
    checks["cache"] = "ok" if cache.is_available else "unavailable"

    return {"status": "ready", "checks": checks}


# ==================== 根路由 ====================


@app.get("/", tags=["根"])
async def root():
    """根路由"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs" if settings.debug else None,
    }


# ==================== 启动 ====================


def run():
    """启动服务器"""
    import uvicorn

    uvicorn.run(
        "bookmark_api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()

