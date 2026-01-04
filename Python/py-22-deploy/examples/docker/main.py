"""
生产级 FastAPI 应用示例

演示:
- 结构化日志
- 健康检查
- 优雅停机
- 配置管理
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings


# ==================== 配置 ====================


class Settings(BaseSettings):
    """应用配置"""
    app_env: str = "development"
    app_debug: bool = False
    app_port: int = 8000
    
    database_url: str = "sqlite:///./app.db"
    redis_url: str = "redis://localhost:6379/0"
    
    secret_key: str = "change-me"
    
    log_level: str = "INFO"
    log_format: str = "json"  # json or console
    
    class Config:
        env_file = ".env"


settings = Settings()


# ==================== 日志配置 ====================


def configure_logging():
    """配置结构化日志"""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # JSON 或控制台格式
            structlog.processors.JSONRenderer()
            if settings.log_format == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
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
    """应用生命周期管理"""
    # 启动时
    logger.info(
        "application_starting",
        env=settings.app_env,
        port=settings.app_port,
    )
    
    # 初始化资源（数据库、缓存等）
    # await init_database()
    # await init_cache()
    
    yield
    
    # 关闭时
    logger.info("application_shutting_down")
    
    # 清理资源
    # await close_database()
    # await close_cache()


# ==================== 创建应用 ====================


app = FastAPI(
    title="Production API",
    version="1.0.0",
    docs_url="/docs" if settings.app_debug else None,
    redoc_url="/redoc" if settings.app_debug else None,
    lifespan=lifespan,
)


# ==================== 中间件 ====================


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_debug else ["https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求日志"""
    import time
    import uuid
    
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    
    # 绑定请求上下文
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    
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


# ==================== 健康检查 ====================


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"


class ReadinessResponse(BaseModel):
    status: str
    checks: dict


@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
async def health():
    """
    存活检查 (Liveness)
    
    用于 Kubernetes liveness probe
    只检查应用是否在运行
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/health/ready", response_model=ReadinessResponse, tags=["健康检查"])
async def readiness():
    """
    就绪检查 (Readiness)
    
    用于 Kubernetes readiness probe
    检查应用是否准备好接收流量
    """
    checks = {}
    all_healthy = True
    
    # 检查数据库
    try:
        # await check_database()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"
        all_healthy = False
    
    # 检查 Redis
    try:
        # await check_redis()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        all_healthy = False
    
    if not all_healthy:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks": checks},
        )
    
    return ReadinessResponse(status="ready", checks=checks)


# ==================== API 路由 ====================


@app.get("/", tags=["根"])
async def root():
    """根路由"""
    return {
        "message": "Welcome to Production API",
        "env": settings.app_env,
    }


@app.get("/api/info", tags=["信息"])
async def info():
    """应用信息"""
    return {
        "app": "Production API",
        "version": "1.0.0",
        "env": settings.app_env,
    }


# ==================== 优雅停机 ====================


def handle_shutdown(signum, frame):
    """处理关闭信号"""
    logger.info("shutdown_signal_received", signal=signum)
    sys.exit(0)


# 注册信号处理器
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


# ==================== 主入口 ====================


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower(),
    )


