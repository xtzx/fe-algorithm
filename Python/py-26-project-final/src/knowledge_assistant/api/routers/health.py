"""
健康检查路由
"""

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Response

from knowledge_assistant import __version__
from knowledge_assistant.api.dependencies.services import get_vector_index
from knowledge_assistant.config import get_settings

router = APIRouter()


@router.get("/healthz")
async def health_check() -> Dict[str, Any]:
    """
    健康检查
    
    用于 Kubernetes liveness probe
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/readyz")
async def readiness_check() -> Dict[str, Any]:
    """
    就绪检查
    
    用于 Kubernetes readiness probe
    """
    settings = get_settings()
    
    # 检查索引是否可用
    try:
        index = get_vector_index()
        index_ready = True
        index_count = len(index)
    except Exception:
        index_ready = False
        index_count = 0
    
    return {
        "status": "ready" if index_ready else "not_ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "index": {
                "ready": index_ready,
                "document_count": index_count,
            },
        },
    }


@router.get("/info")
async def app_info() -> Dict[str, Any]:
    """
    应用信息
    """
    settings = get_settings()
    
    return {
        "name": "知识库助手 API",
        "version": __version__,
        "environment": settings.app_env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


