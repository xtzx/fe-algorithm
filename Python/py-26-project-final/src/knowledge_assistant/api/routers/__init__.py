"""
API 路由
"""

from knowledge_assistant.api.routers import auth, health, ingest, query

__all__ = ["auth", "health", "ingest", "query"]


