"""
依赖注入
"""

from knowledge_assistant.api.dependencies.auth import (
    get_current_user,
    get_current_active_user,
    require_admin,
)
from knowledge_assistant.api.dependencies.services import (
    get_rag_service,
    get_llm_client,
    get_vector_index,
)

__all__ = [
    "get_current_user",
    "get_current_active_user",
    "require_admin",
    "get_rag_service",
    "get_llm_client",
    "get_vector_index",
]


