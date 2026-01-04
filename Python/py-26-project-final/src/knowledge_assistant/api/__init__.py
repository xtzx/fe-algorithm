"""
API 模块

提供:
- FastAPI 路由
- 认证与授权
- 中间件
- 依赖注入
"""

from knowledge_assistant.api.app import create_app

__all__ = ["create_app"]


