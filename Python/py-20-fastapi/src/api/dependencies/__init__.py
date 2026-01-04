"""
依赖注入模块

提供:
- 数据库连接
- 认证依赖
- 其他公共依赖
"""

from api.dependencies.auth import (
    get_current_active_user,
    get_current_user,
    require_admin,
)
from api.dependencies.database import get_db

__all__ = [
    "get_db",
    "get_current_user",
    "get_current_active_user",
    "require_admin",
]

