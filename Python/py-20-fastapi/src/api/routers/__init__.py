"""
路由模块

包含所有 API 路由:
- auth: 认证相关
- users: 用户管理
- items: 商品管理
"""

from api.routers import auth, items, users

__all__ = ["auth", "users", "items"]

