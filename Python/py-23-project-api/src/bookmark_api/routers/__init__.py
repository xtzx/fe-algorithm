"""
API 路由
"""

from bookmark_api.routers.auth import router as auth_router
from bookmark_api.routers.bookmarks import router as bookmarks_router
from bookmark_api.routers.categories import router as categories_router
from bookmark_api.routers.tags import router as tags_router
from bookmark_api.routers.users import router as users_router

__all__ = [
    "auth_router",
    "users_router",
    "bookmarks_router",
    "categories_router",
    "tags_router",
]

