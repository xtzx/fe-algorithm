"""
Repository 模块
"""

from bookmark_api.db.repositories.base import BaseRepository
from bookmark_api.db.repositories.bookmark_repo import BookmarkRepository
from bookmark_api.db.repositories.user_repo import UserRepository

__all__ = [
    "BaseRepository",
    "UserRepository",
    "BookmarkRepository",
]

