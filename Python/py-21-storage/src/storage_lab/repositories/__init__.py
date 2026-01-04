"""
Repository 模块

提供:
- 基础 Repository
- 用户 Repository
- 商品 Repository
"""

from storage_lab.repositories.base import BaseRepository
from storage_lab.repositories.item_repo import ItemRepository
from storage_lab.repositories.user_repo import UserRepository

__all__ = [
    "BaseRepository",
    "UserRepository",
    "ItemRepository",
]


