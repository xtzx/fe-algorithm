"""
Pydantic Schemas
"""

from bookmark_api.schemas.bookmark import (
    BookmarkCreate,
    BookmarkExport,
    BookmarkResponse,
    BookmarkUpdate,
)
from bookmark_api.schemas.category import CategoryCreate, CategoryResponse, CategoryUpdate
from bookmark_api.schemas.common import Message, PaginatedResponse
from bookmark_api.schemas.tag import TagCreate, TagResponse
from bookmark_api.schemas.user import Token, TokenRefresh, UserCreate, UserLogin, UserResponse

__all__ = [
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "Token",
    "TokenRefresh",
    "BookmarkCreate",
    "BookmarkUpdate",
    "BookmarkResponse",
    "BookmarkExport",
    "CategoryCreate",
    "CategoryUpdate",
    "CategoryResponse",
    "TagCreate",
    "TagResponse",
    "Message",
    "PaginatedResponse",
]

