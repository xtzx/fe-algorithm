"""
书签 Schema
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class TagBrief(BaseModel):
    """标签简要信息"""
    id: int
    name: str
    color: Optional[str] = None

    class Config:
        from_attributes = True


class CategoryBrief(BaseModel):
    """分类简要信息"""
    id: int
    name: str
    icon: Optional[str] = None
    color: Optional[str] = None

    class Config:
        from_attributes = True


class BookmarkCreate(BaseModel):
    """创建书签"""
    url: HttpUrl
    title: str = Field(..., max_length=500)
    description: Optional[str] = Field(None, max_length=2000)
    category_id: Optional[int] = None
    tag_ids: Optional[List[int]] = None


class BookmarkUpdate(BaseModel):
    """更新书签"""
    url: Optional[HttpUrl] = None
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = Field(None, max_length=2000)
    category_id: Optional[int] = None
    tag_ids: Optional[List[int]] = None
    is_favorite: Optional[bool] = None
    is_archived: Optional[bool] = None


class BookmarkResponse(BaseModel):
    """书签响应"""
    id: int
    url: str
    title: str
    description: Optional[str]
    favicon: Optional[str]
    is_favorite: bool
    is_archived: bool
    click_count: int
    category: Optional[CategoryBrief]
    tags: List[TagBrief]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class BookmarkExport(BaseModel):
    """书签导出格式"""
    url: str
    title: str
    description: Optional[str]
    category: Optional[str]
    tags: List[str]
    is_favorite: bool
    created_at: str


class BookmarkImport(BaseModel):
    """书签导入格式"""
    url: HttpUrl
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None

