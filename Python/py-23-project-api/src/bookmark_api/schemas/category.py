"""
分类 Schema
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class CategoryCreate(BaseModel):
    """创建分类"""
    name: str = Field(..., max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    icon: Optional[str] = Field(None, max_length=50)
    color: Optional[str] = Field(None, max_length=20)
    parent_id: Optional[int] = None


class CategoryUpdate(BaseModel):
    """更新分类"""
    name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    icon: Optional[str] = Field(None, max_length=50)
    color: Optional[str] = Field(None, max_length=20)
    parent_id: Optional[int] = None


class CategoryResponse(BaseModel):
    """分类响应"""
    id: int
    name: str
    description: Optional[str]
    icon: Optional[str]
    color: Optional[str]
    parent_id: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class CategoryTreeResponse(BaseModel):
    """分类树响应"""
    id: int
    name: str
    description: Optional[str]
    icon: Optional[str]
    color: Optional[str]
    children: List["CategoryTreeResponse"] = []

    class Config:
        from_attributes = True

