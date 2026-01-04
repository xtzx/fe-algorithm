"""
标签 Schema
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TagCreate(BaseModel):
    """创建标签"""
    name: str = Field(..., max_length=50)
    color: Optional[str] = Field(None, max_length=20)


class TagUpdate(BaseModel):
    """更新标签"""
    name: Optional[str] = Field(None, max_length=50)
    color: Optional[str] = Field(None, max_length=20)


class TagResponse(BaseModel):
    """标签响应"""
    id: int
    name: str
    color: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

