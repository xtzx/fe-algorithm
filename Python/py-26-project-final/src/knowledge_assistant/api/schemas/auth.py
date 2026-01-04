"""
认证相关模型
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """用户注册"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)


class UserLogin(BaseModel):
    """用户登录"""
    username: str
    password: str


class UserResponse(BaseModel):
    """用户信息响应"""
    id: str
    username: str
    email: str
    is_active: bool = True
    scopes: List[str] = []
    created_at: datetime


class Token(BaseModel):
    """JWT Token 响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token 数据"""
    username: Optional[str] = None
    scopes: List[str] = []


