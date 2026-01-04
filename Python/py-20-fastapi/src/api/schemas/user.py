"""
用户相关的 Pydantic 模型

职责:
- 定义用户数据结构
- 请求验证
- 响应序列化
"""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """用户基础模型"""

    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱地址")


class UserCreate(UserBase):
    """创建用户请求模型"""

    password: str = Field(..., min_length=8, max_length=100, description="密码")


class UserUpdate(BaseModel):
    """更新用户请求模型（所有字段可选）"""

    email: EmailStr | None = Field(None, description="邮箱地址")
    full_name: str | None = Field(None, max_length=100, description="全名")
    is_active: bool | None = Field(None, description="是否激活")


class User(UserBase):
    """用户响应模型"""

    id: int = Field(..., description="用户 ID")
    full_name: str | None = Field(None, description="全名")
    is_active: bool = Field(True, description="是否激活")
    scopes: list[str] = Field(default_factory=list, description="权限范围")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime | None = Field(None, description="更新时间")

    model_config = {"from_attributes": True}


class UserInDB(User):
    """数据库中的用户模型（包含密码哈希）"""

    hashed_password: str = Field(..., description="密码哈希")

