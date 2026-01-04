"""
认证相关的 Pydantic 模型

职责:
- Token 相关数据结构
"""

from pydantic import BaseModel, Field


class Token(BaseModel):
    """Token 响应模型"""

    access_token: str = Field(..., description="访问令牌")
    token_type: str = Field("bearer", description="令牌类型")


class TokenData(BaseModel):
    """Token 解析后的数据"""

    username: str | None = Field(None, description="用户名")
    scopes: list[str] = Field(default_factory=list, description="权限范围")


class LoginRequest(BaseModel):
    """登录请求模型（用于 JSON 登录）"""

    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    password: str = Field(..., min_length=1, description="密码")

