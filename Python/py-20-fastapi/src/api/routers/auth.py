"""
认证路由

提供:
- 用户登录
- Token 刷新
- 用户注册
"""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from api.config import get_settings
from api.dependencies.auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    get_password_hash,
)
from api.schemas.auth import Token, TokenData
from api.schemas.user import User, UserCreate
from api.services.user_service import UserService

router = APIRouter()
settings = get_settings()


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    user_service: UserService = Depends(),
):
    """
    OAuth2 兼容的登录端点

    - **username**: 用户名
    - **password**: 密码

    返回 JWT access token
    """
    user = authenticate_user(user_service, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes},
        expires_delta=access_token_expires,
    )

    return Token(access_token=access_token, token_type="bearer")


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    user_service: UserService = Depends(),
):
    """
    用户注册

    - **username**: 用户名（唯一）
    - **email**: 邮箱
    - **password**: 密码（至少 8 个字符）
    """
    # 检查用户名是否已存在
    existing_user = user_service.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在",
        )

    # 创建用户
    hashed_password = get_password_hash(user_data.password)
    new_user = user_service.create_user(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
    )

    return new_user


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """获取当前登录用户信息"""
    return current_user


@router.get("/me/token-info", response_model=TokenData)
async def read_token_info(current_user: User = Depends(get_current_active_user)):
    """获取当前 Token 信息"""
    return TokenData(username=current_user.username, scopes=current_user.scopes)

