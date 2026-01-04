"""
认证路由
"""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from knowledge_assistant.api.dependencies.auth import (
    authenticate_user,
    create_access_token,
    create_user,
    get_current_active_user,
)
from knowledge_assistant.api.schemas.auth import (
    Token,
    UserCreate,
    UserResponse,
)
from knowledge_assistant.config import get_settings

router = APIRouter()
settings = get_settings()


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    """
    登录获取 Token
    
    使用 OAuth2 密码模式
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.jwt_access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user["username"], "scopes": user["scopes"]},
        expires_delta=access_token_expires,
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate) -> UserResponse:
    """
    用户注册
    """
    try:
        user = create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
        )
        return UserResponse(
            id=user["id"],
            username=user["username"],
            email=user["email"],
            is_active=user["is_active"],
            scopes=user["scopes"],
            created_at=user["created_at"],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: UserResponse = Depends(get_current_active_user),
) -> UserResponse:
    """
    获取当前用户信息
    """
    return current_user


