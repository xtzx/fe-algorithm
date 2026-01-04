"""
认证路由
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from bookmark_api.auth.dependencies import CurrentUser
from bookmark_api.auth.jwt import create_access_token, create_refresh_token, verify_token_type
from bookmark_api.auth.password import get_password_hash, verify_password
from bookmark_api.db.repositories.user_repo import UserRepository
from bookmark_api.db.session import get_db
from bookmark_api.schemas.common import Message
from bookmark_api.schemas.user import Token, TokenRefresh, UserCreate, UserResponse

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(
    user_in: UserCreate,
    db: Annotated[Session, Depends(get_db)],
):
    """用户注册"""
    repo = UserRepository(db)

    # 检查用户名和邮箱是否已存在
    existing = repo.get_by_username_or_email(user_in.username, user_in.email)
    if existing:
        if existing.username == user_in.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # 创建用户
    hashed_password = get_password_hash(user_in.password)
    user = repo.create_user(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hashed_password,
    )

    return user


@router.post("/login", response_model=Token)
def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
):
    """用户登录"""
    repo = UserRepository(db)

    # 查找用户
    user = repo.get_by_username(form_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 验证密码
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 检查用户状态
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is inactive",
        )

    # 创建令牌
    access_token = create_access_token(data={"sub": user.id})
    refresh_token = create_refresh_token(data={"sub": user.id})

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/refresh", response_model=Token)
def refresh_token(
    token_in: TokenRefresh,
    db: Annotated[Session, Depends(get_db)],
):
    """刷新令牌"""
    # 验证刷新令牌
    payload = verify_token_type(token_in.refresh_token, "refresh")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 检查用户
    user = db.get(UserRepository(db).model, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 创建新令牌
    access_token = create_access_token(data={"sub": user.id})
    refresh_token = create_refresh_token(data={"sub": user.id})

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/logout", response_model=Message)
def logout(current_user: CurrentUser):
    """用户登出（客户端应删除令牌）"""
    return Message(message="Successfully logged out")

