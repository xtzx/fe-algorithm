"""
用户路由
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from bookmark_api.auth.dependencies import CurrentUser
from bookmark_api.db.repositories.user_repo import UserRepository
from bookmark_api.db.session import get_db
from bookmark_api.schemas.common import Message
from bookmark_api.schemas.user import UserResponse, UserUpdate

router = APIRouter(prefix="/users", tags=["用户"])


@router.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: CurrentUser):
    """获取当前用户信息"""
    return current_user


@router.put("/me", response_model=UserResponse)
def update_current_user(
    user_in: UserUpdate,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """更新当前用户信息"""
    repo = UserRepository(db)

    update_data = user_in.model_dump(exclude_unset=True)

    # 检查用户名是否已存在
    if "username" in update_data:
        existing = repo.get_by_username(update_data["username"])
        if existing and existing.id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken",
            )

    # 检查邮箱是否已存在
    if "email" in update_data:
        existing = repo.get_by_email(update_data["email"])
        if existing and existing.id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already taken",
            )

    # 更新用户
    updated_user = repo.update(current_user, update_data)
    return updated_user


@router.delete("/me", response_model=Message)
def delete_current_user(
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """删除当前用户（停用）"""
    repo = UserRepository(db)
    repo.deactivate(current_user.id)
    return Message(message="User account deactivated")

