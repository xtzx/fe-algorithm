"""
用户路由

提供用户 CRUD 操作:
- 获取用户列表
- 获取单个用户
- 更新用户
- 删除用户
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies.auth import get_current_active_user, require_admin
from api.schemas.user import User, UserUpdate
from api.services.user_service import UserService

router = APIRouter()


@router.get("/", response_model=list[User])
async def list_users(
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(10, ge=1, le=100, description="返回的最大记录数"),
    user_service: UserService = Depends(),
    _current_user: User = Depends(get_current_active_user),
):
    """
    获取用户列表（分页）

    - **skip**: 跳过的记录数（用于分页）
    - **limit**: 返回的最大记录数
    """
    return user_service.get_users(skip=skip, limit=limit)


@router.get("/{user_id}", response_model=User)
async def get_user(
    user_id: int,
    user_service: UserService = Depends(),
    _current_user: User = Depends(get_current_active_user),
):
    """
    根据 ID 获取用户

    - **user_id**: 用户 ID
    """
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户 {user_id} 不存在",
        )
    return user


@router.put("/{user_id}", response_model=User)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    user_service: UserService = Depends(),
    current_user: User = Depends(get_current_active_user),
):
    """
    更新用户信息

    - **user_id**: 用户 ID
    - **user_data**: 要更新的字段

    普通用户只能更新自己，管理员可以更新任何用户
    """
    # 权限检查：只能更新自己，除非是管理员
    if current_user.id != user_id and "admin" not in current_user.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限更新其他用户",
        )

    updated_user = user_service.update_user(user_id, user_data)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户 {user_id} 不存在",
        )
    return updated_user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    user_service: UserService = Depends(),
    _current_user: User = Depends(require_admin),  # 需要管理员权限
):
    """
    删除用户（需要管理员权限）

    - **user_id**: 要删除的用户 ID
    """
    success = user_service.delete_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户 {user_id} 不存在",
        )
    return None

