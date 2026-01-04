"""
分类路由
"""

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from bookmark_api.auth.dependencies import CurrentUser
from bookmark_api.db.models import Category
from bookmark_api.db.session import get_db
from bookmark_api.schemas.category import CategoryCreate, CategoryResponse, CategoryUpdate
from bookmark_api.schemas.common import Message

router = APIRouter(prefix="/categories", tags=["分类"])


@router.get("", response_model=List[CategoryResponse])
def list_categories(
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """获取分类列表"""
    categories = (
        db.query(Category)
        .filter(Category.user_id == current_user.id)
        .order_by(Category.name)
        .all()
    )
    return categories


@router.post("", response_model=CategoryResponse, status_code=status.HTTP_201_CREATED)
def create_category(
    category_in: CategoryCreate,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """创建分类"""
    # 检查名称是否已存在
    existing = (
        db.query(Category)
        .filter(
            Category.user_id == current_user.id,
            Category.name == category_in.name,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Category name already exists",
        )

    # 检查父分类是否存在
    if category_in.parent_id:
        parent = (
            db.query(Category)
            .filter(
                Category.id == category_in.parent_id,
                Category.user_id == current_user.id,
            )
            .first()
        )
        if not parent:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parent category not found",
            )

    category = Category(
        name=category_in.name,
        description=category_in.description,
        icon=category_in.icon,
        color=category_in.color,
        parent_id=category_in.parent_id,
        user_id=current_user.id,
    )
    db.add(category)
    db.commit()
    db.refresh(category)
    return category


@router.get("/{category_id}", response_model=CategoryResponse)
def get_category(
    category_id: int,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """获取分类详情"""
    category = (
        db.query(Category)
        .filter(
            Category.id == category_id,
            Category.user_id == current_user.id,
        )
        .first()
    )
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found",
        )
    return category


@router.put("/{category_id}", response_model=CategoryResponse)
def update_category(
    category_id: int,
    category_in: CategoryUpdate,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """更新分类"""
    category = (
        db.query(Category)
        .filter(
            Category.id == category_id,
            Category.user_id == current_user.id,
        )
        .first()
    )
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found",
        )

    update_data = category_in.model_dump(exclude_unset=True)

    # 检查名称是否已存在
    if "name" in update_data:
        existing = (
            db.query(Category)
            .filter(
                Category.user_id == current_user.id,
                Category.name == update_data["name"],
                Category.id != category_id,
            )
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Category name already exists",
            )

    for field, value in update_data.items():
        setattr(category, field, value)

    db.commit()
    db.refresh(category)
    return category


@router.delete("/{category_id}", response_model=Message)
def delete_category(
    category_id: int,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """删除分类"""
    category = (
        db.query(Category)
        .filter(
            Category.id == category_id,
            Category.user_id == current_user.id,
        )
        .first()
    )
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found",
        )

    db.delete(category)
    db.commit()
    return Message(message="Category deleted")

