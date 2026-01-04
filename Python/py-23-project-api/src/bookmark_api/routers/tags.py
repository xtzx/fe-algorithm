"""
标签路由
"""

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from bookmark_api.auth.dependencies import CurrentUser
from bookmark_api.db.models import Tag
from bookmark_api.db.session import get_db
from bookmark_api.schemas.common import Message
from bookmark_api.schemas.tag import TagCreate, TagResponse, TagUpdate

router = APIRouter(prefix="/tags", tags=["标签"])


@router.get("", response_model=List[TagResponse])
def list_tags(
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """获取标签列表"""
    tags = (
        db.query(Tag)
        .filter(Tag.user_id == current_user.id)
        .order_by(Tag.name)
        .all()
    )
    return tags


@router.post("", response_model=TagResponse, status_code=status.HTTP_201_CREATED)
def create_tag(
    tag_in: TagCreate,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """创建标签"""
    # 检查名称是否已存在
    existing = (
        db.query(Tag)
        .filter(
            Tag.user_id == current_user.id,
            Tag.name == tag_in.name,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tag name already exists",
        )

    tag = Tag(
        name=tag_in.name,
        color=tag_in.color,
        user_id=current_user.id,
    )
    db.add(tag)
    db.commit()
    db.refresh(tag)
    return tag


@router.get("/{tag_id}", response_model=TagResponse)
def get_tag(
    tag_id: int,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """获取标签详情"""
    tag = (
        db.query(Tag)
        .filter(
            Tag.id == tag_id,
            Tag.user_id == current_user.id,
        )
        .first()
    )
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found",
        )
    return tag


@router.put("/{tag_id}", response_model=TagResponse)
def update_tag(
    tag_id: int,
    tag_in: TagUpdate,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """更新标签"""
    tag = (
        db.query(Tag)
        .filter(
            Tag.id == tag_id,
            Tag.user_id == current_user.id,
        )
        .first()
    )
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found",
        )

    update_data = tag_in.model_dump(exclude_unset=True)

    # 检查名称是否已存在
    if "name" in update_data:
        existing = (
            db.query(Tag)
            .filter(
                Tag.user_id == current_user.id,
                Tag.name == update_data["name"],
                Tag.id != tag_id,
            )
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tag name already exists",
            )

    for field, value in update_data.items():
        setattr(tag, field, value)

    db.commit()
    db.refresh(tag)
    return tag


@router.delete("/{tag_id}", response_model=Message)
def delete_tag(
    tag_id: int,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """删除标签"""
    tag = (
        db.query(Tag)
        .filter(
            Tag.id == tag_id,
            Tag.user_id == current_user.id,
        )
        .first()
    )
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tag not found",
        )

    db.delete(tag)
    db.commit()
    return Message(message="Tag deleted")

