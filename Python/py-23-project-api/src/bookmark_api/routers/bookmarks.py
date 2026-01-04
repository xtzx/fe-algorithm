"""
书签路由
"""

import json
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from bookmark_api.auth.dependencies import CurrentUser
from bookmark_api.db.models import Tag
from bookmark_api.db.repositories.bookmark_repo import BookmarkRepository
from bookmark_api.db.session import get_db
from bookmark_api.schemas.bookmark import (
    BookmarkCreate,
    BookmarkExport,
    BookmarkImport,
    BookmarkResponse,
    BookmarkUpdate,
)
from bookmark_api.schemas.common import Message, PaginatedResponse

router = APIRouter(prefix="/bookmarks", tags=["书签"])


@router.get("", response_model=PaginatedResponse[BookmarkResponse])
def list_bookmarks(
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category_id: Optional[int] = None,
    is_favorite: Optional[bool] = None,
    is_archived: Optional[bool] = None,
):
    """获取书签列表（分页）"""
    repo = BookmarkRepository(db)
    skip = (page - 1) * page_size

    bookmarks = repo.get_by_user(
        user_id=current_user.id,
        skip=skip,
        limit=page_size,
        category_id=category_id,
        is_favorite=is_favorite,
        is_archived=is_archived,
    )

    total = repo.count_by_user(
        user_id=current_user.id,
        category_id=category_id,
        is_favorite=is_favorite,
        is_archived=is_archived,
    )

    return PaginatedResponse.create(
        items=bookmarks,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=BookmarkResponse, status_code=status.HTTP_201_CREATED)
def create_bookmark(
    bookmark_in: BookmarkCreate,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """创建书签"""
    repo = BookmarkRepository(db)

    bookmark = repo.create_bookmark(
        user_id=current_user.id,
        url=str(bookmark_in.url),
        title=bookmark_in.title,
        description=bookmark_in.description,
        category_id=bookmark_in.category_id,
        tag_ids=bookmark_in.tag_ids,
    )

    # 刷新以获取关联数据
    db.refresh(bookmark)
    return bookmark


@router.get("/search", response_model=List[BookmarkResponse])
def search_bookmarks(
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    q: str = Query(..., min_length=1, max_length=100),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """搜索书签"""
    repo = BookmarkRepository(db)
    skip = (page - 1) * page_size

    bookmarks = repo.search(
        user_id=current_user.id,
        query=q,
        skip=skip,
        limit=page_size,
    )

    return bookmarks


@router.get("/export")
def export_bookmarks(
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """导出书签"""
    repo = BookmarkRepository(db)
    bookmarks = repo.get_all_by_user(current_user.id)

    export_data = [
        BookmarkExport(
            url=b.url,
            title=b.title,
            description=b.description,
            category=b.category.name if b.category else None,
            tags=[t.name for t in b.tags],
            is_favorite=b.is_favorite,
            created_at=b.created_at.isoformat(),
        ).model_dump()
        for b in bookmarks
    ]

    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": "attachment; filename=bookmarks.json",
        },
    )


@router.post("/import", response_model=Message)
async def import_bookmarks(
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    file: UploadFile = File(...),
):
    """导入书签"""
    if not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JSON files are supported",
        )

    try:
        content = await file.read()
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON file",
        )

    repo = BookmarkRepository(db)
    imported_count = 0

    for item in data:
        try:
            bookmark_data = BookmarkImport(**item)

            # 处理标签
            tag_ids = []
            if bookmark_data.tags:
                for tag_name in bookmark_data.tags:
                    tag = db.query(Tag).filter(
                        Tag.user_id == current_user.id,
                        Tag.name == tag_name,
                    ).first()
                    if tag:
                        tag_ids.append(tag.id)

            repo.create_bookmark(
                user_id=current_user.id,
                url=str(bookmark_data.url),
                title=bookmark_data.title,
                description=bookmark_data.description,
                tag_ids=tag_ids if tag_ids else None,
            )
            imported_count += 1
        except Exception:
            continue  # 跳过无效条目

    return Message(message=f"Successfully imported {imported_count} bookmarks")


@router.get("/{bookmark_id}", response_model=BookmarkResponse)
def get_bookmark(
    bookmark_id: int,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """获取书签详情"""
    repo = BookmarkRepository(db)
    bookmark = repo.get_by_id_and_user(bookmark_id, current_user.id)

    if not bookmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bookmark not found",
        )

    return bookmark


@router.put("/{bookmark_id}", response_model=BookmarkResponse)
def update_bookmark(
    bookmark_id: int,
    bookmark_in: BookmarkUpdate,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """更新书签"""
    repo = BookmarkRepository(db)
    bookmark = repo.get_by_id_and_user(bookmark_id, current_user.id)

    if not bookmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bookmark not found",
        )

    update_data = bookmark_in.model_dump(exclude_unset=True)

    # 处理标签更新
    if "tag_ids" in update_data:
        tag_ids = update_data.pop("tag_ids")
        if tag_ids is not None:
            repo.update_tags(bookmark_id, tag_ids)

    # 处理 URL
    if "url" in update_data and update_data["url"]:
        update_data["url"] = str(update_data["url"])

    updated = repo.update(bookmark, update_data)
    db.refresh(updated)
    return updated


@router.delete("/{bookmark_id}", response_model=Message)
def delete_bookmark(
    bookmark_id: int,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """删除书签"""
    repo = BookmarkRepository(db)
    bookmark = repo.get_by_id_and_user(bookmark_id, current_user.id)

    if not bookmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bookmark not found",
        )

    repo.delete(bookmark_id)
    return Message(message="Bookmark deleted")


@router.post("/{bookmark_id}/favorite", response_model=BookmarkResponse)
def toggle_favorite(
    bookmark_id: int,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """切换收藏状态"""
    repo = BookmarkRepository(db)
    bookmark = repo.get_by_id_and_user(bookmark_id, current_user.id)

    if not bookmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bookmark not found",
        )

    repo.toggle_favorite(bookmark_id)
    db.refresh(bookmark)
    return bookmark


@router.post("/{bookmark_id}/archive", response_model=BookmarkResponse)
def toggle_archive(
    bookmark_id: int,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """切换归档状态"""
    repo = BookmarkRepository(db)
    bookmark = repo.get_by_id_and_user(bookmark_id, current_user.id)

    if not bookmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bookmark not found",
        )

    repo.toggle_archive(bookmark_id)
    db.refresh(bookmark)
    return bookmark


@router.post("/{bookmark_id}/click", response_model=Message)
def record_click(
    bookmark_id: int,
    current_user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
):
    """记录点击"""
    repo = BookmarkRepository(db)
    bookmark = repo.get_by_id_and_user(bookmark_id, current_user.id)

    if not bookmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bookmark not found",
        )

    repo.increment_click(bookmark_id)
    return Message(message="Click recorded")

