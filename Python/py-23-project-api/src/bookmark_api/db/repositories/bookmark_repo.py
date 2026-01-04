"""
书签 Repository
"""

from typing import List, Optional

from sqlalchemy import and_, desc, or_
from sqlalchemy.orm import Session, joinedload

from bookmark_api.db.models import Bookmark, Tag
from bookmark_api.db.repositories.base import BaseRepository


class BookmarkRepository(BaseRepository[Bookmark]):
    """书签 Repository"""

    def __init__(self, session: Session):
        super().__init__(Bookmark, session)

    def get_by_user(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 20,
        category_id: Optional[int] = None,
        is_favorite: Optional[bool] = None,
        is_archived: Optional[bool] = None,
    ) -> List[Bookmark]:
        """获取用户的书签（分页、过滤）"""
        query = self.session.query(Bookmark).filter(Bookmark.user_id == user_id)

        if category_id is not None:
            query = query.filter(Bookmark.category_id == category_id)
        if is_favorite is not None:
            query = query.filter(Bookmark.is_favorite == is_favorite)
        if is_archived is not None:
            query = query.filter(Bookmark.is_archived == is_archived)

        return (
            query.order_by(desc(Bookmark.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def count_by_user(
        self,
        user_id: int,
        category_id: Optional[int] = None,
        is_favorite: Optional[bool] = None,
        is_archived: Optional[bool] = None,
    ) -> int:
        """计算用户书签数量"""
        query = self.session.query(Bookmark).filter(Bookmark.user_id == user_id)

        if category_id is not None:
            query = query.filter(Bookmark.category_id == category_id)
        if is_favorite is not None:
            query = query.filter(Bookmark.is_favorite == is_favorite)
        if is_archived is not None:
            query = query.filter(Bookmark.is_archived == is_archived)

        return query.count()

    def get_by_id_and_user(self, bookmark_id: int, user_id: int) -> Optional[Bookmark]:
        """获取指定用户的书签"""
        return (
            self.session.query(Bookmark)
            .options(joinedload(Bookmark.tags), joinedload(Bookmark.category))
            .filter(and_(Bookmark.id == bookmark_id, Bookmark.user_id == user_id))
            .first()
        )

    def search(
        self,
        user_id: int,
        query: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[Bookmark]:
        """搜索书签"""
        search_pattern = f"%{query}%"
        return (
            self.session.query(Bookmark)
            .filter(
                and_(
                    Bookmark.user_id == user_id,
                    or_(
                        Bookmark.title.ilike(search_pattern),
                        Bookmark.url.ilike(search_pattern),
                        Bookmark.description.ilike(search_pattern),
                    ),
                )
            )
            .order_by(desc(Bookmark.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_by_tag(
        self,
        user_id: int,
        tag_id: int,
        skip: int = 0,
        limit: int = 20,
    ) -> List[Bookmark]:
        """获取指定标签的书签"""
        return (
            self.session.query(Bookmark)
            .join(Bookmark.tags)
            .filter(and_(Bookmark.user_id == user_id, Tag.id == tag_id))
            .order_by(desc(Bookmark.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def create_bookmark(
        self,
        user_id: int,
        url: str,
        title: str,
        description: Optional[str] = None,
        favicon: Optional[str] = None,
        category_id: Optional[int] = None,
        tag_ids: Optional[List[int]] = None,
    ) -> Bookmark:
        """创建书签"""
        bookmark = Bookmark(
            user_id=user_id,
            url=url,
            title=title,
            description=description,
            favicon=favicon,
            category_id=category_id,
        )

        if tag_ids:
            tags = self.session.query(Tag).filter(Tag.id.in_(tag_ids)).all()
            bookmark.tags = tags

        self.session.add(bookmark)
        self.session.commit()
        self.session.refresh(bookmark)
        return bookmark

    def update_tags(self, bookmark_id: int, tag_ids: List[int]) -> bool:
        """更新书签标签"""
        bookmark = self.get_by_id(bookmark_id)
        if not bookmark:
            return False

        tags = self.session.query(Tag).filter(Tag.id.in_(tag_ids)).all()
        bookmark.tags = tags
        self.session.commit()
        return True

    def toggle_favorite(self, bookmark_id: int) -> Optional[bool]:
        """切换收藏状态"""
        bookmark = self.get_by_id(bookmark_id)
        if bookmark:
            bookmark.is_favorite = not bookmark.is_favorite
            self.session.commit()
            return bookmark.is_favorite
        return None

    def toggle_archive(self, bookmark_id: int) -> Optional[bool]:
        """切换归档状态"""
        bookmark = self.get_by_id(bookmark_id)
        if bookmark:
            bookmark.is_archived = not bookmark.is_archived
            self.session.commit()
            return bookmark.is_archived
        return None

    def increment_click(self, bookmark_id: int) -> bool:
        """增加点击计数"""
        bookmark = self.get_by_id(bookmark_id)
        if bookmark:
            bookmark.click_count += 1
            self.session.commit()
            return True
        return False

    def get_all_by_user(self, user_id: int) -> List[Bookmark]:
        """获取用户所有书签（用于导出）"""
        return (
            self.session.query(Bookmark)
            .options(joinedload(Bookmark.tags), joinedload(Bookmark.category))
            .filter(Bookmark.user_id == user_id)
            .order_by(desc(Bookmark.created_at))
            .all()
        )

