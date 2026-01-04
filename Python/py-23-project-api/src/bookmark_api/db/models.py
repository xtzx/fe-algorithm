"""
SQLAlchemy 数据库模型
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """声明式基类"""
    pass


# 书签-标签多对多关联表
bookmark_tags = Table(
    "bookmark_tags",
    Base.metadata,
    Column("bookmark_id", Integer, ForeignKey("bookmarks.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)


class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(200), nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, onupdate=func.now())

    # 关系
    bookmarks: Mapped[List["Bookmark"]] = relationship(
        "Bookmark", back_populates="user", cascade="all, delete-orphan"
    )
    categories: Mapped[List["Category"]] = relationship(
        "Category", back_populates="user", cascade="all, delete-orphan"
    )
    tags: Mapped[List["Tag"]] = relationship(
        "Tag", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}')>"


class Category(Base):
    """分类模型"""
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(500))
    icon: Mapped[Optional[str]] = mapped_column(String(50))
    color: Mapped[Optional[str]] = mapped_column(String(20))

    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    parent_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("categories.id"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # 关系
    user: Mapped["User"] = relationship("User", back_populates="categories")
    bookmarks: Mapped[List["Bookmark"]] = relationship("Bookmark", back_populates="category")

    # 自引用（子分类）
    parent: Mapped[Optional["Category"]] = relationship(
        "Category", remote_side=[id], back_populates="children"
    )
    children: Mapped[List["Category"]] = relationship("Category", back_populates="parent")

    __table_args__ = (
        Index("ix_categories_user_name", "user_id", "name"),
    )

    def __repr__(self) -> str:
        return f"<Category(id={self.id}, name='{self.name}')>"


class Tag(Base):
    """标签模型"""
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    color: Mapped[Optional[str]] = mapped_column(String(20))

    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # 关系
    user: Mapped["User"] = relationship("User", back_populates="tags")
    bookmarks: Mapped[List["Bookmark"]] = relationship(
        "Bookmark", secondary=bookmark_tags, back_populates="tags"
    )

    __table_args__ = (
        Index("ix_tags_user_name", "user_id", "name", unique=True),
    )

    def __repr__(self) -> str:
        return f"<Tag(id={self.id}, name='{self.name}')>"


class Bookmark(Base):
    """书签模型"""
    __tablename__ = "bookmarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(2000), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    favicon: Mapped[Optional[str]] = mapped_column(String(2000))

    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)
    click_count: Mapped[int] = mapped_column(Integer, default=0)

    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    category_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("categories.id", ondelete="SET NULL"))

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, onupdate=func.now())

    # 关系
    user: Mapped["User"] = relationship("User", back_populates="bookmarks")
    category: Mapped[Optional["Category"]] = relationship("Category", back_populates="bookmarks")
    tags: Mapped[List["Tag"]] = relationship(
        "Tag", secondary=bookmark_tags, back_populates="bookmarks"
    )

    __table_args__ = (
        Index("ix_bookmarks_user_id", "user_id"),
        Index("ix_bookmarks_user_category", "user_id", "category_id"),
        Index("ix_bookmarks_user_favorite", "user_id", "is_favorite"),
    )

    def __repr__(self) -> str:
        return f"<Bookmark(id={self.id}, title='{self.title[:30]}...')>"

