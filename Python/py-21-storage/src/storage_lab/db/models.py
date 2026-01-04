"""
SQLAlchemy 模型定义

演示:
- 模型定义
- 一对多关系
- 多对多关系
- 常用字段类型
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    DECIMAL,
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
    """
    声明式基类
    
    所有模型都继承自这个基类
    """
    pass


# ==================== 多对多关联表 ====================

item_tags = Table(
    "item_tags",
    Base.metadata,
    Column("item_id", Integer, ForeignKey("items.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)


# ==================== 模型定义 ====================


class User(Base):
    """
    用户模型
    
    演示:
    - 基础字段
    - 一对多关系
    - 索引
    """
    __tablename__ = "users"
    
    # 主键
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # 基础字段
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(200), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # 状态字段
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, onupdate=func.now(), nullable=True
    )
    
    # 一对多关系：用户 -> 商品
    items: Mapped[List["Item"]] = relationship(
        "Item",
        back_populates="owner",
        cascade="all, delete-orphan",  # 级联删除
        lazy="selectin",  # 预加载策略
    )
    
    # 表级配置
    __table_args__ = (
        Index("ix_users_email", "email"),
        Index("ix_users_username", "username"),
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}')>"


class Item(Base):
    """
    商品模型
    
    演示:
    - 外键关系
    - 多对多关系
    - Decimal 类型
    """
    __tablename__ = "items"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # 基础字段
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    price: Mapped[Decimal] = mapped_column(DECIMAL(10, 2), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    
    # 状态
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # 外键
    owner_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, onupdate=func.now(), nullable=True
    )
    
    # 多对一关系：商品 -> 用户
    owner: Mapped["User"] = relationship("User", back_populates="items")
    
    # 多对多关系：商品 <-> 标签
    tags: Mapped[List["Tag"]] = relationship(
        "Tag",
        secondary=item_tags,
        back_populates="items",
        lazy="selectin",
    )
    
    __table_args__ = (
        Index("ix_items_name", "name"),
        Index("ix_items_owner_id", "owner_id"),
    )
    
    def __repr__(self) -> str:
        return f"<Item(id={self.id}, name='{self.name}', price={self.price})>"


class Tag(Base):
    """
    标签模型
    
    演示:
    - 多对多关系
    """
    __tablename__ = "tags"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    
    # 多对多关系：标签 <-> 商品
    items: Mapped[List["Item"]] = relationship(
        "Item",
        secondary=item_tags,
        back_populates="tags",
    )
    
    def __repr__(self) -> str:
        return f"<Tag(id={self.id}, name='{self.name}')>"


# ==================== 自引用关系示例 ====================


class Category(Base):
    """
    分类模型（自引用关系示例）
    
    演示:
    - 自引用（父子分类）
    """
    __tablename__ = "categories"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # 父分类 ID
    parent_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("categories.id"), nullable=True
    )
    
    # 自引用关系
    parent: Mapped[Optional["Category"]] = relationship(
        "Category",
        remote_side=[id],
        back_populates="children",
    )
    children: Mapped[List["Category"]] = relationship(
        "Category",
        back_populates="parent",
    )
    
    def __repr__(self) -> str:
        return f"<Category(id={self.id}, name='{self.name}')>"


