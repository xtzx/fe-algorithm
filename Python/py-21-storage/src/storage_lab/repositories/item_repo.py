"""
商品 Repository

提供商品相关的数据库操作
"""

from decimal import Decimal
from typing import List, Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, joinedload

from storage_lab.db.models import Item, Tag
from storage_lab.repositories.base import AsyncBaseRepository, BaseRepository


class ItemRepository(BaseRepository[Item]):
    """
    商品 Repository（同步）
    """

    def __init__(self, session: Session):
        super().__init__(Item, session)

    def get_by_owner(
        self,
        owner_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Item]:
        """获取用户的所有商品"""
        return (
            self.session.query(Item)
            .filter(Item.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_available(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Item]:
        """获取所有上架商品"""
        return (
            self.session.query(Item)
            .filter(Item.is_available == True)  # noqa: E712
            .offset(skip)
            .limit(limit)
            .all()
        )

    def search(
        self,
        query: str,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Item]:
        """搜索商品（支持价格过滤）"""
        filters = [Item.name.ilike(f"%{query}%")]

        if min_price is not None:
            filters.append(Item.price >= min_price)
        if max_price is not None:
            filters.append(Item.price <= max_price)

        return (
            self.session.query(Item)
            .filter(and_(*filters))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_with_tags(self, item_id: int) -> Optional[Item]:
        """获取商品及其标签（解决 N+1）"""
        return (
            self.session.query(Item)
            .options(joinedload(Item.tags))
            .filter(Item.id == item_id)
            .first()
        )

    def get_by_tag(
        self,
        tag_name: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Item]:
        """获取具有指定标签的商品"""
        return (
            self.session.query(Item)
            .join(Item.tags)
            .filter(Tag.name == tag_name)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def create_item(
        self,
        name: str,
        price: Decimal,
        owner_id: int,
        description: Optional[str] = None,
        quantity: int = 0,
    ) -> Item:
        """创建商品"""
        item = Item(
            name=name,
            price=price,
            owner_id=owner_id,
            description=description,
            quantity=quantity,
        )
        self.session.add(item)
        self.session.commit()
        self.session.refresh(item)
        return item

    def add_tag(self, item_id: int, tag_name: str) -> bool:
        """给商品添加标签"""
        item = self.get_by_id(item_id)
        if not item:
            return False

        # 查找或创建标签
        tag = self.session.query(Tag).filter(Tag.name == tag_name).first()
        if not tag:
            tag = Tag(name=tag_name)
            self.session.add(tag)

        if tag not in item.tags:
            item.tags.append(tag)
            self.session.commit()

        return True

    def remove_tag(self, item_id: int, tag_name: str) -> bool:
        """移除商品标签"""
        item = self.get_with_tags(item_id)
        if not item:
            return False

        tag = self.session.query(Tag).filter(Tag.name == tag_name).first()
        if tag and tag in item.tags:
            item.tags.remove(tag)
            self.session.commit()
            return True

        return False

    def update_quantity(self, item_id: int, delta: int) -> bool:
        """更新库存数量"""
        item = self.get_by_id(item_id)
        if item:
            new_quantity = item.quantity + delta
            if new_quantity >= 0:
                item.quantity = new_quantity
                self.session.commit()
                return True
        return False


class AsyncItemRepository(AsyncBaseRepository[Item]):
    """
    商品 Repository（异步）
    """

    def __init__(self, session: AsyncSession):
        super().__init__(Item, session)

    async def get_by_owner(
        self,
        owner_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Item]:
        """获取用户的所有商品"""
        result = await self.session.execute(
            select(Item)
            .where(Item.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_available(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Item]:
        """获取所有上架商品"""
        result = await self.session.execute(
            select(Item)
            .where(Item.is_available == True)  # noqa: E712
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_with_tags(self, item_id: int) -> Optional[Item]:
        """获取商品及其标签"""
        result = await self.session.execute(
            select(Item)
            .options(joinedload(Item.tags))
            .where(Item.id == item_id)
        )
        return result.unique().scalar_one_or_none()

    async def create_item(
        self,
        name: str,
        price: Decimal,
        owner_id: int,
        description: Optional[str] = None,
        quantity: int = 0,
    ) -> Item:
        """创建商品"""
        item = Item(
            name=name,
            price=price,
            owner_id=owner_id,
            description=description,
            quantity=quantity,
        )
        self.session.add(item)
        await self.session.commit()
        await self.session.refresh(item)
        return item


