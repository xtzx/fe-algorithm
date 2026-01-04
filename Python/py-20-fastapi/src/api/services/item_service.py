"""
商品服务

提供商品相关业务逻辑:
- 商品查询
- 商品创建
- 商品更新
- 商品删除
"""

from datetime import datetime
from decimal import Decimal

from api.schemas.item import Item, ItemCreate, ItemUpdate


# 模拟数据库
FAKE_ITEMS_DB: dict[int, dict] = {
    1: {
        "id": 1,
        "name": "iPhone 15 Pro",
        "description": "最新款 iPhone",
        "price": Decimal("7999.00"),
        "quantity": 100,
        "is_available": True,
        "owner_id": 1,
        "created_at": datetime(2024, 1, 1, 0, 0, 0),
        "updated_at": None,
    },
    2: {
        "id": 2,
        "name": "MacBook Pro 14",
        "description": "M3 Pro 芯片笔记本",
        "price": Decimal("14999.00"),
        "quantity": 50,
        "is_available": True,
        "owner_id": 1,
        "created_at": datetime(2024, 1, 2, 0, 0, 0),
        "updated_at": None,
    },
    3: {
        "id": 3,
        "name": "AirPods Pro 2",
        "description": "主动降噪耳机",
        "price": Decimal("1899.00"),
        "quantity": 200,
        "is_available": True,
        "owner_id": 2,
        "created_at": datetime(2024, 1, 3, 0, 0, 0),
        "updated_at": None,
    },
}


class ItemService:
    """商品服务类"""

    def __init__(self):
        self._next_id = max(FAKE_ITEMS_DB.keys()) + 1 if FAKE_ITEMS_DB else 1

    def get_items(
        self,
        skip: int = 0,
        limit: int = 10,
        search: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
    ) -> list[Item]:
        """获取商品列表（支持过滤）"""
        items = list(FAKE_ITEMS_DB.values())

        # 搜索过滤
        if search:
            items = [
                item
                for item in items
                if search.lower() in item["name"].lower()
                or (item["description"] and search.lower() in item["description"].lower())
            ]

        # 价格过滤
        if min_price is not None:
            items = [item for item in items if item["price"] >= Decimal(str(min_price))]
        if max_price is not None:
            items = [item for item in items if item["price"] <= Decimal(str(max_price))]

        # 分页
        items = items[skip : skip + limit]

        return [Item(**item) for item in items]

    def get_item_by_id(self, item_id: int) -> Item | None:
        """根据 ID 获取商品"""
        item_data = FAKE_ITEMS_DB.get(item_id)
        if item_data:
            return Item(**item_data)
        return None

    def create_item(self, item_data: ItemCreate, owner_id: int) -> Item:
        """创建商品"""
        item_dict = {
            "id": self._next_id,
            **item_data.model_dump(),
            "owner_id": owner_id,
            "created_at": datetime.now(),
            "updated_at": None,
        }
        FAKE_ITEMS_DB[self._next_id] = item_dict
        self._next_id += 1
        return Item(**item_dict)

    def update_item(self, item_id: int, item_data: ItemUpdate) -> Item | None:
        """更新商品"""
        if item_id not in FAKE_ITEMS_DB:
            return None

        update_dict = item_data.model_dump(exclude_unset=True)
        if update_dict:
            FAKE_ITEMS_DB[item_id].update(update_dict)
            FAKE_ITEMS_DB[item_id]["updated_at"] = datetime.now()

        return Item(**FAKE_ITEMS_DB[item_id])

    def delete_item(self, item_id: int) -> bool:
        """删除商品"""
        if item_id in FAKE_ITEMS_DB:
            del FAKE_ITEMS_DB[item_id]
            return True
        return False

    def get_items_by_owner(self, owner_id: int) -> list[Item]:
        """获取用户的所有商品"""
        items = [
            Item(**item)
            for item in FAKE_ITEMS_DB.values()
            if item["owner_id"] == owner_id
        ]
        return items

