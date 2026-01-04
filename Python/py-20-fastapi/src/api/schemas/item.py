"""
商品相关的 Pydantic 模型

职责:
- 定义商品数据结构
- 请求验证
- 响应序列化
"""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class ItemBase(BaseModel):
    """商品基础模型"""

    name: str = Field(..., min_length=1, max_length=100, description="商品名称")
    description: str | None = Field(None, max_length=500, description="商品描述")
    price: Decimal = Field(..., ge=0, decimal_places=2, description="商品价格")
    quantity: int = Field(0, ge=0, description="库存数量")
    is_available: bool = Field(True, description="是否上架")


class ItemCreate(ItemBase):
    """创建商品请求模型"""

    pass


class ItemUpdate(BaseModel):
    """更新商品请求模型（所有字段可选）"""

    name: str | None = Field(None, min_length=1, max_length=100, description="商品名称")
    description: str | None = Field(None, max_length=500, description="商品描述")
    price: Decimal | None = Field(None, ge=0, decimal_places=2, description="商品价格")
    quantity: int | None = Field(None, ge=0, description="库存数量")
    is_available: bool | None = Field(None, description="是否上架")


class Item(ItemBase):
    """商品响应模型"""

    id: int = Field(..., description="商品 ID")
    owner_id: int = Field(..., description="所有者 ID")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime | None = Field(None, description="更新时间")

    model_config = {"from_attributes": True}


class ItemList(BaseModel):
    """商品列表响应（带分页信息）"""

    items: list[Item] = Field(..., description="商品列表")
    total: int = Field(..., description="总数")
    skip: int = Field(..., description="跳过的记录数")
    limit: int = Field(..., description="返回的最大记录数")

