"""
商品路由

提供商品 CRUD 操作:
- 获取商品列表
- 获取单个商品
- 创建商品
- 更新商品
- 删除商品
"""

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from api.dependencies.auth import get_current_active_user
from api.schemas.item import Item, ItemCreate, ItemUpdate
from api.schemas.user import User
from api.services.item_service import ItemService

router = APIRouter()


@router.get("/", response_model=list[Item])
async def list_items(
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(10, ge=1, le=100, description="返回的最大记录数"),
    q: str | None = Query(None, min_length=1, max_length=50, description="搜索关键词"),
    min_price: float | None = Query(None, ge=0, description="最低价格"),
    max_price: float | None = Query(None, ge=0, description="最高价格"),
    item_service: ItemService = Depends(),
):
    """
    获取商品列表（支持分页和过滤）

    - **skip**: 跳过的记录数
    - **limit**: 返回的最大记录数
    - **q**: 搜索关键词（模糊匹配名称）
    - **min_price**: 最低价格过滤
    - **max_price**: 最高价格过滤
    """
    return item_service.get_items(
        skip=skip,
        limit=limit,
        search=q,
        min_price=min_price,
        max_price=max_price,
    )


@router.get("/{item_id}", response_model=Item)
async def get_item(
    item_id: int = Path(..., ge=1, description="商品 ID"),
    item_service: ItemService = Depends(),
):
    """
    根据 ID 获取商品详情

    - **item_id**: 商品 ID
    """
    item = item_service.get_item_by_id(item_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"商品 {item_id} 不存在",
        )
    return item


@router.post("/", response_model=Item, status_code=status.HTTP_201_CREATED)
async def create_item(
    item_data: ItemCreate,
    item_service: ItemService = Depends(),
    current_user: User = Depends(get_current_active_user),
):
    """
    创建商品

    - **name**: 商品名称
    - **description**: 商品描述（可选）
    - **price**: 商品价格
    - **quantity**: 库存数量
    """
    return item_service.create_item(item_data, owner_id=current_user.id)


@router.put("/{item_id}", response_model=Item)
async def update_item(
    item_id: int,
    item_data: ItemUpdate,
    item_service: ItemService = Depends(),
    current_user: User = Depends(get_current_active_user),
):
    """
    更新商品信息

    - **item_id**: 商品 ID
    - **item_data**: 要更新的字段
    """
    # 检查商品是否存在
    existing_item = item_service.get_item_by_id(item_id)
    if not existing_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"商品 {item_id} 不存在",
        )

    # 检查权限：只有所有者或管理员可以更新
    if existing_item.owner_id != current_user.id and "admin" not in current_user.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限更新此商品",
        )

    return item_service.update_item(item_id, item_data)


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(
    item_id: int,
    item_service: ItemService = Depends(),
    current_user: User = Depends(get_current_active_user),
):
    """
    删除商品

    - **item_id**: 商品 ID
    """
    # 检查商品是否存在
    existing_item = item_service.get_item_by_id(item_id)
    if not existing_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"商品 {item_id} 不存在",
        )

    # 检查权限
    if existing_item.owner_id != current_user.id and "admin" not in current_user.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限删除此商品",
        )

    item_service.delete_item(item_id)
    return None

