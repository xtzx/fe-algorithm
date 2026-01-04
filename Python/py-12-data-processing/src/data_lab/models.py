"""pydantic 数据模型

定义各种数据验证模型。
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Gender(str, Enum):
    """性别枚举"""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class Status(str, Enum):
    """状态枚举"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class User(BaseModel):
    """用户模型"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="ignore",
    )

    id: int | None = None
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=255)
    age: int | None = Field(default=None, ge=0, le=150)
    gender: Gender | None = None
    phone: str | None = None
    status: Status = Status.ACTIVE
    created_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower()

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # 只保留数字
        digits = "".join(c for c in v if c.isdigit())
        if len(digits) < 10:
            return None
        return digits

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        # 规范化名字
        return " ".join(v.split()).title()


class Product(BaseModel):
    """产品模型"""

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    id: int | None = None
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    price: float = Field(..., gt=0)
    category: str = Field(..., min_length=1)
    in_stock: bool = True
    quantity: int = Field(default=0, ge=0)
    tags: list[str] = Field(default_factory=list)

    @field_validator("price")
    @classmethod
    def round_price(cls, v: float) -> float:
        return round(v, 2)


class OrderItem(BaseModel):
    """订单项模型"""

    product_id: int
    product_name: str
    quantity: int = Field(..., gt=0)
    unit_price: float = Field(..., gt=0)

    @property
    def total(self) -> float:
        return round(self.quantity * self.unit_price, 2)


class Order(BaseModel):
    """订单模型"""

    id: int | None = None
    user_id: int
    items: list[OrderItem] = Field(..., min_length=1)
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.now)
    notes: str = ""

    @property
    def total_amount(self) -> float:
        return round(sum(item.total for item in self.items), 2)

    @property
    def total_items(self) -> int:
        return sum(item.quantity for item in self.items)


class Address(BaseModel):
    """地址模型"""

    street: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    state: str = ""
    country: str = "China"
    zip_code: str = ""

    @field_validator("zip_code")
    @classmethod
    def validate_zip(cls, v: str) -> str:
        if v and not v.isdigit():
            raise ValueError("Zip code must be digits only")
        return v


class CleaningResult(BaseModel):
    """清洗结果"""

    total: int = 0
    success: int = 0
    failed: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)
    cleaned_data: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return round(self.success / self.total * 100, 2)


class RawRecord(BaseModel):
    """原始记录（宽松验证）"""

    model_config = ConfigDict(extra="allow")

    # 所有字段都是可选的
    id: str | int | None = None
    name: str | None = None
    email: str | None = None
    age: str | int | None = None
    phone: str | None = None

    def to_user(self) -> User | None:
        """尝试转换为 User"""
        try:
            data = {
                "name": self.name or "",
                "email": self.email or "",
                "age": int(self.age) if self.age else None,
                "phone": self.phone,
            }
            if self.id:
                data["id"] = int(self.id) if isinstance(self.id, str) else self.id
            return User(**data)
        except Exception:
            return None

