"""模型测试"""

import pytest
from pydantic import ValidationError

from data_lab.models import User, Product, Order, OrderItem, Gender, Status


class TestUser:
    """User 模型测试"""

    def test_create_user(self):
        """测试创建用户"""
        user = User(name="Alice", email="alice@example.com", age=30)
        assert user.name == "Alice"
        assert user.email == "alice@example.com"
        assert user.age == 30

    def test_email_lowercase(self):
        """测试邮箱转小写"""
        user = User(name="Alice", email="ALICE@EXAMPLE.COM")
        assert user.email == "alice@example.com"

    def test_name_strip_and_title(self):
        """测试名字清洗"""
        user = User(name="  alice  smith  ", email="a@b.com")
        assert user.name == "Alice Smith"

    def test_invalid_email(self):
        """测试无效邮箱"""
        with pytest.raises(ValidationError):
            User(name="Alice", email="invalid")

    def test_age_validation(self):
        """测试年龄验证"""
        with pytest.raises(ValidationError):
            User(name="Alice", email="a@b.com", age=-1)

        with pytest.raises(ValidationError):
            User(name="Alice", email="a@b.com", age=200)

    def test_phone_normalization(self):
        """测试电话号码规范化"""
        user = User(name="Alice", email="a@b.com", phone="123-456-7890")
        assert user.phone == "1234567890"

    def test_optional_fields(self):
        """测试可选字段"""
        user = User(name="Alice", email="a@b.com")
        assert user.age is None
        assert user.gender is None
        assert user.phone is None

    def test_gender_enum(self):
        """测试性别枚举"""
        user = User(name="Alice", email="a@b.com", gender="female")
        assert user.gender == Gender.FEMALE

    def test_serialization(self):
        """测试序列化"""
        user = User(name="Alice", email="a@b.com", age=30)
        data = user.model_dump()
        assert data["name"] == "Alice"
        assert data["email"] == "a@b.com"

    def test_json_serialization(self):
        """测试 JSON 序列化"""
        user = User(name="Alice", email="a@b.com", age=30)
        json_str = user.model_dump_json()
        assert "Alice" in json_str


class TestProduct:
    """Product 模型测试"""

    def test_create_product(self):
        """测试创建产品"""
        product = Product(name="iPhone", price=999.99, category="electronics")
        assert product.name == "iPhone"
        assert product.price == 999.99

    def test_price_validation(self):
        """测试价格验证"""
        with pytest.raises(ValidationError):
            Product(name="iPhone", price=-1, category="electronics")

    def test_price_rounding(self):
        """测试价格四舍五入"""
        product = Product(name="iPhone", price=999.999, category="electronics")
        assert product.price == 1000.0


class TestOrder:
    """Order 模型测试"""

    def test_create_order(self):
        """测试创建订单"""
        items = [
            OrderItem(product_id=1, product_name="iPhone", quantity=2, unit_price=999.99),
        ]
        order = Order(user_id=1, items=items)
        assert order.user_id == 1
        assert len(order.items) == 1

    def test_total_amount(self):
        """测试订单总额"""
        items = [
            OrderItem(product_id=1, product_name="iPhone", quantity=2, unit_price=100),
            OrderItem(product_id=2, product_name="Case", quantity=1, unit_price=20),
        ]
        order = Order(user_id=1, items=items)
        assert order.total_amount == 220

    def test_empty_order(self):
        """测试空订单"""
        with pytest.raises(ValidationError):
            Order(user_id=1, items=[])

