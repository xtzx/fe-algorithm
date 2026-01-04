"""验证器测试"""

import pytest

from data_lab.models import User
from data_lab.validators import validate_item, validate_batch, validate_users


class TestValidateItem:
    """validate_item 测试"""

    def test_valid_item(self):
        """测试有效数据"""
        data = {"name": "Alice", "email": "alice@example.com", "age": 30}
        instance, errors = validate_item(data, User)

        assert instance is not None
        assert len(errors) == 0
        assert instance.name == "Alice"

    def test_invalid_item(self):
        """测试无效数据"""
        data = {"name": "Alice", "email": "invalid", "age": 30}
        instance, errors = validate_item(data, User)

        assert instance is None
        assert len(errors) > 0


class TestValidateBatch:
    """validate_batch 测试"""

    def test_validate_batch(self, sample_users):
        """测试批量验证"""
        result = validate_batch(sample_users, User)

        assert result.total == 3
        assert result.success == 3
        assert result.failed == 0

    def test_validate_batch_with_errors(self, dirty_users):
        """测试带错误的批量验证"""
        result = validate_batch(dirty_users, User)

        assert result.total == 4
        assert result.failed > 0

    def test_success_rate(self, sample_users):
        """测试成功率"""
        result = validate_batch(sample_users, User)
        assert result.success_rate == 100.0


class TestValidateUsers:
    """validate_users 测试"""

    def test_validate_users(self, sample_users):
        """测试用户验证"""
        result = validate_users(sample_users)

        assert result.success == 3
        assert len(result.cleaned_data) == 3

