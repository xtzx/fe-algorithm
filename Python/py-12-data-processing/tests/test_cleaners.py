"""清洗器测试"""

import pytest

from data_lab.cleaners import (
    clean_string,
    clean_email,
    clean_phone,
    clean_name,
    safe_int,
    safe_float,
    safe_bool,
    DataCleaner,
    CleaningRule,
    dedupe_by_field,
    create_user_cleaner,
)


class TestCleanFunctions:
    """清洗函数测试"""

    def test_clean_string(self):
        """测试字符串清洗"""
        assert clean_string("  hello  ") == "hello"
        assert clean_string(None) == ""
        assert clean_string("null") == ""
        assert clean_string("N/A") == ""

    def test_clean_email(self):
        """测试邮箱清洗"""
        assert clean_email("ALICE@EXAMPLE.COM") == "alice@example.com"
        assert clean_email("invalid") == ""
        assert clean_email(None) == ""

    def test_clean_phone(self):
        """测试电话清洗"""
        assert clean_phone("123-456-7890") == "1234567890"
        assert clean_phone("(123) 456-7890") == "1234567890"
        assert clean_phone("123") == ""  # 太短

    def test_clean_name(self):
        """测试姓名清洗"""
        assert clean_name("  alice  ") == "Alice"
        assert clean_name("ALICE") == "Alice"
        assert clean_name("") == ""


class TestSafeConversions:
    """安全转换测试"""

    def test_safe_int(self):
        """测试安全整数转换"""
        assert safe_int("30") == 30
        assert safe_int("30.5") == 30
        assert safe_int("abc") == 0
        assert safe_int(None) == 0
        assert safe_int("invalid", -1) == -1

    def test_safe_float(self):
        """测试安全浮点数转换"""
        assert safe_float("30.5") == 30.5
        assert safe_float("abc") == 0.0
        assert safe_float(None) == 0.0

    def test_safe_bool(self):
        """测试安全布尔转换"""
        assert safe_bool("true") is True
        assert safe_bool("yes") is True
        assert safe_bool("1") is True
        assert safe_bool("false") is False
        assert safe_bool("no") is False
        assert safe_bool(None) is False


class TestDataCleaner:
    """DataCleaner 测试"""

    def test_clean_single(self):
        """测试清洗单条数据"""
        cleaner = DataCleaner([
            CleaningRule("name", clean_name),
            CleaningRule("age", safe_int),
        ])

        data = {"name": "  alice  ", "age": "30"}
        result = cleaner.clean(data)

        assert result["name"] == "Alice"
        assert result["age"] == 30

    def test_clean_batch(self, dirty_users):
        """测试批量清洗"""
        cleaner = create_user_cleaner()
        cleaned, stats = cleaner.clean_batch(dirty_users)

        assert stats.total == 4
        assert stats.success <= 4

    def test_required_field(self):
        """测试必填字段"""
        cleaner = DataCleaner([
            CleaningRule("name", clean_string, required=True),
        ])

        data = {"name": ""}
        cleaned, stats = cleaner.clean_batch([data])

        assert stats.failed == 1


class TestDedupe:
    """去重测试"""

    def test_dedupe_by_field(self):
        """测试单字段去重"""
        items = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 1, "name": "Alice Copy"},
        ]
        result = dedupe_by_field(items, "id")

        assert len(result) == 2
        assert result[0]["name"] == "Alice"  # 保留第一个

