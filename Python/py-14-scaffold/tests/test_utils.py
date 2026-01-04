"""
测试工具函数模块
"""

import time
import pytest
from pathlib import Path

from scaffold.utils import (
    timer,
    retry,
    read_json,
    write_json,
    file_hash,
    ensure_dir,
    snake_to_camel,
    camel_to_snake,
    truncate,
    format_duration,
    chunk_list,
    flatten,
    unique,
)


class TestTimer:
    """测试计时装饰器"""

    def test_timer_returns_result(self):
        """测试返回正确结果"""

        @timer
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_timer_logs(self, caplog):
        """测试记录日志"""
        import logging

        with caplog.at_level(logging.DEBUG):

            @timer
            def slow_func():
                time.sleep(0.01)

            slow_func()

        assert "slow_func" in caplog.text
        assert "took" in caplog.text


class TestRetry:
    """测试重试装饰器"""

    def test_retry_success(self):
        """测试成功不重试"""
        call_count = 0

        @retry(max_attempts=3)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()

        assert result == "success"
        assert call_count == 1

    def test_retry_eventual_success(self):
        """测试最终成功"""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = fails_twice()

        assert result == "success"
        assert call_count == 3

    def test_retry_all_fail(self):
        """测试全部失败"""

        @retry(max_attempts=3, delay=0.01)
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()


class TestJsonOperations:
    """测试 JSON 操作"""

    def test_write_and_read_json(self, tmp_path: Path):
        """测试写入和读取 JSON"""
        file_path = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        write_json(file_path, data)
        result = read_json(file_path)

        assert result == data

    def test_write_creates_parent_dirs(self, tmp_path: Path):
        """测试自动创建父目录"""
        file_path = tmp_path / "nested" / "dir" / "test.json"
        data = {"key": "value"}

        write_json(file_path, data)

        assert file_path.exists()


class TestFileHash:
    """测试文件哈希"""

    def test_file_hash_md5(self, tmp_path: Path):
        """测试 MD5 哈希"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello world")

        hash_value = file_hash(file_path, "md5")

        assert len(hash_value) == 32
        assert hash_value == "5eb63bbbe01eeed093cb22bb8f5acdc3"

    def test_file_hash_sha256(self, tmp_path: Path):
        """测试 SHA256 哈希"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello world")

        hash_value = file_hash(file_path, "sha256")

        assert len(hash_value) == 64


class TestEnsureDir:
    """测试确保目录存在"""

    def test_creates_directory(self, tmp_path: Path):
        """测试创建目录"""
        new_dir = tmp_path / "new" / "nested" / "dir"

        result = ensure_dir(new_dir)

        assert new_dir.exists()
        assert result == new_dir

    def test_existing_directory(self, tmp_path: Path):
        """测试已存在的目录"""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = ensure_dir(existing_dir)

        assert existing_dir.exists()
        assert result == existing_dir


class TestStringOperations:
    """测试字符串操作"""

    def test_snake_to_camel(self):
        """测试 snake_case 转 camelCase"""
        assert snake_to_camel("hello_world") == "helloWorld"
        assert snake_to_camel("foo_bar_baz") == "fooBarBaz"
        assert snake_to_camel("single") == "single"

    def test_camel_to_snake(self):
        """测试 camelCase 转 snake_case"""
        assert camel_to_snake("helloWorld") == "hello_world"
        assert camel_to_snake("fooBarBaz") == "foo_bar_baz"
        assert camel_to_snake("single") == "single"

    def test_truncate(self):
        """测试字符串截断"""
        assert truncate("hello", 10) == "hello"
        assert truncate("hello world", 8) == "hello..."
        assert truncate("hello world", 8, "..") == "hello .."


class TestFormatDuration:
    """测试持续时间格式化"""

    def test_milliseconds(self):
        """测试毫秒"""
        assert format_duration(0.5) == "500ms"
        assert format_duration(0.001) == "1ms"

    def test_seconds(self):
        """测试秒"""
        assert format_duration(1.5) == "1.5s"
        assert format_duration(30) == "30.0s"

    def test_minutes(self):
        """测试分钟"""
        assert format_duration(65) == "1m 5s"
        assert format_duration(120) == "2m 0s"

    def test_hours(self):
        """测试小时"""
        assert format_duration(3700) == "1h 1m"
        assert format_duration(7200) == "2h 0m"


class TestListOperations:
    """测试列表操作"""

    def test_chunk_list(self):
        """测试列表分块"""
        assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
        assert chunk_list([1, 2, 3], 5) == [[1, 2, 3]]

    def test_flatten(self):
        """测试列表展平"""
        assert flatten([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]
        assert flatten([[], [1], []]) == [1]

    def test_unique(self):
        """测试去重（保持顺序）"""
        assert unique([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]
        assert unique(["a", "b", "a", "c"]) == ["a", "b", "c"]

