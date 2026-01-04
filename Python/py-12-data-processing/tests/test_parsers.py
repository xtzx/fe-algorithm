"""解析器测试"""

import json
import pytest
from pathlib import Path

from data_lab.parsers import (
    read_json,
    write_json,
    read_jsonl,
    write_jsonl,
    read_csv,
    write_csv,
    csv_to_jsonl,
)


class TestJSON:
    """JSON 解析测试"""

    def test_read_json(self, tmp_path: Path):
        """测试读取 JSON"""
        json_path = tmp_path / "test.json"
        json_path.write_text('{"name": "Alice"}')

        data = read_json(json_path)
        assert data["name"] == "Alice"

    def test_write_json(self, tmp_path: Path):
        """测试写入 JSON"""
        json_path = tmp_path / "test.json"
        write_json(json_path, {"name": "Alice"})

        content = json_path.read_text()
        assert "Alice" in content


class TestJSONL:
    """JSONL 解析测试"""

    def test_read_jsonl(self, temp_jsonl: Path):
        """测试读取 JSONL"""
        items = list(read_jsonl(temp_jsonl))
        assert len(items) == 2
        assert items[0]["name"] == "Alice"

    def test_write_jsonl(self, tmp_path: Path):
        """测试写入 JSONL"""
        jsonl_path = tmp_path / "test.jsonl"
        data = [{"id": 1}, {"id": 2}]
        count = write_jsonl(jsonl_path, data)

        assert count == 2
        items = list(read_jsonl(jsonl_path))
        assert len(items) == 2

    def test_read_jsonl_empty_lines(self, tmp_path: Path):
        """测试 JSONL 空行处理"""
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text('{"id": 1}\n\n{"id": 2}\n')

        items = list(read_jsonl(jsonl_path))
        assert len(items) == 2


class TestCSV:
    """CSV 解析测试"""

    def test_read_csv(self, temp_csv: Path):
        """测试读取 CSV"""
        data = read_csv(temp_csv)
        assert len(data) == 2
        assert data[0]["name"] == "Alice"

    def test_write_csv(self, tmp_path: Path):
        """测试写入 CSV"""
        csv_path = tmp_path / "test.csv"
        data = [{"name": "Alice", "age": 30}]
        write_csv(csv_path, data)

        result = read_csv(csv_path)
        assert len(result) == 1
        assert result[0]["name"] == "Alice"

    def test_write_csv_empty(self, tmp_path: Path):
        """测试写入空 CSV"""
        csv_path = tmp_path / "empty.csv"
        write_csv(csv_path, [])
        assert not csv_path.exists() or csv_path.stat().st_size == 0


class TestConversion:
    """格式转换测试"""

    def test_csv_to_jsonl(self, temp_csv: Path, tmp_path: Path):
        """测试 CSV 转 JSONL"""
        jsonl_path = tmp_path / "output.jsonl"
        count = csv_to_jsonl(temp_csv, jsonl_path)

        assert count == 2
        items = list(read_jsonl(jsonl_path))
        assert len(items) == 2

