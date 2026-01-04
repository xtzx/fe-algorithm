"""测试 fixture"""

import pytest
from pathlib import Path


@pytest.fixture
def sample_users() -> list[dict]:
    """示例用户数据"""
    return [
        {"name": "Alice", "email": "alice@example.com", "age": 30},
        {"name": "Bob", "email": "bob@example.com", "age": 25},
        {"name": "Charlie", "email": "charlie@example.com", "age": 35},
    ]


@pytest.fixture
def dirty_users() -> list[dict]:
    """脏用户数据"""
    return [
        {"name": "  alice  ", "email": "ALICE@EXAMPLE.COM", "age": "30"},
        {"name": "BOB", "email": "bob@example.com", "age": "25"},
        {"name": "", "email": "invalid", "age": "abc"},
        {"name": "Charlie", "email": None, "age": 35},
    ]


@pytest.fixture
def temp_csv(tmp_path: Path) -> Path:
    """临时 CSV 文件"""
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "name,email,age\n"
        "Alice,alice@example.com,30\n"
        "Bob,bob@example.com,25\n"
    )
    return csv_path


@pytest.fixture
def temp_jsonl(tmp_path: Path) -> Path:
    """临时 JSONL 文件"""
    jsonl_path = tmp_path / "test.jsonl"
    jsonl_path.write_text(
        '{"name": "Alice", "email": "alice@example.com", "age": 30}\n'
        '{"name": "Bob", "email": "bob@example.com", "age": 25}\n'
    )
    return jsonl_path


@pytest.fixture
def dirty_csv(tmp_path: Path) -> Path:
    """脏数据 CSV 文件"""
    csv_path = tmp_path / "dirty.csv"
    csv_path.write_text(
        "name,email,age,phone\n"
        "  Alice  ,ALICE@EXAMPLE.COM,30,123-456-7890\n"
        "Bob,bob@example.com,25,\n"
        ",invalid-email,abc,\n"
        "Charlie,charlie@example.com,35,9876543210\n"
    )
    return csv_path

