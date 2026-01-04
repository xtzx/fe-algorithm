"""
共享测试 fixtures
"""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # 清理
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_files(temp_dir: Path):
    """创建示例文件"""
    files = [
        "report_20240101.txt",
        "report_20240102.txt",
        "image.jpg",
        "document.pdf",
        "data.json",
    ]

    for filename in files:
        file_path = temp_dir / filename
        file_path.write_text(f"Content of {filename}")

    return temp_dir


@pytest.fixture
def nested_files(temp_dir: Path):
    """创建嵌套目录结构"""
    structure = {
        "docs/readme.md": "# Readme",
        "docs/guide.txt": "Guide content",
        "src/main.py": "print('hello')",
        "src/utils/helpers.py": "def helper(): pass",
        "data/sample.json": '{"key": "value"}',
    }

    for path, content in structure.items():
        file_path = temp_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    return temp_dir


@pytest.fixture
def state_file(temp_dir: Path):
    """状态文件路径"""
    return temp_dir / "state.json"


@pytest.fixture
def rollback_file(temp_dir: Path):
    """回滚日志路径"""
    return temp_dir / "rollback.json"

