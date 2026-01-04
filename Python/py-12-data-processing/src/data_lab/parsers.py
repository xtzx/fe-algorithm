"""数据解析器

支持 JSON、CSV、JSONL、YAML、TOML 格式。
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ============================================================
# JSON
# ============================================================


def read_json(path: str | Path, encoding: str = "utf-8") -> Any:
    """读取 JSON 文件

    Args:
        path: 文件路径
        encoding: 编码

    Returns:
        解析后的数据
    """
    with open(path, encoding=encoding) as f:
        return json.load(f)


def write_json(
    path: str | Path,
    data: Any,
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """写入 JSON 文件

    Args:
        path: 文件路径
        data: 数据
        encoding: 编码
        indent: 缩进
        ensure_ascii: 是否转义非 ASCII 字符
    """
    with open(path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


# ============================================================
# JSONL
# ============================================================


def read_jsonl(path: str | Path, encoding: str = "utf-8") -> Iterator[dict]:
    """读取 JSONL 文件（流式）

    Args:
        path: 文件路径
        encoding: 编码

    Yields:
        每行解析的字典
    """
    with open(path, encoding=encoding) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")


def write_jsonl(
    path: str | Path,
    data: Iterator[dict] | list[dict],
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
) -> int:
    """写入 JSONL 文件

    Args:
        path: 文件路径
        data: 数据迭代器
        encoding: 编码
        ensure_ascii: 是否转义非 ASCII 字符

    Returns:
        写入的行数
    """
    count = 0
    with open(path, "w", encoding=encoding) as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
            count += 1
    return count


# ============================================================
# CSV
# ============================================================


def read_csv(
    path: str | Path,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> list[dict]:
    """读取 CSV 文件

    Args:
        path: 文件路径
        encoding: 编码
        delimiter: 分隔符

    Returns:
        字典列表
    """
    # 尝试处理 BOM
    try:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return list(reader)
    except UnicodeDecodeError:
        pass

    with open(path, encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader)


def read_csv_stream(
    path: str | Path,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> Iterator[dict]:
    """流式读取 CSV 文件

    Args:
        path: 文件路径
        encoding: 编码
        delimiter: 分隔符

    Yields:
        每行的字典
    """
    with open(path, encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        yield from reader


def write_csv(
    path: str | Path,
    data: list[dict],
    fieldnames: list[str] | None = None,
    encoding: str = "utf-8",
) -> None:
    """写入 CSV 文件

    Args:
        path: 文件路径
        data: 数据列表
        fieldnames: 字段名列表
        encoding: 编码
    """
    if not data:
        return

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(path, "w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


# ============================================================
# YAML / TOML
# ============================================================


def read_yaml(path: str | Path) -> Any:
    """读取 YAML 文件

    Args:
        path: 文件路径

    Returns:
        解析后的数据
    """
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        raise ImportError("Please install pyyaml: pip install pyyaml")


def read_toml(path: str | Path) -> dict:
    """读取 TOML 文件

    Args:
        path: 文件路径

    Returns:
        解析后的字典
    """
    import tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


# ============================================================
# 格式转换
# ============================================================


def csv_to_jsonl(
    csv_path: str | Path,
    jsonl_path: str | Path,
    encoding: str = "utf-8",
) -> int:
    """CSV 转 JSONL

    Args:
        csv_path: CSV 文件路径
        jsonl_path: JSONL 输出路径
        encoding: 编码

    Returns:
        转换的行数
    """
    count = 0
    with open(csv_path, encoding=encoding) as fin:
        with open(jsonl_path, "w", encoding=encoding) as fout:
            reader = csv.DictReader(fin)
            for row in reader:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count


def jsonl_to_csv(
    jsonl_path: str | Path,
    csv_path: str | Path,
    fieldnames: list[str] | None = None,
) -> int:
    """JSONL 转 CSV

    Args:
        jsonl_path: JSONL 文件路径
        csv_path: CSV 输出路径
        fieldnames: 字段名列表

    Returns:
        转换的行数
    """
    # 先读取所有数据以确定字段
    data = list(read_jsonl(jsonl_path))
    if not data:
        return 0

    if fieldnames is None:
        # 收集所有字段
        all_fields: set[str] = set()
        for item in data:
            all_fields.update(item.keys())
        fieldnames = sorted(all_fields)

    write_csv(csv_path, data, fieldnames)
    return len(data)

