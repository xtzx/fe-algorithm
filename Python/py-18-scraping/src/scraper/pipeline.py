"""
数据管道模块

支持:
- JSONL 输出
- 数据清洗
- 数据验证
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class PipelineStep(ABC):
    """管道步骤基类"""

    @abstractmethod
    async def process(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """
        处理数据项

        Args:
            item: 输入数据

        Returns:
            处理后的数据，返回 None 则丢弃该项
        """
        pass

    async def open(self) -> None:
        """初始化"""
        pass

    async def close(self) -> None:
        """清理"""
        pass


class Pipeline:
    """
    数据管道

    Example:
        ```python
        pipeline = Pipeline([
            CleanStep(),
            ValidateStep(),
            JsonLineWriter("items.jsonl"),
        ])

        async with pipeline:
            await pipeline.process({"title": "Hello"})
        ```
    """

    def __init__(self, steps: list[PipelineStep]) -> None:
        self.steps = steps
        self._processed_count = 0
        self._dropped_count = 0

    async def __aenter__(self) -> "Pipeline":
        for step in self.steps:
            await step.open()
        return self

    async def __aexit__(self, *args) -> None:
        for step in self.steps:
            await step.close()

    async def process(self, item: Any) -> bool:
        """
        处理数据项

        Returns:
            True 如果成功处理，False 如果被丢弃
        """
        # 转换为字典
        if is_dataclass(item) and not isinstance(item, dict):
            item = asdict(item)

        # 依次执行每个步骤
        for step in self.steps:
            item = await step.process(item)
            if item is None:
                self._dropped_count += 1
                return False

        self._processed_count += 1
        return True

    @property
    def processed_count(self) -> int:
        """已处理的数量"""
        return self._processed_count

    @property
    def dropped_count(self) -> int:
        """被丢弃的数量"""
        return self._dropped_count


class JsonLineWriter(PipelineStep):
    """
    JSONL 写入器

    每行一个 JSON 对象

    Example:
        ```python
        writer = JsonLineWriter("items.jsonl")

        async with Pipeline([writer]) as pipeline:
            await pipeline.process({"title": "Hello"})
        ```
    """

    def __init__(
        self,
        path: str | Path,
        append: bool = True,
        ensure_ascii: bool = False,
    ) -> None:
        self._path = Path(path)
        self._append = append
        self._ensure_ascii = ensure_ascii
        self._file = None
        self._count = 0

    async def open(self) -> None:
        mode = "a" if self._append else "w"
        self._file = self._path.open(mode, encoding="utf-8")

    async def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    async def process(self, item: dict[str, Any]) -> dict[str, Any]:
        if self._file:
            line = json.dumps(item, ensure_ascii=self._ensure_ascii)
            self._file.write(line + "\n")
            self._file.flush()
            self._count += 1
        return item

    @property
    def count(self) -> int:
        return self._count


class CleanStep(PipelineStep):
    """
    数据清洗步骤

    - 去除空白
    - 移除空值
    """

    def __init__(
        self,
        strip_strings: bool = True,
        remove_empty: bool = True,
        fields: list[str] | None = None,
    ) -> None:
        self.strip_strings = strip_strings
        self.remove_empty = remove_empty
        self.fields = fields

    async def process(self, item: dict[str, Any]) -> dict[str, Any]:
        result = {}

        for key, value in item.items():
            # 只处理指定字段
            if self.fields and key not in self.fields:
                result[key] = value
                continue

            # 字符串处理
            if isinstance(value, str):
                if self.strip_strings:
                    value = value.strip()
                if self.remove_empty and not value:
                    continue

            # 列表处理
            elif isinstance(value, list):
                if self.remove_empty and not value:
                    continue

            # None 处理
            elif value is None:
                if self.remove_empty:
                    continue

            result[key] = value

        return result


class ValidateStep(PipelineStep):
    """
    数据验证步骤

    Example:
        ```python
        validate = ValidateStep(
            required_fields=["title", "url"],
            validators={
                "url": lambda x: x.startswith("http"),
            },
        )
        ```
    """

    def __init__(
        self,
        required_fields: list[str] | None = None,
        validators: dict[str, Any] | None = None,
        drop_invalid: bool = True,
    ) -> None:
        self.required_fields = required_fields or []
        self.validators = validators or {}
        self.drop_invalid = drop_invalid

    async def process(self, item: dict[str, Any]) -> dict[str, Any] | None:
        # 检查必填字段
        for field in self.required_fields:
            if field not in item or not item[field]:
                if self.drop_invalid:
                    return None
                raise ValueError(f"Missing required field: {field}")

        # 运行验证器
        for field, validator in self.validators.items():
            if field in item:
                if not validator(item[field]):
                    if self.drop_invalid:
                        return None
                    raise ValueError(f"Validation failed for field: {field}")

        return item


class DeduplicateStep(PipelineStep):
    """
    去重步骤

    基于指定字段去重
    """

    def __init__(self, field: str = "url") -> None:
        self.field = field
        self._seen: set[str] = set()

    async def process(self, item: dict[str, Any]) -> dict[str, Any] | None:
        value = item.get(self.field)
        if value is None:
            return item

        key = str(value)
        if key in self._seen:
            return None

        self._seen.add(key)
        return item


class TransformStep(PipelineStep):
    """
    数据转换步骤

    Example:
        ```python
        transform = TransformStep({
            "title": str.upper,
            "url": lambda x: x.lower(),
        })
        ```
    """

    def __init__(self, transforms: dict[str, Any]) -> None:
        self.transforms = transforms

    async def process(self, item: dict[str, Any]) -> dict[str, Any]:
        for field, transform in self.transforms.items():
            if field in item:
                item[field] = transform(item[field])
        return item


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    加载 JSONL 文件

    Example:
        ```python
        items = load_jsonl("items.jsonl")
        ```
    """
    items = []
    path = Path(path)
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    return items

