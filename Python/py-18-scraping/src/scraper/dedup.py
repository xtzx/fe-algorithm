"""
URL 去重模块

支持:
- 内存哈希去重
- 布隆过滤器（大规模）
- 持久化去重
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator


class UrlDedup(ABC):
    """URL 去重基类"""

    @abstractmethod
    def add(self, url: str) -> bool:
        """
        添加 URL

        Returns:
            True 如果是新 URL，False 如果已存在
        """
        pass

    @abstractmethod
    def contains(self, url: str) -> bool:
        """检查 URL 是否已存在"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """返回已添加的 URL 数量"""
        pass


class HashSetDedup(UrlDedup):
    """
    基于 HashSet 的去重

    适用于中小规模爬虫

    Example:
        ```python
        dedup = HashSetDedup()

        if dedup.add("https://example.com"):
            print("New URL!")
        else:
            print("Already seen")
        ```
    """

    def __init__(self, initial_urls: set[str] | None = None) -> None:
        self._seen: set[str] = initial_urls or set()

    def add(self, url: str) -> bool:
        """添加 URL，返回是否是新 URL"""
        normalized = self._normalize(url)
        if normalized in self._seen:
            return False
        self._seen.add(normalized)
        return True

    def contains(self, url: str) -> bool:
        """检查 URL 是否已存在"""
        return self._normalize(url) in self._seen

    def __len__(self) -> int:
        return len(self._seen)

    def __iter__(self) -> Iterator[str]:
        return iter(self._seen)

    def _normalize(self, url: str) -> str:
        """标准化 URL"""
        # 移除尾部斜杠和 fragment
        url = url.split("#")[0]
        return url.rstrip("/")

    def save(self, path: str | Path) -> None:
        """保存到文件"""
        path = Path(path)
        with path.open("w") as f:
            json.dump(list(self._seen), f)

    @classmethod
    def load(cls, path: str | Path) -> "HashSetDedup":
        """从文件加载"""
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open() as f:
            urls = json.load(f)
        return cls(set(urls))


class BloomFilter(UrlDedup):
    """
    布隆过滤器去重

    适用于大规模爬虫，内存效率高

    特点:
    - 可能有假阳性（说有但实际没有）
    - 不会有假阴性（说没有就一定没有）

    Example:
        ```python
        dedup = BloomFilter(expected_items=1000000)

        dedup.add("https://example.com")
        dedup.contains("https://example.com")  # True
        ```
    """

    def __init__(
        self,
        expected_items: int = 1000000,
        false_positive_rate: float = 0.01,
    ) -> None:
        """
        初始化布隆过滤器

        Args:
            expected_items: 预期存储的元素数量
            false_positive_rate: 假阳性率
        """
        import math

        # 计算最优的 bit 数组大小
        self.size = int(
            -(expected_items * math.log(false_positive_rate)) / (math.log(2) ** 2)
        )
        # 计算最优的哈希函数数量
        self.num_hashes = int((self.size / expected_items) * math.log(2))

        self._bit_array = bytearray((self.size + 7) // 8)
        self._count = 0

    def _get_hashes(self, url: str) -> list[int]:
        """生成多个哈希值"""
        hashes = []
        for i in range(self.num_hashes):
            # 使用不同的种子生成不同的哈希
            h = hashlib.md5(f"{url}_{i}".encode()).hexdigest()
            hashes.append(int(h, 16) % self.size)
        return hashes

    def _set_bit(self, position: int) -> None:
        """设置指定位"""
        byte_index = position // 8
        bit_index = position % 8
        self._bit_array[byte_index] |= 1 << bit_index

    def _get_bit(self, position: int) -> bool:
        """获取指定位"""
        byte_index = position // 8
        bit_index = position % 8
        return bool(self._bit_array[byte_index] & (1 << bit_index))

    def add(self, url: str) -> bool:
        """添加 URL"""
        if self.contains(url):
            return False

        for h in self._get_hashes(url):
            self._set_bit(h)

        self._count += 1
        return True

    def contains(self, url: str) -> bool:
        """检查 URL 是否可能存在"""
        return all(self._get_bit(h) for h in self._get_hashes(url))

    def __len__(self) -> int:
        return self._count


class PersistentDedup(UrlDedup):
    """
    持久化去重

    每次添加都写入文件，支持断点续爬

    Example:
        ```python
        dedup = PersistentDedup("seen_urls.txt")

        # 自动从文件加载已有 URL
        dedup.add("https://example.com")
        ```
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._seen: set[str] = set()
        self._load()

    def _load(self) -> None:
        """从文件加载"""
        if self._path.exists():
            with self._path.open() as f:
                for line in f:
                    self._seen.add(line.strip())

    def add(self, url: str) -> bool:
        """添加 URL 并持久化"""
        normalized = url.split("#")[0].rstrip("/")
        if normalized in self._seen:
            return False

        self._seen.add(normalized)

        # 追加到文件
        with self._path.open("a") as f:
            f.write(normalized + "\n")

        return True

    def contains(self, url: str) -> bool:
        """检查 URL 是否已存在"""
        normalized = url.split("#")[0].rstrip("/")
        return normalized in self._seen

    def __len__(self) -> int:
        return len(self._seen)


class UrlQueue:
    """
    URL 队列（带去重）

    Example:
        ```python
        queue = UrlQueue()
        queue.add("https://example.com/page1")
        queue.add("https://example.com/page2")

        while url := queue.pop():
            process(url)
        ```
    """

    def __init__(self, dedup: UrlDedup | None = None) -> None:
        self._queue: list[str] = []
        self._dedup = dedup or HashSetDedup()

    def add(self, url: str) -> bool:
        """添加 URL 到队列"""
        if self._dedup.add(url):
            self._queue.append(url)
            return True
        return False

    def add_many(self, urls: list[str]) -> int:
        """批量添加 URL"""
        count = 0
        for url in urls:
            if self.add(url):
                count += 1
        return count

    def pop(self) -> str | None:
        """获取下一个 URL"""
        if self._queue:
            return self._queue.pop(0)
        return None

    def __len__(self) -> int:
        """队列中剩余的 URL 数量"""
        return len(self._queue)

    @property
    def seen_count(self) -> int:
        """已处理的 URL 数量"""
        return len(self._dedup)

    @property
    def is_empty(self) -> bool:
        """队列是否为空"""
        return len(self._queue) == 0

