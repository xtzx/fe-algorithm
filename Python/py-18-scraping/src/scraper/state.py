"""
状态管理模块

支持:
- 断点续爬
- 进度保存
- 失败重试队列
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CrawlState:
    """爬虫状态"""

    # 已处理的 URL
    processed_urls: list[str] = field(default_factory=list)
    # 待处理的 URL
    pending_urls: list[str] = field(default_factory=list)
    # 失败的 URL（待重试）
    failed_urls: list[str] = field(default_factory=list)

    # 统计
    total_fetched: int = 0
    total_parsed: int = 0
    total_failed: int = 0
    total_items: int = 0

    # 时间
    started_at: float = 0.0
    last_updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed_urls": self.processed_urls,
            "pending_urls": self.pending_urls,
            "failed_urls": self.failed_urls,
            "total_fetched": self.total_fetched,
            "total_parsed": self.total_parsed,
            "total_failed": self.total_failed,
            "total_items": self.total_items,
            "started_at": self.started_at,
            "last_updated_at": self.last_updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrawlState":
        return cls(
            processed_urls=data.get("processed_urls", []),
            pending_urls=data.get("pending_urls", []),
            failed_urls=data.get("failed_urls", []),
            total_fetched=data.get("total_fetched", 0),
            total_parsed=data.get("total_parsed", 0),
            total_failed=data.get("total_failed", 0),
            total_items=data.get("total_items", 0),
            started_at=data.get("started_at", 0.0),
            last_updated_at=data.get("last_updated_at", 0.0),
        )


class State(ABC):
    """状态管理基类"""

    @abstractmethod
    def load(self) -> CrawlState:
        """加载状态"""
        pass

    @abstractmethod
    def save(self, state: CrawlState) -> None:
        """保存状态"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清除状态"""
        pass


class MemoryState(State):
    """
    内存状态管理

    不持久化，适用于简单场景
    """

    def __init__(self) -> None:
        self._state = CrawlState()

    def load(self) -> CrawlState:
        return self._state

    def save(self, state: CrawlState) -> None:
        self._state = state

    def clear(self) -> None:
        self._state = CrawlState()


class FileState(State):
    """
    文件状态管理

    持久化到 JSON 文件，支持断点续爬

    Example:
        ```python
        state = FileState("crawl_state.json")

        # 加载（如果存在）
        crawl_state = state.load()

        # 保存
        state.save(crawl_state)
        ```
    """

    def __init__(
        self,
        path: str | Path,
        auto_save_interval: int = 100,
    ) -> None:
        """
        初始化文件状态管理

        Args:
            path: 状态文件路径
            auto_save_interval: 自动保存间隔（每 N 次操作）
        """
        self._path = Path(path)
        self._auto_save_interval = auto_save_interval
        self._operation_count = 0

    def load(self) -> CrawlState:
        """从文件加载状态"""
        if not self._path.exists():
            return CrawlState(started_at=time.time())

        with self._path.open(encoding="utf-8") as f:
            data = json.load(f)
        return CrawlState.from_dict(data)

    def save(self, state: CrawlState) -> None:
        """保存状态到文件"""
        state.last_updated_at = time.time()
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)

    def clear(self) -> None:
        """删除状态文件"""
        if self._path.exists():
            self._path.unlink()

    def maybe_save(self, state: CrawlState) -> None:
        """根据操作计数决定是否保存"""
        self._operation_count += 1
        if self._operation_count >= self._auto_save_interval:
            self.save(state)
            self._operation_count = 0


class StateManager:
    """
    状态管理器

    提供便捷的状态操作方法

    Example:
        ```python
        manager = StateManager(FileState("state.json"))

        # 添加待处理 URL
        manager.add_pending("https://example.com")

        # 标记为已处理
        manager.mark_processed("https://example.com")

        # 获取下一个待处理 URL
        url = manager.next_pending()
        ```
    """

    def __init__(self, state: State) -> None:
        self._state_storage = state
        self._state = state.load()
        self._processed_set: set[str] = set(self._state.processed_urls)

    def add_pending(self, url: str) -> bool:
        """添加待处理 URL"""
        if url in self._processed_set:
            return False
        if url in self._state.pending_urls:
            return False
        self._state.pending_urls.append(url)
        return True

    def add_pending_many(self, urls: list[str]) -> int:
        """批量添加待处理 URL"""
        count = 0
        for url in urls:
            if self.add_pending(url):
                count += 1
        return count

    def next_pending(self) -> str | None:
        """获取下一个待处理 URL"""
        if self._state.pending_urls:
            return self._state.pending_urls.pop(0)
        return None

    def mark_processed(self, url: str) -> None:
        """标记 URL 为已处理"""
        self._processed_set.add(url)
        self._state.processed_urls.append(url)
        self._state.total_fetched += 1
        self._maybe_save()

    def mark_failed(self, url: str) -> None:
        """标记 URL 为失败"""
        self._state.failed_urls.append(url)
        self._state.total_failed += 1
        self._maybe_save()

    def retry_failed(self) -> int:
        """将失败的 URL 重新加入队列"""
        count = len(self._state.failed_urls)
        self._state.pending_urls.extend(self._state.failed_urls)
        self._state.failed_urls.clear()
        return count

    def increment_items(self, count: int = 1) -> None:
        """增加已抓取项目数"""
        self._state.total_items += count
        self._maybe_save()

    def is_processed(self, url: str) -> bool:
        """检查 URL 是否已处理"""
        return url in self._processed_set

    def _maybe_save(self) -> None:
        """根据需要保存状态"""
        if isinstance(self._state_storage, FileState):
            self._state_storage.maybe_save(self._state)

    def save(self) -> None:
        """强制保存状态"""
        self._state_storage.save(self._state)

    @property
    def state(self) -> CrawlState:
        """获取当前状态"""
        return self._state

    @property
    def pending_count(self) -> int:
        """待处理 URL 数量"""
        return len(self._state.pending_urls)

    @property
    def processed_count(self) -> int:
        """已处理 URL 数量"""
        return len(self._processed_set)

    @property
    def failed_count(self) -> int:
        """失败 URL 数量"""
        return len(self._state.failed_urls)

    def summary(self) -> dict[str, Any]:
        """返回状态摘要"""
        return {
            "pending": self.pending_count,
            "processed": self.processed_count,
            "failed": self.failed_count,
            "total_items": self._state.total_items,
            "started_at": self._state.started_at,
        }

