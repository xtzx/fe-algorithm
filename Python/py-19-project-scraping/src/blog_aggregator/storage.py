"""
数据存储

支持:
- JSONL 存储
- 状态管理
- 去重与增量更新
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from blog_aggregator.models import Article


class ArticleStorage:
    """
    文章存储

    使用 JSONL 格式存储文章

    Example:
        ```python
        storage = ArticleStorage("data/articles.jsonl")

        # 保存文章
        storage.save(article)

        # 检查是否存在
        if not storage.exists(article.id):
            storage.save(article)

        # 加载所有文章
        articles = storage.load_all()
        ```
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # 加载已有 ID 用于去重
        self._ids: set[str] = set()
        self._load_ids()

    def _load_ids(self) -> None:
        """加载已有 ID"""
        if not self._path.exists():
            return

        with self._path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        self._ids.add(data.get("id", ""))
                    except json.JSONDecodeError:
                        pass

    def exists(self, article_id: str) -> bool:
        """检查文章是否已存在"""
        return article_id in self._ids

    def save(self, article: Article) -> bool:
        """
        保存文章

        Returns:
            True 如果是新文章，False 如果已存在
        """
        if self.exists(article.id):
            return False

        with self._path.open("a", encoding="utf-8") as f:
            data = article.to_dict()
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        self._ids.add(article.id)
        return True

    def save_many(self, articles: list[Article]) -> int:
        """
        批量保存

        Returns:
            新保存的文章数
        """
        count = 0
        with self._path.open("a", encoding="utf-8") as f:
            for article in articles:
                if not self.exists(article.id):
                    data = article.to_dict()
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    self._ids.add(article.id)
                    count += 1
        return count

    def load_all(self) -> list[Article]:
        """加载所有文章"""
        if not self._path.exists():
            return []

        articles = []
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        articles.append(Article.from_dict(data))
                    except Exception:
                        pass
        return articles

    def iter_articles(self) -> Iterator[Article]:
        """迭代文章"""
        if not self._path.exists():
            return

        with self._path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        yield Article.from_dict(data)
                    except Exception:
                        pass

    @property
    def count(self) -> int:
        """文章数量"""
        return len(self._ids)

    def clear(self) -> None:
        """清空存储"""
        if self._path.exists():
            self._path.unlink()
        self._ids.clear()


@dataclass
class CollectState:
    """采集状态"""

    # 最后采集时间（每个源）
    last_collect: dict[str, str] = field(default_factory=dict)
    # 采集统计
    total_collected: int = 0
    total_new: int = 0
    # 错误统计
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_collect": self.last_collect,
            "total_collected": self.total_collected,
            "total_new": self.total_new,
            "errors": self.errors[-100:],  # 只保留最近 100 条错误
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CollectState":
        return cls(
            last_collect=data.get("last_collect", {}),
            total_collected=data.get("total_collected", 0),
            total_new=data.get("total_new", 0),
            errors=data.get("errors", []),
        )


class StateStorage:
    """
    状态存储

    用于增量更新

    Example:
        ```python
        state = StateStorage("data/state.json")

        # 更新采集时间
        state.update_collect_time("dev_to")

        # 获取上次采集时间
        last = state.get_last_collect("dev_to")
        ```
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    def _load(self) -> CollectState:
        """加载状态"""
        if not self._path.exists():
            return CollectState()

        try:
            with self._path.open(encoding="utf-8") as f:
                data = json.load(f)
            return CollectState.from_dict(data)
        except Exception:
            return CollectState()

    def save(self) -> None:
        """保存状态"""
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._state.to_dict(), f, indent=2, ensure_ascii=False)

    def update_collect_time(self, source: str) -> None:
        """更新采集时间"""
        self._state.last_collect[source] = datetime.now().isoformat()
        self.save()

    def get_last_collect(self, source: str) -> datetime | None:
        """获取上次采集时间"""
        time_str = self._state.last_collect.get(source)
        if time_str:
            try:
                return datetime.fromisoformat(time_str)
            except ValueError:
                pass
        return None

    def add_stats(self, collected: int, new: int) -> None:
        """添加统计"""
        self._state.total_collected += collected
        self._state.total_new += new
        self.save()

    def add_error(self, source: str, error: str) -> None:
        """添加错误"""
        self._state.errors.append(
            {
                "source": source,
                "error": error,
                "time": datetime.now().isoformat(),
            }
        )
        self.save()

    @property
    def state(self) -> CollectState:
        return self._state

    def clear(self) -> None:
        """清空状态"""
        self._state = CollectState()
        if self._path.exists():
            self._path.unlink()

