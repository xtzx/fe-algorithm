# 练习题

> 10 道练习题，从基础到进阶

## 练习 1：基础重命名器（⭐）

实现一个简单的批量重命名工具：

```python
def simple_rename(
    directory: Path,
    old_ext: str,
    new_ext: str,
    dry_run: bool = True,
) -> list[tuple[str, str]]:
    """
    批量修改文件扩展名

    Args:
        directory: 目标目录
        old_ext: 原扩展名（如 ".txt"）
        new_ext: 新扩展名（如 ".md"）
        dry_run: 预览模式

    Returns:
        [(old_name, new_name), ...]
    """
    # TODO: 实现
    pass
```

**测试用例**：
```python
# 准备测试文件
# a.txt, b.txt, c.md

result = simple_rename(Path("./test"), ".txt", ".md", dry_run=True)
# 输出: [("a.txt", "a.md"), ("b.txt", "b.md")]
```

---

## 练习 2：状态保存器（⭐）

实现状态的 JSON 序列化和反序列化：

```python
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class TaskProgress:
    total: int
    completed: list[int]
    failed: list[int]
    current: int | None
    started_at: datetime

    def to_json(self, file_path: Path) -> None:
        """保存到 JSON 文件"""
        # TODO: 实现（注意 datetime 的序列化）
        pass

    @classmethod
    def from_json(cls, file_path: Path) -> "TaskProgress":
        """从 JSON 文件加载"""
        # TODO: 实现
        pass
```

---

## 练习 3：Dry-run 装饰器（⭐⭐）

创建一个装饰器，让任何函数支持 dry-run 模式：

```python
def with_dry_run(func):
    """
    装饰器：为函数添加 dry_run 参数

    当 dry_run=True 时，不执行函数，只打印要执行的内容
    """
    # TODO: 实现
    pass


# 使用示例
@with_dry_run
def delete_file(path: Path) -> None:
    path.unlink()

@with_dry_run
def create_directory(path: Path) -> None:
    path.mkdir(parents=True)

# 调用
delete_file(Path("test.txt"), dry_run=True)
# 输出: [DRY-RUN] Would execute: delete_file(path=test.txt)
```

---

## 练习 4：幂等移动操作（⭐⭐）

实现一个幂等的文件移动函数：

```python
from enum import Enum

class MoveResult(Enum):
    MOVED = "moved"           # 执行了移动
    ALREADY_DONE = "already"  # 已经完成
    SOURCE_MISSING = "missing"
    TARGET_EXISTS = "conflict"


def idempotent_move(src: Path, dst: Path) -> MoveResult:
    """
    幂等移动文件

    满足以下条件：
    1. 如果 dst 存在且 src 不存在 → ALREADY_DONE
    2. 如果 src 存在且 dst 不存在 → 执行移动 → MOVED
    3. 如果两者都存在 → TARGET_EXISTS
    4. 如果两者都不存在 → SOURCE_MISSING
    """
    # TODO: 实现
    pass
```

---

## 练习 5：文件分类器（⭐⭐）

实现一个可扩展的文件分类器：

```python
from abc import ABC, abstractmethod

class FileClassifier(ABC):
    """文件分类器抽象类"""

    @abstractmethod
    def classify(self, file_path: Path) -> str:
        """返回文件的分类"""
        pass


class ExtensionClassifier(FileClassifier):
    """按扩展名分类"""
    # TODO: 实现


class SizeClassifier(FileClassifier):
    """按大小分类"""
    # TODO: 实现


class DateClassifier(FileClassifier):
    """按日期分类"""
    # TODO: 实现


def organize_files(
    source: Path,
    classifier: FileClassifier,
    dry_run: bool = True,
) -> dict[str, list[Path]]:
    """使用分类器组织文件"""
    # TODO: 实现
    pass
```

---

## 练习 6：重试管理器（⭐⭐⭐）

实现一个支持指数退避的重试管理器：

```python
from typing import Callable, TypeVar
import time
import random

T = TypeVar("T")

class RetryManager:
    """
    重试管理器

    支持：
    - 最大重试次数
    - 指数退避
    - 抖动（jitter）
    - 可重试异常过滤
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: tuple[type[Exception], ...] = (Exception,),
    ):
        # TODO: 初始化
        pass

    def execute(self, func: Callable[[], T]) -> T:
        """
        执行函数，失败时自动重试

        Returns:
            函数返回值

        Raises:
            最后一次失败的异常
        """
        # TODO: 实现
        pass

    def _calculate_delay(self, attempt: int) -> float:
        """计算下次重试的延迟时间"""
        # TODO: 实现指数退避 + 抖动
        pass


# 测试
manager = RetryManager(max_attempts=5, retry_on=(ConnectionError,))

def unstable_operation():
    if random.random() < 0.7:
        raise ConnectionError("Network error")
    return "success"

result = manager.execute(unstable_operation)
```

---

## 练习 7：回滚日志（⭐⭐⭐）

实现一个完整的回滚日志系统：

```python
@dataclass
class RollbackAction:
    """回滚动作"""
    forward_action: str     # 正向操作描述
    rollback_action: str    # 回滚操作描述
    forward_fn: Callable[[], bool]   # 正向执行函数
    rollback_fn: Callable[[], bool]  # 回滚执行函数


class RollbackManager:
    """
    回滚管理器

    功能：
    1. 记录已执行的操作
    2. 支持回滚到任意检查点
    3. 持久化回滚信息
    """

    def __init__(self, log_file: Path):
        # TODO: 实现
        pass

    def execute(self, action: RollbackAction) -> bool:
        """执行操作并记录"""
        # TODO: 实现
        pass

    def rollback_all(self) -> bool:
        """回滚所有已执行的操作"""
        # TODO: 实现
        pass

    def rollback_to(self, checkpoint: int) -> bool:
        """回滚到指定检查点"""
        # TODO: 实现
        pass

    def create_checkpoint(self) -> int:
        """创建检查点"""
        # TODO: 实现
        pass
```

---

## 练习 8：批处理报告生成器（⭐⭐⭐）

实现一个批处理结果报告生成器：

```python
@dataclass
class BatchResult:
    """批处理结果"""
    operation: str
    source: str
    target: str | None
    success: bool
    error: str | None
    duration_ms: float
    timestamp: datetime


class ReportGenerator:
    """
    报告生成器

    支持多种输出格式：
    - 文本报告
    - JSON 报告
    - Markdown 报告
    - HTML 报告
    """

    def __init__(self, results: list[BatchResult]):
        self.results = results

    def generate_text(self) -> str:
        """生成文本报告"""
        # TODO: 实现
        pass

    def generate_json(self) -> dict:
        """生成 JSON 报告"""
        # TODO: 实现
        pass

    def generate_markdown(self) -> str:
        """生成 Markdown 报告"""
        # TODO: 实现表格格式
        pass

    def generate_html(self) -> str:
        """生成 HTML 报告"""
        # TODO: 实现带样式的 HTML
        pass
```

---

## 练习 9：并行批处理器（⭐⭐⭐⭐）

实现一个支持并行执行的批处理器：

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class ParallelBatchProcessor:
    """
    并行批处理器

    特性：
    - 多线程并行执行
    - 线程安全的状态管理
    - 进度回调
    - 可配置的并发数
    """

    def __init__(
        self,
        max_workers: int = 4,
        state_file: Path | None = None,
    ):
        # TODO: 实现
        pass

    def process(
        self,
        items: list[T],
        processor: Callable[[T], R],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[tuple[T, R | None, Exception | None]]:
        """
        并行处理

        Args:
            items: 待处理项目
            processor: 处理函数
            on_progress: 进度回调 (completed, total)
        """
        # TODO: 实现
        pass

    def _save_state_thread_safe(self, index: int, status: str) -> None:
        """线程安全地保存状态"""
        # TODO: 实现
        pass
```

---

## 练习 10：完整的文件同步工具（⭐⭐⭐⭐⭐）

实现一个功能完整的文件同步工具：

```python
@dataclass
class SyncOptions:
    """同步选项"""
    delete_extra: bool = False     # 删除目标中多余的文件
    update_only: bool = False      # 只更新已存在的文件
    verify_checksum: bool = True   # 使用校验和验证
    preserve_times: bool = True    # 保留修改时间
    exclude_patterns: list[str] = field(default_factory=list)
    include_patterns: list[str] = field(default_factory=list)


class FileSync:
    """
    文件同步工具

    功能：
    1. 单向同步（源 → 目标）
    2. 支持排除/包含模式
    3. Dry-run 预览
    4. 断点续传
    5. 详细日志
    6. 进度显示
    """

    def __init__(
        self,
        source: Path,
        target: Path,
        options: SyncOptions | None = None,
    ):
        # TODO: 实现
        pass

    def analyze(self) -> dict:
        """
        分析同步需求

        Returns:
            {
                "to_copy": [...],
                "to_update": [...],
                "to_delete": [...],
                "unchanged": [...],
            }
        """
        # TODO: 实现
        pass

    def sync(self, dry_run: bool = True) -> dict:
        """
        执行同步

        Returns:
            执行统计
        """
        # TODO: 实现
        pass

    def _should_include(self, path: Path) -> bool:
        """检查文件是否应该包含"""
        # TODO: 实现
        pass

    def _files_differ(self, src: Path, dst: Path) -> bool:
        """检查两个文件是否不同"""
        # TODO: 实现
        pass


# 使用示例
sync = FileSync(
    source=Path("/data/source"),
    target=Path("/backup/target"),
    options=SyncOptions(
        delete_extra=True,
        exclude_patterns=["*.tmp", ".git/**"],
    ),
)

# 预览
analysis = sync.analyze()
print(f"将复制: {len(analysis['to_copy'])} 个文件")
print(f"将更新: {len(analysis['to_update'])} 个文件")
print(f"将删除: {len(analysis['to_delete'])} 个文件")

# 执行
stats = sync.sync(dry_run=False)
```

---

## 参考答案提示

### 练习 1 提示
使用 `Path.glob()` 和 `Path.rename()`

### 练习 2 提示
`datetime` 需要转换为 ISO 格式字符串

### 练习 3 提示
使用 `functools.wraps` 保留原函数信息

### 练习 4 提示
按照四种状态分别处理

### 练习 5 提示
使用策略模式，不同分类器实现同一接口

### 练习 6 提示
```python
delay = min(base_delay * (exponential_base ** attempt), max_delay)
if jitter:
    delay *= random.uniform(0.5, 1.5)
```

### 练习 7 提示
使用栈结构存储操作，LIFO 顺序回滚

### 练习 8 提示
Markdown 表格格式：`| col1 | col2 |`

### 练习 9 提示
使用 `threading.Lock` 保护共享状态

### 练习 10 提示
组合使用前面所有练习的技术

