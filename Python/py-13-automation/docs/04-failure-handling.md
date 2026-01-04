# 失败处理

> 批处理不可能 100% 成功，关键是如何优雅地处理失败

## 1. 失败是常态

批处理中的常见失败原因：

- 📁 文件不存在或被占用
- 🔒 权限不足
- 💾 磁盘空间不足
- 🌐 网络中断
- ⏱️ 操作超时
- 🐛 程序 Bug

## 2. 失败处理策略

### 2.1 停止 vs 继续

```python
from enum import Enum
from dataclasses import dataclass


class FailurePolicy(Enum):
    """失败策略"""
    STOP_ON_FIRST = "stop"       # 遇到第一个失败就停止
    CONTINUE = "continue"        # 继续执行，最后汇总
    STOP_ON_THRESHOLD = "threshold"  # 失败超过阈值停止


@dataclass
class ExecutionConfig:
    """执行配置"""
    failure_policy: FailurePolicy = FailurePolicy.CONTINUE
    failure_threshold: int = 10  # 用于 STOP_ON_THRESHOLD
    max_retries: int = 3
    retry_delay: float = 1.0  # 秒
```

### 2.2 实现失败策略

```python
from typing import Callable, TypeVar
import time

T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor:
    """批处理器"""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.results: list[tuple[T, R | None, Exception | None]] = []
        self.failure_count = 0

    def process(
        self,
        items: list[T],
        processor: Callable[[T], R],
    ) -> list[tuple[T, R | None, Exception | None]]:
        """处理项目列表"""
        for item in items:
            result, error = self._process_one(item, processor)
            self.results.append((item, result, error))

            if error:
                self.failure_count += 1

                # 检查是否需要停止
                if self._should_stop():
                    break

        return self.results

    def _process_one(
        self,
        item: T,
        processor: Callable[[T], R],
    ) -> tuple[R | None, Exception | None]:
        """处理单个项目（带重试）"""
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                result = processor(item)
                return result, None
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        return None, last_error

    def _should_stop(self) -> bool:
        """检查是否应该停止"""
        match self.config.failure_policy:
            case FailurePolicy.STOP_ON_FIRST:
                return self.failure_count >= 1
            case FailurePolicy.STOP_ON_THRESHOLD:
                return self.failure_count >= self.config.failure_threshold
            case FailurePolicy.CONTINUE:
                return False
        return False

    def get_summary(self) -> dict:
        """获取处理摘要"""
        success = sum(1 for _, _, e in self.results if e is None)
        failed = sum(1 for _, _, e in self.results if e is not None)

        return {
            "total": len(self.results),
            "success": success,
            "failed": failed,
            "success_rate": f"{success / len(self.results) * 100:.1f}%" if self.results else "N/A",
        }
```

## 3. 重试策略

### 3.1 简单重试

```python
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def simple_retry(
    fn: Callable[[], T],
    max_attempts: int = 3,
    delay: float = 1.0,
) -> T:
    """简单重试"""
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(delay)

    raise last_error  # type: ignore
```

### 3.2 使用 tenacity 库

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

logger = logging.getLogger(__name__)


# 装饰器方式
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((IOError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def robust_file_operation(src: Path, dst: Path) -> None:
    """带自动重试的文件操作"""
    shutil.copy2(src, dst)


# 或者创建可重用的重试器
from tenacity import Retrying


def create_retrier(max_attempts: int = 3) -> Retrying:
    """创建重试器"""
    return Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((IOError, OSError, TimeoutError)),
    )


# 使用
retrier = create_retrier(3)

for attempt in retrier:
    with attempt:
        shutil.move(src, dst)
```

### 3.3 区分可重试错误

```python
from dataclasses import dataclass


@dataclass
class ErrorClassification:
    """错误分类"""
    retryable: bool
    category: str
    message: str


def classify_error(error: Exception) -> ErrorClassification:
    """分类错误"""
    error_type = type(error).__name__

    # 可重试错误
    retryable_errors = {
        "TimeoutError": ("timeout", "操作超时"),
        "ConnectionError": ("network", "网络连接失败"),
        "TemporaryError": ("temporary", "临时错误"),
        "IOError": ("io", "IO 错误"),
    }

    # 不可重试错误
    permanent_errors = {
        "FileNotFoundError": ("not_found", "文件不存在"),
        "PermissionError": ("permission", "权限不足"),
        "IsADirectoryError": ("type_error", "类型错误：是目录"),
        "NotADirectoryError": ("type_error", "类型错误：不是目录"),
        "FileExistsError": ("exists", "文件已存在"),
    }

    if error_type in retryable_errors:
        category, msg = retryable_errors[error_type]
        return ErrorClassification(True, category, msg)

    if error_type in permanent_errors:
        category, msg = permanent_errors[error_type]
        return ErrorClassification(False, category, msg)

    # 未知错误默认不重试
    return ErrorClassification(False, "unknown", str(error))
```

## 4. 失败汇总报告

### 4.1 收集失败信息

```python
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class FailureRecord:
    """失败记录"""
    operation_index: int
    operation_type: str
    source: Path
    target: Path | None
    error_type: str
    error_message: str
    attempts: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FailureSummary:
    """失败汇总"""
    failures: list[FailureRecord] = field(default_factory=list)

    def add(self, record: FailureRecord) -> None:
        self.failures.append(record)

    def by_error_type(self) -> dict[str, list[FailureRecord]]:
        """按错误类型分组"""
        result: dict[str, list[FailureRecord]] = {}
        for f in self.failures:
            result.setdefault(f.error_type, []).append(f)
        return result

    def generate_report(self) -> str:
        """生成报告"""
        lines = [
            "=" * 60,
            "失败汇总报告",
            f"生成时间: {datetime.now().isoformat()}",
            "=" * 60,
            "",
            f"总失败数: {len(self.failures)}",
            "",
        ]

        # 按错误类型统计
        by_type = self.by_error_type()
        lines.append("按错误类型统计:")
        for error_type, records in by_type.items():
            lines.append(f"  - {error_type}: {len(records)} 个")

        lines.append("")
        lines.append("-" * 60)
        lines.append("详细失败列表:")
        lines.append("")

        for i, f in enumerate(self.failures, 1):
            lines.extend([
                f"[{i}] 操作 #{f.operation_index}",
                f"    类型: {f.operation_type}",
                f"    源: {f.source}",
                f"    目标: {f.target or 'N/A'}",
                f"    错误: {f.error_type} - {f.error_message}",
                f"    尝试次数: {f.attempts}",
                "",
            ])

        return "\n".join(lines)
```

### 4.2 保存失败列表用于重试

```python
import json


def save_failures_for_retry(
    failures: list[FailureRecord],
    output_file: Path,
) -> None:
    """保存失败列表，便于后续重试"""
    data = {
        "created_at": datetime.now().isoformat(),
        "total_failures": len(failures),
        "failures": [
            {
                "operation_index": f.operation_index,
                "operation_type": f.operation_type,
                "source": str(f.source),
                "target": str(f.target) if f.target else None,
                "error_type": f.error_type,
                "error_message": f.error_message,
                "attempts": f.attempts,
            }
            for f in failures
        ]
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_failures_for_retry(input_file: Path) -> list[dict]:
    """加载失败列表用于重试"""
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)
    return data["failures"]
```

## 5. 回滚机制

### 5.1 回滚设计原则

```
执行前:  A → B → C → D
执行到:  A ✓ → B ✓ → C ✗ (失败)
回滚:    A ← B ← (从 B 回滚到初始状态)
```

关键：**记录反向操作**

### 5.2 实现回滚

```python
from dataclasses import dataclass
from typing import Protocol


class Reversible(Protocol):
    """可回滚操作协议"""

    def execute(self) -> bool:
        """执行操作"""
        ...

    def rollback(self) -> bool:
        """回滚操作"""
        ...


@dataclass
class ReversibleRename:
    """可回滚的重命名"""
    source: Path
    target: Path
    _executed: bool = False

    def execute(self) -> bool:
        try:
            self.source.rename(self.target)
            self._executed = True
            return True
        except Exception:
            return False

    def rollback(self) -> bool:
        if not self._executed:
            return True  # 没执行过，不需要回滚

        try:
            self.target.rename(self.source)
            self._executed = False
            return True
        except Exception:
            return False


@dataclass
class ReversibleMove:
    """可回滚的移动"""
    source: Path
    target: Path
    _executed: bool = False

    def execute(self) -> bool:
        try:
            self.target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(self.source, self.target)
            self._executed = True
            return True
        except Exception:
            return False

    def rollback(self) -> bool:
        if not self._executed:
            return True

        try:
            self.source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(self.target, self.source)
            self._executed = False
            return True
        except Exception:
            return False


@dataclass
class ReversibleCopy:
    """可回滚的复制"""
    source: Path
    target: Path
    _executed: bool = False

    def execute(self) -> bool:
        try:
            self.target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.source, self.target)
            self._executed = True
            return True
        except Exception:
            return False

    def rollback(self) -> bool:
        if not self._executed:
            return True

        try:
            self.target.unlink()  # 复制的回滚是删除目标
            self._executed = False
            return True
        except Exception:
            return False


@dataclass
class ReversibleDelete:
    """可回滚的删除（需要备份）"""
    source: Path
    backup_dir: Path
    _backup_path: Path | None = None
    _executed: bool = False

    def execute(self) -> bool:
        try:
            # 先备份到临时位置
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            self._backup_path = self.backup_dir / f"{self.source.name}.backup"
            shutil.move(self.source, self._backup_path)
            self._executed = True
            return True
        except Exception:
            return False

    def rollback(self) -> bool:
        if not self._executed or self._backup_path is None:
            return True

        try:
            shutil.move(self._backup_path, self.source)
            self._executed = False
            return True
        except Exception:
            return False

    def commit(self) -> None:
        """确认删除，清理备份"""
        if self._backup_path and self._backup_path.exists():
            self._backup_path.unlink()
            self._backup_path = None
```

### 5.3 事务执行器

```python
class TransactionalBatch:
    """事务性批处理：全部成功或全部回滚"""

    def __init__(self):
        self.executed: list[Reversible] = []

    def execute(self, operations: list[Reversible]) -> bool:
        """执行所有操作"""
        for op in operations:
            if op.execute():
                self.executed.append(op)
            else:
                # 失败，回滚已执行的操作
                self.rollback()
                return False

        return True

    def rollback(self) -> None:
        """回滚所有已执行的操作（逆序）"""
        for op in reversed(self.executed):
            if not op.rollback():
                print(f"警告：回滚失败: {op}")

        self.executed.clear()


# 使用示例
batch = TransactionalBatch()
operations = [
    ReversibleRename(Path("a.txt"), Path("a_new.txt")),
    ReversibleMove(Path("b.txt"), Path("archive/b.txt")),
    ReversibleCopy(Path("c.txt"), Path("backup/c.txt")),
]

if batch.execute(operations):
    print("所有操作成功")
else:
    print("操作失败，已回滚")
```

### 5.4 回滚日志

```python
@dataclass
class RollbackEntry:
    """回滚日志条目"""
    operation_index: int
    operation_type: str
    original_state: dict
    new_state: dict
    timestamp: datetime = field(default_factory=datetime.now)


class RollbackLog:
    """回滚日志"""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.entries: list[RollbackEntry] = []

    def record(self, entry: RollbackEntry) -> None:
        """记录回滚点"""
        self.entries.append(entry)
        self._persist()

    def _persist(self) -> None:
        """持久化日志"""
        data = [
            {
                "operation_index": e.operation_index,
                "operation_type": e.operation_type,
                "original_state": e.original_state,
                "new_state": e.new_state,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in self.entries
        ]

        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def rollback_to(self, index: int) -> list[RollbackEntry]:
        """获取需要回滚的条目（从最新到指定索引）"""
        return list(reversed(self.entries[index:]))
```

## 6. 完整失败处理流程

```python
def robust_batch_process(
    operations: list[Operation],
    config: ExecutionConfig,
    state_file: Path,
    rollback_log_file: Path,
) -> dict:
    """健壮的批处理流程"""

    # 初始化
    state_mgr = StateManager(state_file)
    rollback_log = RollbackLog(rollback_log_file)
    failure_summary = FailureSummary()

    # 加载或初始化状态
    state = state_mgr.load()
    if state is None:
        state = state_mgr.init_state("batch", len(operations))

    # 获取待处理任务
    pending = state_mgr.get_pending_indices()

    for idx in pending:
        op = operations[idx]
        state_mgr.mark_started(idx)

        # 记录回滚点
        rollback_log.record(RollbackEntry(
            operation_index=idx,
            operation_type=op.op_type.value,
            original_state={"source": str(op.source), "exists": op.source.exists()},
            new_state={"target": str(op.target) if op.target else None},
        ))

        # 执行（带重试）
        success = False
        last_error = None

        for attempt in range(config.max_retries):
            try:
                execute_operation(op)
                success = True
                break
            except Exception as e:
                last_error = e
                classification = classify_error(e)

                if not classification.retryable:
                    break  # 不可重试错误，立即失败

                time.sleep(config.retry_delay)

        if success:
            state_mgr.mark_completed(idx)
        else:
            state_mgr.mark_failed(idx, str(last_error))
            failure_summary.add(FailureRecord(
                operation_index=idx,
                operation_type=op.op_type.value,
                source=op.source,
                target=op.target,
                error_type=type(last_error).__name__,
                error_message=str(last_error),
                attempts=config.max_retries,
            ))

            # 检查是否需要停止
            if config.failure_policy == FailurePolicy.STOP_ON_FIRST:
                break
            if (config.failure_policy == FailurePolicy.STOP_ON_THRESHOLD
                and failure_summary.failures >= config.failure_threshold):
                break

    # 输出结果
    final_state = state_mgr.load()
    result = {
        "completed": final_state.completed_count,
        "failed": final_state.failed_count,
        "pending": final_state.pending_count,
    }

    if failure_summary.failures:
        print(failure_summary.generate_report())
        save_failures_for_retry(
            failure_summary.failures,
            state_file.parent / "failures.json"
        )

    return result
```

## 小结

| 机制 | 作用 |
|------|------|
| 失败策略 | 决定失败后继续还是停止 |
| 重试机制 | 自动重试临时性错误 |
| 错误分类 | 区分可重试和不可重试错误 |
| 失败汇总 | 收集和报告所有失败 |
| 回滚机制 | 撤销已执行的操作 |

下一节我们将学习常见的自动化任务实现。

