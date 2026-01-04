# 可恢复机制

> 批处理中途失败不可怕，可怕的是无法恢复

## 1. 为什么需要可恢复？

批处理任务常见问题：

```python
# 处理 10000 个文件
for i, file in enumerate(files):
    process(file)  # 第 5000 个失败了
    # 现在怎么办？从头开始？
```

可恢复机制解决：
- 🔄 **断点续跑**：从失败点继续，不重复处理
- 📊 **进度追踪**：知道完成了多少，还剩多少
- 🔍 **状态审计**：记录每个操作的结果

## 2. State 状态管理

### 2.1 状态数据结构

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import json


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"      # 待执行
    RUNNING = "running"      # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    SKIPPED = "skipped"      # 跳过


@dataclass
class TaskState:
    """单个任务状态"""
    index: int
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    attempts: int = 0


@dataclass
class BatchState:
    """批处理状态"""
    batch_id: str
    created_at: datetime
    updated_at: datetime
    total_tasks: int
    tasks: dict[int, TaskState] = field(default_factory=dict)

    @property
    def pending_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)

    @property
    def progress(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        done = self.completed_count + self.failed_count
        return done / self.total_tasks * 100
```

### 2.2 状态持久化

```python
class StateManager:
    """状态管理器"""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._state: BatchState | None = None

    def init_state(self, batch_id: str, total_tasks: int) -> BatchState:
        """初始化状态"""
        now = datetime.now()
        self._state = BatchState(
            batch_id=batch_id,
            created_at=now,
            updated_at=now,
            total_tasks=total_tasks,
            tasks={i: TaskState(index=i) for i in range(total_tasks)}
        )
        self.save()
        return self._state

    def load(self) -> BatchState | None:
        """加载状态"""
        if not self.state_file.exists():
            return None

        with open(self.state_file, encoding="utf-8") as f:
            data = json.load(f)

        self._state = BatchState(
            batch_id=data["batch_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            total_tasks=data["total_tasks"],
            tasks={
                int(k): TaskState(
                    index=int(k),
                    status=TaskStatus(v["status"]),
                    started_at=datetime.fromisoformat(v["started_at"]) if v.get("started_at") else None,
                    completed_at=datetime.fromisoformat(v["completed_at"]) if v.get("completed_at") else None,
                    error=v.get("error"),
                    attempts=v.get("attempts", 0),
                )
                for k, v in data["tasks"].items()
            }
        )
        return self._state

    def save(self) -> None:
        """保存状态"""
        if self._state is None:
            return

        self._state.updated_at = datetime.now()

        data = {
            "batch_id": self._state.batch_id,
            "created_at": self._state.created_at.isoformat(),
            "updated_at": self._state.updated_at.isoformat(),
            "total_tasks": self._state.total_tasks,
            "tasks": {
                str(k): {
                    "status": v.status.value,
                    "started_at": v.started_at.isoformat() if v.started_at else None,
                    "completed_at": v.completed_at.isoformat() if v.completed_at else None,
                    "error": v.error,
                    "attempts": v.attempts,
                }
                for k, v in self._state.tasks.items()
            }
        }

        # 原子写入：先写临时文件，再重命名
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_file.rename(self.state_file)

    def mark_started(self, index: int) -> None:
        """标记任务开始"""
        if self._state:
            task = self._state.tasks[index]
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            task.attempts += 1
            self.save()

    def mark_completed(self, index: int) -> None:
        """标记任务完成"""
        if self._state:
            task = self._state.tasks[index]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            self.save()

    def mark_failed(self, index: int, error: str) -> None:
        """标记任务失败"""
        if self._state:
            task = self._state.tasks[index]
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = error
            self.save()

    def get_pending_indices(self) -> list[int]:
        """获取待执行任务索引"""
        if self._state is None:
            return []
        return [
            i for i, t in self._state.tasks.items()
            if t.status in (TaskStatus.PENDING, TaskStatus.FAILED)
        ]
```

## 3. 断点续跑执行器

```python
from typing import Callable, TypeVar
from tqdm import tqdm

T = TypeVar("T")


class ResumableExecutor:
    """可恢复执行器"""

    def __init__(
        self,
        state_manager: StateManager,
        max_retries: int = 3,
        continue_on_error: bool = True,
    ):
        self.state_manager = state_manager
        self.max_retries = max_retries
        self.continue_on_error = continue_on_error

    def execute(
        self,
        tasks: list[T],
        processor: Callable[[T], None],
        batch_id: str | None = None,
    ) -> BatchState:
        """执行任务列表"""
        # 尝试加载现有状态
        state = self.state_manager.load()

        if state is None:
            # 新批次
            batch_id = batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            state = self.state_manager.init_state(batch_id, len(tasks))
            print(f"开始新批次: {batch_id}, 共 {len(tasks)} 个任务")
        else:
            print(f"恢复批次: {state.batch_id}")
            print(f"  已完成: {state.completed_count}")
            print(f"  待执行: {state.pending_count}")
            print(f"  失败: {state.failed_count}")

        # 获取待执行任务
        pending_indices = self.state_manager.get_pending_indices()

        # 执行
        with tqdm(total=len(pending_indices), desc="Processing") as pbar:
            for idx in pending_indices:
                task = tasks[idx]
                task_state = state.tasks[idx]

                # 检查重试次数
                if task_state.attempts >= self.max_retries:
                    print(f"\n跳过任务 {idx}: 已达最大重试次数")
                    continue

                self.state_manager.mark_started(idx)

                try:
                    processor(task)
                    self.state_manager.mark_completed(idx)
                except Exception as e:
                    self.state_manager.mark_failed(idx, str(e))
                    if not self.continue_on_error:
                        print(f"\n任务 {idx} 失败，停止执行: {e}")
                        break

                pbar.update(1)

        # 返回最终状态
        return self.state_manager.load()  # type: ignore
```

## 4. 幂等操作设计

### 4.1 什么是幂等？

**幂等操作**：执行一次和执行多次效果相同。

```python
# ✅ 幂等操作
def ensure_file_exists(path: Path, content: str) -> None:
    """确保文件存在且内容正确"""
    if path.exists() and path.read_text() == content:
        return  # 已经是期望状态
    path.write_text(content)

# ❌ 非幂等操作
def append_to_file(path: Path, content: str) -> None:
    """追加内容到文件 - 多次执行会重复追加"""
    with open(path, "a") as f:
        f.write(content)
```

### 4.2 设计幂等操作

```python
from pathlib import Path
import shutil
import hashlib


def idempotent_rename(src: Path, dst: Path) -> bool:
    """
    幂等重命名

    可能的状态：
    1. src 存在, dst 不存在 → 执行重命名
    2. src 不存在, dst 存在 → 已完成，跳过
    3. src 存在, dst 存在 → 冲突，失败
    4. src 不存在, dst 不存在 → 源文件丢失，失败
    """
    if not src.exists() and dst.exists():
        return True  # 状态 2: 已完成

    if src.exists() and dst.exists():
        raise FileExistsError(f"目标已存在: {dst}")  # 状态 3

    if not src.exists() and not dst.exists():
        raise FileNotFoundError(f"源文件不存在: {src}")  # 状态 4

    # 状态 1: 执行重命名
    src.rename(dst)
    return True


def idempotent_copy(src: Path, dst: Path) -> bool:
    """
    幂等复制（基于内容哈希）
    """
    if dst.exists():
        # 检查内容是否相同
        src_hash = hashlib.md5(src.read_bytes()).hexdigest()
        dst_hash = hashlib.md5(dst.read_bytes()).hexdigest()
        if src_hash == dst_hash:
            return True  # 已完成

    shutil.copy2(src, dst)
    return True


def idempotent_mkdir(path: Path) -> bool:
    """幂等创建目录"""
    path.mkdir(parents=True, exist_ok=True)  # exist_ok=True 就是幂等的
    return True


def idempotent_delete(path: Path) -> bool:
    """幂等删除"""
    if not path.exists():
        return True  # 已删除

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True
```

### 4.3 带状态检查的操作

```python
from dataclasses import dataclass
from enum import Enum


class OperationOutcome(Enum):
    EXECUTED = "executed"    # 执行了操作
    SKIPPED = "skipped"      # 跳过（已完成）
    FAILED = "failed"        # 失败


@dataclass
class IdempotentResult:
    outcome: OperationOutcome
    message: str


class IdempotentOperations:
    """幂等操作集合"""

    @staticmethod
    def move_file(src: Path, dst: Path) -> IdempotentResult:
        """幂等移动文件"""
        # 检查目标状态
        if dst.exists() and not src.exists():
            return IdempotentResult(
                OperationOutcome.SKIPPED,
                f"已完成: {dst} 存在"
            )

        if not src.exists():
            return IdempotentResult(
                OperationOutcome.FAILED,
                f"源文件不存在: {src}"
            )

        if dst.exists():
            return IdempotentResult(
                OperationOutcome.FAILED,
                f"目标已存在: {dst}"
            )

        # 确保目标目录存在
        dst.parent.mkdir(parents=True, exist_ok=True)

        # 执行移动
        shutil.move(src, dst)

        return IdempotentResult(
            OperationOutcome.EXECUTED,
            f"已移动: {src} → {dst}"
        )
```

## 5. 事务思维

### 5.1 操作日志（WAL）

Write-Ahead Logging：先记录要做什么，再执行。

```python
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class LogEntry:
    """操作日志条目"""
    timestamp: datetime
    operation_index: int
    operation_type: str
    source: str
    target: str | None
    status: str  # "planned" | "started" | "completed" | "failed"
    error: str | None = None


class OperationLog:
    """操作日志管理"""

    def __init__(self, log_file: Path):
        self.log_file = log_file

    def append(self, entry: LogEntry) -> None:
        """追加日志条目"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            data = {
                "timestamp": entry.timestamp.isoformat(),
                "operation_index": entry.operation_index,
                "operation_type": entry.operation_type,
                "source": entry.source,
                "target": entry.target,
                "status": entry.status,
                "error": entry.error,
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def read_all(self) -> list[LogEntry]:
        """读取所有日志"""
        if not self.log_file.exists():
            return []

        entries = []
        with open(self.log_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entries.append(LogEntry(
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        operation_index=data["operation_index"],
                        operation_type=data["operation_type"],
                        source=data["source"],
                        target=data["target"],
                        status=data["status"],
                        error=data.get("error"),
                    ))
        return entries

    def get_incomplete_operations(self) -> list[int]:
        """获取未完成的操作索引"""
        entries = self.read_all()

        # 按操作索引分组，找最新状态
        latest_status: dict[int, str] = {}
        for entry in entries:
            latest_status[entry.operation_index] = entry.status

        # 返回未完成的
        return [
            idx for idx, status in latest_status.items()
            if status not in ("completed", "failed")
        ]
```

### 5.2 两阶段执行

```python
class TransactionalExecutor:
    """事务性执行器"""

    def __init__(self, log: OperationLog):
        self.log = log

    def execute_with_logging(
        self,
        operation: Operation,
        index: int,
    ) -> bool:
        """带日志的执行"""
        now = datetime.now()

        # 阶段 1: 记录计划
        self.log.append(LogEntry(
            timestamp=now,
            operation_index=index,
            operation_type=operation.op_type.value,
            source=str(operation.source),
            target=str(operation.target) if operation.target else None,
            status="planned",
        ))

        # 阶段 2: 开始执行
        self.log.append(LogEntry(
            timestamp=datetime.now(),
            operation_index=index,
            operation_type=operation.op_type.value,
            source=str(operation.source),
            target=str(operation.target) if operation.target else None,
            status="started",
        ))

        try:
            # 执行操作
            self._do_execute(operation)

            # 阶段 3: 记录完成
            self.log.append(LogEntry(
                timestamp=datetime.now(),
                operation_index=index,
                operation_type=operation.op_type.value,
                source=str(operation.source),
                target=str(operation.target) if operation.target else None,
                status="completed",
            ))
            return True

        except Exception as e:
            # 记录失败
            self.log.append(LogEntry(
                timestamp=datetime.now(),
                operation_index=index,
                operation_type=operation.op_type.value,
                source=str(operation.source),
                target=str(operation.target) if operation.target else None,
                status="failed",
                error=str(e),
            ))
            return False

    def _do_execute(self, operation: Operation) -> None:
        """实际执行操作"""
        match operation.op_type:
            case OpType.RENAME:
                operation.source.rename(operation.target)
            case OpType.MOVE:
                shutil.move(operation.source, operation.target)
            case OpType.COPY:
                shutil.copy2(operation.source, operation.target)
            case OpType.DELETE:
                if operation.source.is_dir():
                    shutil.rmtree(operation.source)
                else:
                    operation.source.unlink()
            case OpType.MKDIR:
                operation.source.mkdir(parents=True, exist_ok=True)
```

## 6. 使用示例

```python
from pathlib import Path

# 场景：处理 1000 个文件，中途可能失败

def process_file(file_path: Path) -> None:
    """模拟文件处理"""
    # 实际处理逻辑
    pass

# 设置状态管理
state_mgr = StateManager(Path("batch_state.json"))
executor = ResumableExecutor(
    state_manager=state_mgr,
    max_retries=3,
    continue_on_error=True,
)

# 获取文件列表
files = list(Path("./data").glob("*.txt"))

# 执行（自动支持断点续跑）
final_state = executor.execute(
    tasks=files,
    processor=process_file,
    batch_id="file_processing_20240101",
)

# 输出结果
print(f"\n处理完成!")
print(f"  成功: {final_state.completed_count}")
print(f"  失败: {final_state.failed_count}")
print(f"  进度: {final_state.progress:.1f}%")

# 如果有失败，可以再次运行来重试
if final_state.failed_count > 0:
    print("\n存在失败任务，可重新运行脚本进行重试")
```

## 7. 最佳实践

### 7.1 状态文件

- ✅ 使用原子写入（先写临时文件，再重命名）
- ✅ 每个操作后保存状态
- ✅ 包含足够的信息用于恢复
- ✅ 人类可读的 JSON 格式

### 7.2 幂等设计

- ✅ 检查目标状态而非假设初始状态
- ✅ 相同输入多次执行结果相同
- ✅ 操作前后都能正确判断完成状态

### 7.3 错误处理

- ✅ 区分可重试和不可重试错误
- ✅ 记录详细的错误信息
- ✅ 设置最大重试次数
- ✅ 允许选择失败后继续还是停止

## 小结

| 机制 | 作用 |
|------|------|
| 状态持久化 | 记录进度，支持恢复 |
| 幂等操作 | 多次执行结果相同 |
| 操作日志 | 追踪执行历史 |
| 原子写入 | 防止状态文件损坏 |

下一节我们将学习 dry-run 模式的实现。

