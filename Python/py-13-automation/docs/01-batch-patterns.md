# 批处理设计模式

> 好的批处理脚本应该可控、可预测、可恢复

## 1. 为什么需要设计模式？

批处理脚本常见问题：

```python
# ❌ 糟糕的批处理脚本
import os
for f in os.listdir("."):
    if f.endswith(".txt"):
        os.rename(f, f.replace(".txt", ".md"))  # 直接执行，无法预览
        # 失败了怎么办？已经改了一半了...
```

问题：
1. 无法预览将要发生的变更
2. 中途失败无法恢复
3. 无法回滚已执行的操作
4. 没有执行日志

## 2. Planner/Executor 分离模式

核心思想：**计划和执行分离**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Planner   │ ──▶ │    Plan     │ ──▶ │  Executor   │
│  (分析阶段)  │     │  (操作列表)  │     │  (执行阶段)  │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │   Preview   │
                    │  (dry-run)  │
                    └─────────────┘
```

### 2.1 定义操作（Operation）

```python
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from typing import Any


class OpType(Enum):
    """操作类型"""
    RENAME = "rename"
    MOVE = "move"
    COPY = "copy"
    DELETE = "delete"
    MKDIR = "mkdir"


@dataclass
class Operation:
    """单个原子操作"""
    op_type: OpType
    source: Path
    target: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        match self.op_type:
            case OpType.RENAME:
                return f"RENAME: {self.source} → {self.target}"
            case OpType.MOVE:
                return f"MOVE: {self.source} → {self.target}"
            case OpType.COPY:
                return f"COPY: {self.source} → {self.target}"
            case OpType.DELETE:
                return f"DELETE: {self.source}"
            case OpType.MKDIR:
                return f"MKDIR: {self.source}"
```

### 2.2 Planner（计划器）

```python
from pathlib import Path
import re


class RenamePlanner:
    """重命名计划器"""

    def __init__(self, directory: Path):
        self.directory = directory

    def plan_regex_rename(
        self,
        pattern: str,
        replacement: str,
        file_glob: str = "*"
    ) -> list[Operation]:
        """计划正则重命名"""
        operations: list[Operation] = []
        regex = re.compile(pattern)

        for file_path in self.directory.glob(file_glob):
            if not file_path.is_file():
                continue

            old_name = file_path.name
            new_name = regex.sub(replacement, old_name)

            if old_name != new_name:
                operations.append(Operation(
                    op_type=OpType.RENAME,
                    source=file_path,
                    target=file_path.parent / new_name,
                    metadata={"old_name": old_name, "new_name": new_name}
                ))

        return operations

    def plan_sequential_rename(
        self,
        prefix: str,
        file_glob: str = "*",
        start: int = 1,
        width: int = 3
    ) -> list[Operation]:
        """计划序号重命名"""
        operations: list[Operation] = []
        files = sorted(self.directory.glob(file_glob))

        for i, file_path in enumerate(files, start=start):
            if not file_path.is_file():
                continue

            suffix = file_path.suffix
            new_name = f"{prefix}{i:0{width}d}{suffix}"

            operations.append(Operation(
                op_type=OpType.RENAME,
                source=file_path,
                target=file_path.parent / new_name,
            ))

        return operations
```

### 2.3 Executor（执行器）

```python
import shutil
import logging
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """执行结果"""
    operation: Operation
    success: bool
    error: str | None = None


class Executor:
    """操作执行器"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)

    def execute(self, operations: list[Operation]) -> list[ExecutionResult]:
        """执行操作列表"""
        results: list[ExecutionResult] = []

        for op in operations:
            result = self._execute_one(op)
            results.append(result)

            if not result.success:
                self.logger.error(f"Failed: {op} - {result.error}")

        return results

    def _execute_one(self, op: Operation) -> ExecutionResult:
        """执行单个操作"""
        try:
            if self.dry_run:
                self.logger.info(f"[DRY-RUN] Would execute: {op}")
                return ExecutionResult(op, success=True)

            match op.op_type:
                case OpType.RENAME:
                    op.source.rename(op.target)
                case OpType.MOVE:
                    shutil.move(op.source, op.target)
                case OpType.COPY:
                    shutil.copy2(op.source, op.target)
                case OpType.DELETE:
                    if op.source.is_dir():
                        shutil.rmtree(op.source)
                    else:
                        op.source.unlink()
                case OpType.MKDIR:
                    op.source.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Executed: {op}")
            return ExecutionResult(op, success=True)

        except Exception as e:
            return ExecutionResult(op, success=False, error=str(e))
```

## 3. 使用示例

```python
from pathlib import Path

# 1. 创建计划
planner = RenamePlanner(Path("./documents"))
plan = planner.plan_regex_rename(
    pattern=r"report_(\d{4})(\d{2})(\d{2})",
    replacement=r"\1-\2-\3_report",
    file_glob="*.pdf"
)

# 2. 预览变更（dry-run）
print("=== 预览变更 ===")
for op in plan:
    print(f"  {op}")

# 3. 确认后执行
if input("确认执行？(y/n): ").lower() == "y":
    executor = Executor(dry_run=False)
    results = executor.execute(plan)

    # 4. 输出结果
    success = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    print(f"完成: {success} 成功, {failed} 失败")
```

## 4. 原子操作设计

### 4.1 什么是原子操作？

原子操作是**不可分割的最小操作单元**，要么完全成功，要么完全失败。

```python
# ✅ 原子操作示例
def atomic_rename(src: Path, dst: Path) -> bool:
    """原子重命名：要么成功，要么失败，不会有中间状态"""
    try:
        src.rename(dst)  # 文件系统的 rename 是原子的
        return True
    except OSError:
        return False

# ❌ 非原子操作
def non_atomic_copy_and_delete(src: Path, dst: Path) -> bool:
    """非原子操作：可能复制成功但删除失败"""
    shutil.copy2(src, dst)  # 如果这里成功
    src.unlink()            # 但这里失败，就会有两份文件
    return True
```

### 4.2 设计原子操作

```python
@dataclass
class AtomicOperation:
    """原子操作基类"""

    def execute(self) -> bool:
        """执行操作"""
        raise NotImplementedError

    def can_execute(self) -> bool:
        """检查是否可以执行"""
        raise NotImplementedError

    def get_rollback(self) -> "AtomicOperation | None":
        """获取回滚操作"""
        raise NotImplementedError


@dataclass
class AtomicRename(AtomicOperation):
    """原子重命名操作"""
    source: Path
    target: Path

    def execute(self) -> bool:
        try:
            self.source.rename(self.target)
            return True
        except OSError:
            return False

    def can_execute(self) -> bool:
        return self.source.exists() and not self.target.exists()

    def get_rollback(self) -> "AtomicRename":
        # 回滚操作就是反向重命名
        return AtomicRename(source=self.target, target=self.source)
```

## 5. 操作序列化

将计划保存为 JSON，便于：
- 预览和审核
- 断点续跑
- 审计日志

```python
import json
from datetime import datetime


def serialize_plan(operations: list[Operation], output_file: Path) -> None:
    """序列化操作计划"""
    data = {
        "created_at": datetime.now().isoformat(),
        "total_operations": len(operations),
        "operations": [
            {
                "index": i,
                "type": op.op_type.value,
                "source": str(op.source),
                "target": str(op.target) if op.target else None,
                "metadata": op.metadata,
            }
            for i, op in enumerate(operations)
        ]
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def deserialize_plan(input_file: Path) -> list[Operation]:
    """反序列化操作计划"""
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    operations = []
    for op_data in data["operations"]:
        operations.append(Operation(
            op_type=OpType(op_data["type"]),
            source=Path(op_data["source"]),
            target=Path(op_data["target"]) if op_data["target"] else None,
            metadata=op_data.get("metadata", {}),
        ))

    return operations
```

## 6. 与 JS/TS 对比

| 概念 | Python | JS/TS |
|------|--------|-------|
| 操作定义 | `@dataclass` | `interface Operation` |
| 枚举类型 | `Enum` | `enum` / `type` |
| 模式匹配 | `match/case` | `switch` |
| 文件操作 | `pathlib` + `shutil` | `fs` / `fs-extra` |
| 序列化 | `json.dump/load` | `JSON.stringify/parse` |

```typescript
// JS/TS 等价实现
interface Operation {
  type: 'rename' | 'move' | 'copy' | 'delete';
  source: string;
  target?: string;
}

class Executor {
  constructor(private dryRun: boolean = false) {}

  async execute(operations: Operation[]): Promise<Result[]> {
    return Promise.all(operations.map(op => this.executeOne(op)));
  }
}
```

## 7. 最佳实践

### 7.1 计划阶段

- ✅ 收集所有操作，不立即执行
- ✅ 验证操作的可行性（文件存在、权限等）
- ✅ 检测冲突（如重命名后文件名重复）
- ✅ 生成操作摘要供用户确认

### 7.2 执行阶段

- ✅ 支持 dry-run 模式
- ✅ 记录每个操作的结果
- ✅ 失败时决定是否继续
- ✅ 提供进度反馈

### 7.3 操作设计

- ✅ 保持操作原子性
- ✅ 设计对应的回滚操作
- ✅ 考虑幂等性

## 小结

| 模式 | 作用 |
|------|------|
| Planner/Executor 分离 | 计划和执行分离，支持预览 |
| 原子操作 | 保证操作的完整性 |
| 操作序列化 | 支持保存、审核、恢复 |
| 结果记录 | 追踪执行状态 |

下一节我们将学习可恢复机制，实现断点续跑功能。

