# 面试题

> 文件自动化相关的高频面试题

## 1. 如何设计可恢复的批处理任务？

**考察点**：系统设计能力、异常处理思维

### 答案要点

```python
"""
可恢复批处理的核心设计：

1. 状态持久化
   - 记录每个任务的执行状态（pending/running/completed/failed）
   - 每个操作后立即保存状态
   - 使用原子写入防止状态文件损坏

2. 断点续跑
   - 启动时检查是否存在状态文件
   - 跳过已完成的任务，只执行待处理的任务
   - 失败任务可选择重试

3. 任务设计
   - 任务应该有唯一标识（索引或ID）
   - 任务之间尽量独立，减少依赖
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json


@dataclass
class TaskState:
    index: int
    status: str  # pending, running, completed, failed
    error: str | None = None


class ResumableBatch:
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.states: dict[int, TaskState] = {}

    def run(self, tasks: list, processor):
        # 1. 加载已有状态
        self._load_state()

        # 2. 初始化新任务
        for i, task in enumerate(tasks):
            if i not in self.states:
                self.states[i] = TaskState(i, "pending")

        # 3. 执行待处理任务
        for i, task in enumerate(tasks):
            if self.states[i].status == "completed":
                continue  # 跳过已完成

            self.states[i].status = "running"
            self._save_state()

            try:
                processor(task)
                self.states[i].status = "completed"
            except Exception as e:
                self.states[i].status = "failed"
                self.states[i].error = str(e)

            self._save_state()  # 每个任务后保存

    def _save_state(self):
        # 原子写入：先写临时文件，再重命名
        temp = self.state_file.with_suffix('.tmp')
        data = {str(k): {"status": v.status, "error": v.error}
                for k, v in self.states.items()}
        temp.write_text(json.dumps(data))
        temp.rename(self.state_file)
```

### 追问

**Q: 如何处理任务之间有依赖的情况？**

A:
1. 使用 DAG（有向无环图）表示依赖关系
2. 拓扑排序确定执行顺序
3. 状态中记录依赖是否满足
4. 失败时分析影响范围

---

## 2. 什么是幂等操作？为什么重要？

**考察点**：分布式系统基础概念

### 答案要点

```python
"""
幂等操作定义：
执行一次和执行多次，效果完全相同。

数学表达：f(f(x)) = f(x)

重要性：
1. 安全重试：网络超时后可以放心重试
2. 断点续跑：不用担心重复执行
3. 并发安全：多个进程执行同一操作结果一致
4. 故障恢复：系统崩溃后重启不会造成数据不一致
"""

# ❌ 非幂等操作
def append_log(file: Path, message: str):
    with open(file, "a") as f:
        f.write(message + "\n")
    # 多次调用会重复追加


# ✅ 幂等操作
def ensure_line_exists(file: Path, line: str):
    """确保文件包含指定行"""
    if file.exists():
        content = file.read_text()
        if line in content.split("\n"):
            return  # 已存在，跳过

    with open(file, "a") as f:
        f.write(line + "\n")


# ✅ 幂等的文件移动
def idempotent_move(src: Path, dst: Path) -> bool:
    """
    检查目标状态而非假设初始状态
    """
    if dst.exists() and not src.exists():
        return True  # 目标状态已达成

    if src.exists() and not dst.exists():
        src.rename(dst)
        return True

    return False  # 冲突或错误状态
```

### 追问

**Q: 如何让 DELETE 操作幂等？**

A:
```python
def idempotent_delete(path: Path) -> bool:
    if not path.exists():
        return True  # 已删除，满足目标状态
    path.unlink()
    return True
```

---

## 3. Dry-run 模式的作用？

**考察点**：工程实践、用户体验

### 答案要点

```python
"""
Dry-run 的作用：

1. 预览变更
   - 让用户看到将要发生什么
   - 避免误操作

2. 调试脚本
   - 验证逻辑正确性
   - 不产生副作用

3. 审计合规
   - 生成变更报告
   - 记录操作意图

4. 测试环境
   - 在生产数据上安全测试
   - 评估影响范围
"""

class FileProcessor:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def delete(self, path: Path):
        if self.dry_run:
            print(f"[DRY-RUN] Would delete: {path}")
            return True

        path.unlink()
        print(f"Deleted: {path}")
        return True

    def move(self, src: Path, dst: Path):
        if self.dry_run:
            print(f"[DRY-RUN] Would move: {src} → {dst}")
            return True

        shutil.move(src, dst)
        return True


# CLI 通常默认开启 dry-run
# 需要 --execute 或 -x 才真正执行
```

### 追问

**Q: 如何确保 dry-run 和实际执行的逻辑一致？**

A:
1. 使用相同的验证逻辑
2. 只在最后执行步骤分叉
3. 抽象执行函数，通过参数控制是否执行

---

## 4. 如何处理批处理中的部分失败？

**考察点**：错误处理策略

### 答案要点

```python
"""
处理策略：

1. 失败策略选择
   - STOP_ON_FIRST: 遇到第一个失败就停止
   - CONTINUE: 继续执行，最后汇总
   - THRESHOLD: 失败超过阈值停止

2. 错误分类
   - 可重试错误：网络、超时
   - 不可重试错误：权限、文件不存在

3. 失败记录
   - 收集所有失败信息
   - 保存失败列表便于后续重试
   - 生成详细的错误报告

4. 重试机制
   - 自动重试（带退避）
   - 手动重试（加载失败列表）
"""

from enum import Enum

class FailurePolicy(Enum):
    STOP_ON_FIRST = "stop"
    CONTINUE = "continue"
    THRESHOLD = "threshold"


def process_with_policy(
    tasks: list,
    processor,
    policy: FailurePolicy = FailurePolicy.CONTINUE,
    threshold: int = 10,
):
    results = []
    failures = []

    for i, task in enumerate(tasks):
        try:
            result = processor(task)
            results.append((task, result, None))
        except Exception as e:
            results.append((task, None, e))
            failures.append({"index": i, "task": task, "error": str(e)})

            if policy == FailurePolicy.STOP_ON_FIRST:
                break
            if policy == FailurePolicy.THRESHOLD and len(failures) >= threshold:
                break

    # 保存失败列表供重试
    if failures:
        with open("failures.json", "w") as f:
            json.dump(failures, f)

    return results, failures
```

---

## 5. 如何记录批处理的操作日志？

**考察点**：日志设计、可观测性

### 答案要点

```python
"""
日志设计原则：

1. 日志级别
   - DEBUG: 详细执行过程
   - INFO: 正常操作记录
   - WARNING: 可恢复的问题
   - ERROR: 操作失败

2. 日志内容
   - 时间戳
   - 操作类型
   - 源和目标
   - 执行结果
   - 错误信息

3. 日志格式
   - 结构化（JSON）便于解析
   - 人类可读便于调试
   - 包含上下文信息

4. 日志存储
   - 按批次/日期分文件
   - 支持日志轮转
   - 关键操作持久化
"""

import logging
from datetime import datetime


def setup_batch_logging(log_dir: Path, batch_id: str):
    """配置批处理日志"""
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{batch_id}_{datetime.now():%Y%m%d_%H%M%S}.log"

    # JSON 格式日志
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "extra": getattr(record, "extra", {}),
            })

    handler = logging.FileHandler(log_file)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger(batch_id)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger


# 使用
logger = setup_batch_logging(Path("./logs"), "file_rename")

def log_operation(logger, op_type, source, target, success, error=None):
    extra = {
        "operation": op_type,
        "source": str(source),
        "target": str(target),
        "success": success,
    }
    if error:
        extra["error"] = error
        logger.error("Operation failed", extra={"extra": extra})
    else:
        logger.info("Operation completed", extra={"extra": extra})
```

---

## 6. 如何设计回滚机制？

**考察点**：事务思维、系统设计

### 答案要点

```python
"""
回滚设计原则：

1. 记录反向操作
   - 每个操作都有对应的回滚操作
   - RENAME: A→B 回滚为 B→A
   - COPY: 回滚为删除目标
   - DELETE: 需要备份才能回滚

2. 操作顺序
   - 执行时正序
   - 回滚时逆序（LIFO）

3. 回滚粒度
   - 全部回滚：一个失败全部撤销
   - 部分回滚：回滚到某个检查点

4. 回滚日志
   - 持久化已执行的操作
   - 包含回滚所需的所有信息
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class ReversibleOp:
    """可回滚的操作"""
    execute: Callable[[], bool]
    rollback: Callable[[], bool]
    description: str


class TransactionalBatch:
    """事务性批处理"""

    def __init__(self):
        self.executed: list[ReversibleOp] = []

    def run(self, operations: list[ReversibleOp]) -> bool:
        """全部成功或全部回滚"""
        for op in operations:
            try:
                if op.execute():
                    self.executed.append(op)
                else:
                    raise RuntimeError(f"Operation failed: {op.description}")
            except Exception as e:
                print(f"Error: {e}, rolling back...")
                self.rollback()
                return False
        return True

    def rollback(self):
        """逆序回滚"""
        for op in reversed(self.executed):
            try:
                op.rollback()
            except Exception as e:
                print(f"Rollback failed: {op.description}: {e}")
        self.executed.clear()


# 使用示例
def create_rename_op(src: Path, dst: Path) -> ReversibleOp:
    return ReversibleOp(
        execute=lambda: (src.rename(dst), True)[1],
        rollback=lambda: (dst.rename(src), True)[1],
        description=f"rename {src} → {dst}",
    )
```

---

## 7. Planner/Executor 模式有什么好处？

**考察点**：设计模式、关注点分离

### 答案

```python
"""
Planner/Executor 分离的好处：

1. 关注点分离
   - Planner: 分析需求，生成操作计划
   - Executor: 执行操作，处理异常

2. 支持预览
   - 计划生成后可以预览
   - 用户确认后再执行

3. 计划可持久化
   - 保存计划供审核
   - 计划可以在不同时间/机器执行

4. 易于测试
   - 单独测试计划生成逻辑
   - 单独测试执行逻辑

5. 支持优化
   - 分析计划，合并/优化操作
   - 检测冲突
"""

# 计划阶段
class Planner:
    def plan(self, rules) -> list[Operation]:
        operations = []
        # 分析并生成操作
        # 检测冲突
        # 优化顺序
        return operations

# 执行阶段
class Executor:
    def execute(self, plan: list[Operation], dry_run=True):
        for op in plan:
            if dry_run:
                print(f"Would: {op}")
            else:
                self._do(op)

# 使用流程
plan = Planner().plan(rules)      # 1. 生成计划
print(plan)                        # 2. 预览/审核
if confirm():                      # 3. 确认
    Executor().execute(plan, dry_run=False)  # 4. 执行
```

---

## 8. 如何确保批处理的原子性？

**考察点**：事务概念、一致性

### 答案

```python
"""
确保原子性的方法：

1. 文件系统级别
   - rename() 是原子的
   - 先写临时文件，再 rename

2. 备份恢复
   - 执行前备份
   - 失败后从备份恢复

3. 两阶段提交
   - 准备阶段：检查所有前置条件
   - 执行阶段：实际执行（此时应该不会失败）

4. WAL (Write-Ahead Logging)
   - 先记录要做什么
   - 再执行
   - 恢复时根据日志重做
"""

import shutil
from pathlib import Path


class AtomicBatch:
    """原子性批处理"""

    def __init__(self, backup_dir: Path):
        self.backup_dir = backup_dir
        self.backups: list[tuple[Path, Path]] = []

    def run(self, operations):
        try:
            # 阶段 1: 备份
            for op in operations:
                if op.source.exists():
                    backup = self.backup_dir / op.source.name
                    shutil.copy2(op.source, backup)
                    self.backups.append((op.source, backup))

            # 阶段 2: 执行
            for op in operations:
                op.execute()

            # 成功，清理备份
            self._cleanup_backups()

        except Exception as e:
            # 失败，恢复备份
            self._restore_backups()
            raise

    def _restore_backups(self):
        for original, backup in self.backups:
            if backup.exists():
                shutil.copy2(backup, original)

    def _cleanup_backups(self):
        for _, backup in self.backups:
            if backup.exists():
                backup.unlink()
```

---

## 总结

| 问题 | 核心考点 |
|------|---------|
| 可恢复批处理 | 状态持久化、断点续跑 |
| 幂等操作 | 多次执行结果相同 |
| Dry-run | 预览变更、安全执行 |
| 部分失败 | 策略选择、错误分类 |
| 操作日志 | 结构化、可追溯 |
| 回滚机制 | 反向操作、逆序执行 |
| Planner/Executor | 分离关注点、支持预览 |
| 原子性 | 备份恢复、两阶段提交 |

