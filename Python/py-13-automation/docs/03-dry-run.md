# Dry-run 模式

> 先看看会发生什么，确认无误再真正执行

## 1. 什么是 Dry-run？

**Dry-run**（干跑/模拟运行）：执行所有逻辑但不产生副作用，用于预览变更。

```bash
# 常见的 dry-run 示例
rsync -avz --dry-run source/ dest/    # rsync 预览
rm -rf --dry-run ./folder             # 虽然 rm 没这选项，但概念一样
git push --dry-run                    # Git 预览推送
```

## 2. 为什么需要 Dry-run？

```python
# ❌ 危险：直接执行，无法回头
for f in Path(".").glob("*.log"):
    f.unlink()  # 删了才发现删错了

# ✅ 安全：先预览再执行
plan = planner.plan_delete("*.log")
print("将删除以下文件:")
for op in plan:
    print(f"  - {op.source}")

if confirm("确认删除？"):
    executor.execute(plan)
```

Dry-run 的价值：
- 🔍 **预览变更**：知道会发生什么
- ✅ **确认执行**：用户明确同意
- 📝 **生成报告**：变更审计
- 🐛 **调试脚本**：验证逻辑正确性

## 3. 实现 Dry-run 模式

### 3.1 基础实现

```python
from dataclasses import dataclass
from pathlib import Path
import shutil
import logging


@dataclass
class DryRunResult:
    """Dry-run 结果"""
    operation: str
    would_succeed: bool
    message: str


class DryRunExecutor:
    """支持 dry-run 的执行器"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)

    def rename(self, src: Path, dst: Path) -> DryRunResult:
        """重命名文件"""
        op_desc = f"RENAME: {src} → {dst}"

        # 检查前置条件
        if not src.exists():
            return DryRunResult(op_desc, False, f"源文件不存在: {src}")

        if dst.exists():
            return DryRunResult(op_desc, False, f"目标已存在: {dst}")

        if self.dry_run:
            self.logger.info(f"[DRY-RUN] {op_desc}")
            return DryRunResult(op_desc, True, "Would rename")

        # 实际执行
        src.rename(dst)
        self.logger.info(f"[EXECUTED] {op_desc}")
        return DryRunResult(op_desc, True, "Renamed")

    def move(self, src: Path, dst: Path) -> DryRunResult:
        """移动文件"""
        op_desc = f"MOVE: {src} → {dst}"

        if not src.exists():
            return DryRunResult(op_desc, False, f"源文件不存在: {src}")

        if self.dry_run:
            self.logger.info(f"[DRY-RUN] {op_desc}")
            return DryRunResult(op_desc, True, "Would move")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
        self.logger.info(f"[EXECUTED] {op_desc}")
        return DryRunResult(op_desc, True, "Moved")

    def delete(self, path: Path) -> DryRunResult:
        """删除文件"""
        op_desc = f"DELETE: {path}"

        if not path.exists():
            return DryRunResult(op_desc, False, f"文件不存在: {path}")

        if self.dry_run:
            self.logger.info(f"[DRY-RUN] {op_desc}")
            return DryRunResult(op_desc, True, "Would delete")

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

        self.logger.info(f"[EXECUTED] {op_desc}")
        return DryRunResult(op_desc, True, "Deleted")

    def copy(self, src: Path, dst: Path) -> DryRunResult:
        """复制文件"""
        op_desc = f"COPY: {src} → {dst}"

        if not src.exists():
            return DryRunResult(op_desc, False, f"源文件不存在: {src}")

        if self.dry_run:
            self.logger.info(f"[DRY-RUN] {op_desc}")
            return DryRunResult(op_desc, True, "Would copy")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.logger.info(f"[EXECUTED] {op_desc}")
        return DryRunResult(op_desc, True, "Copied")
```

### 3.2 通用操作执行器

```python
from typing import Callable, Any
from enum import Enum


class ExecutionMode(Enum):
    DRY_RUN = "dry_run"
    EXECUTE = "execute"
    INTERACTIVE = "interactive"  # 每步确认


class UniversalExecutor:
    """通用执行器"""

    def __init__(self, mode: ExecutionMode = ExecutionMode.DRY_RUN):
        self.mode = mode
        self.results: list[DryRunResult] = []

    def run_operation(
        self,
        description: str,
        check_fn: Callable[[], bool],
        execute_fn: Callable[[], Any],
    ) -> DryRunResult:
        """
        执行操作

        Args:
            description: 操作描述
            check_fn: 检查是否可执行
            execute_fn: 实际执行函数
        """
        # 检查前置条件
        if not check_fn():
            result = DryRunResult(description, False, "前置条件不满足")
            self.results.append(result)
            return result

        # Dry-run 模式
        if self.mode == ExecutionMode.DRY_RUN:
            result = DryRunResult(description, True, "[DRY-RUN] Would execute")
            print(f"  {result.message}: {description}")
            self.results.append(result)
            return result

        # 交互模式
        if self.mode == ExecutionMode.INTERACTIVE:
            print(f"\n即将执行: {description}")
            choice = input("执行？[y/n/q]: ").lower()
            if choice == "q":
                raise KeyboardInterrupt("用户取消")
            if choice != "y":
                result = DryRunResult(description, True, "用户跳过")
                self.results.append(result)
                return result

        # 实际执行
        try:
            execute_fn()
            result = DryRunResult(description, True, "已执行")
            self.results.append(result)
            return result
        except Exception as e:
            result = DryRunResult(description, False, f"执行失败: {e}")
            self.results.append(result)
            return result

    def get_summary(self) -> dict[str, int]:
        """获取执行摘要"""
        return {
            "total": len(self.results),
            "would_succeed": sum(1 for r in self.results if r.would_succeed),
            "would_fail": sum(1 for r in self.results if not r.would_succeed),
        }
```

## 4. 预览报告

### 4.1 文本报告

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChangePreview:
    """变更预览"""
    operation_type: str
    source: Path
    target: Path | None
    size_bytes: int = 0

    def format_line(self) -> str:
        size_str = format_size(self.size_bytes) if self.size_bytes > 0 else ""
        match self.operation_type:
            case "rename":
                return f"  [RENAME] {self.source.name} → {self.target.name} {size_str}"
            case "move":
                return f"  [MOVE]   {self.source} → {self.target} {size_str}"
            case "delete":
                return f"  [DELETE] {self.source} {size_str}"
            case "copy":
                return f"  [COPY]   {self.source} → {self.target} {size_str}"
            case _:
                return f"  [{self.operation_type.upper()}] {self.source}"


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"({size_bytes:.1f} {unit})"
        size_bytes /= 1024
    return f"({size_bytes:.1f} TB)"


def generate_preview_report(changes: list[ChangePreview]) -> str:
    """生成预览报告"""
    lines = [
        "=" * 60,
        "变更预览报告",
        "=" * 60,
        "",
    ]

    # 按操作类型分组
    by_type: dict[str, list[ChangePreview]] = {}
    for change in changes:
        by_type.setdefault(change.operation_type, []).append(change)

    for op_type, items in by_type.items():
        lines.append(f"{op_type.upper()} ({len(items)} 个文件):")
        for item in items:
            lines.append(item.format_line())
        lines.append("")

    # 统计
    total_size = sum(c.size_bytes for c in changes)
    lines.extend([
        "-" * 60,
        f"总计: {len(changes)} 个操作",
        f"涉及数据量: {format_size(total_size)}",
        "=" * 60,
    ])

    return "\n".join(lines)
```

### 4.2 JSON 报告

```python
import json
from datetime import datetime


def generate_json_report(
    changes: list[ChangePreview],
    output_file: Path | None = None,
) -> dict:
    """生成 JSON 格式报告"""
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_operations": len(changes),
            "by_type": {},
            "total_size_bytes": sum(c.size_bytes for c in changes),
        },
        "operations": []
    }

    # 按类型统计
    for change in changes:
        report["summary"]["by_type"].setdefault(change.operation_type, 0)
        report["summary"]["by_type"][change.operation_type] += 1

        report["operations"].append({
            "type": change.operation_type,
            "source": str(change.source),
            "target": str(change.target) if change.target else None,
            "size_bytes": change.size_bytes,
        })

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report
```

## 5. 确认执行流程

### 5.1 简单确认

```python
def confirm(message: str, default: bool = False) -> bool:
    """简单确认"""
    suffix = "[Y/n]" if default else "[y/N]"
    response = input(f"{message} {suffix}: ").strip().lower()

    if not response:
        return default
    return response in ("y", "yes")


# 使用
if confirm("确认执行以上操作？"):
    executor.execute(plan)
else:
    print("已取消")
```

### 5.2 详细确认流程

```python
from enum import Enum


class ConfirmChoice(Enum):
    YES = "y"           # 执行
    NO = "n"            # 取消
    SHOW_DETAILS = "d"  # 显示详情
    SAVE_PLAN = "s"     # 保存计划
    QUIT = "q"          # 退出


def interactive_confirm(
    changes: list[ChangePreview],
    plan_file: Path | None = None,
) -> bool:
    """交互式确认"""
    # 显示摘要
    print(f"\n即将执行 {len(changes)} 个操作:")

    by_type: dict[str, int] = {}
    for c in changes:
        by_type[c.operation_type] = by_type.get(c.operation_type, 0) + 1

    for op_type, count in by_type.items():
        print(f"  - {op_type}: {count} 个")

    while True:
        print("\n选项:")
        print("  [y] 执行")
        print("  [n] 取消")
        print("  [d] 显示详情")
        print("  [s] 保存计划到文件")
        print("  [q] 退出")

        choice = input("\n请选择: ").strip().lower()

        if choice == "y":
            return True
        elif choice == "n":
            print("已取消")
            return False
        elif choice == "d":
            print("\n" + generate_preview_report(changes))
        elif choice == "s":
            save_path = plan_file or Path("plan.json")
            generate_json_report(changes, save_path)
            print(f"计划已保存到: {save_path}")
        elif choice == "q":
            print("退出")
            exit(0)
        else:
            print("无效选项，请重新选择")
```

## 6. 日志记录

### 6.1 配置日志

```python
import logging
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_dir: Path,
    dry_run: bool = False,
) -> logging.Logger:
    """配置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件名包含时间和模式
    mode = "dry-run" if dry_run else "execute"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{mode}_{timestamp}.log"

    # 创建 logger
    logger = logging.getLogger("file_automation")
    logger.setLevel(logging.DEBUG)

    # 文件处理器 - 详细日志
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # 控制台处理器 - 简要日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"日志文件: {log_file}")
    logger.info(f"模式: {'Dry-run' if dry_run else 'Execute'}")

    return logger
```

### 6.2 操作日志格式

```python
def log_operation(
    logger: logging.Logger,
    operation: str,
    source: Path,
    target: Path | None,
    dry_run: bool,
    success: bool,
    error: str | None = None,
) -> None:
    """记录操作日志"""
    mode = "[DRY-RUN]" if dry_run else "[EXECUTE]"
    status = "SUCCESS" if success else "FAILED"

    if target:
        msg = f"{mode} {operation}: {source} → {target}"
    else:
        msg = f"{mode} {operation}: {source}"

    if success:
        logger.info(f"{status} | {msg}")
    else:
        logger.error(f"{status} | {msg} | Error: {error}")
```

## 7. 完整工作流

```python
def batch_rename_workflow(
    directory: Path,
    pattern: str,
    replacement: str,
    dry_run: bool = True,
    log_dir: Path = Path("./logs"),
) -> None:
    """完整的批量重命名工作流"""

    # 1. 配置日志
    logger = setup_logging(log_dir, dry_run)
    logger.info(f"目录: {directory}")
    logger.info(f"模式: {pattern} → {replacement}")

    # 2. 创建计划
    planner = RenamePlanner(directory)
    operations = planner.plan_regex_rename(pattern, replacement)

    if not operations:
        logger.info("没有匹配的文件")
        return

    logger.info(f"找到 {len(operations)} 个文件需要重命名")

    # 3. 生成预览
    changes = [
        ChangePreview(
            operation_type="rename",
            source=op.source,
            target=op.target,
            size_bytes=op.source.stat().st_size if op.source.exists() else 0,
        )
        for op in operations
    ]

    # 4. 显示预览报告
    print("\n" + generate_preview_report(changes))

    # 5. Dry-run 模式直接返回
    if dry_run:
        logger.info("Dry-run 完成，未执行任何操作")
        return

    # 6. 确认执行
    if not interactive_confirm(changes):
        return

    # 7. 执行
    executor = Executor(dry_run=False)
    results = executor.execute(operations)

    # 8. 输出结果
    success_count = sum(1 for r in results if r.success)
    failed_count = sum(1 for r in results if not r.success)

    logger.info(f"执行完成: {success_count} 成功, {failed_count} 失败")

    if failed_count > 0:
        logger.warning("失败的操作:")
        for r in results:
            if not r.success:
                logger.warning(f"  - {r.operation}: {r.error}")
```

## 8. CLI 集成

```python
import argparse


def main():
    parser = argparse.ArgumentParser(description="文件批处理工具")
    parser.add_argument("directory", type=Path, help="目标目录")
    parser.add_argument("--pattern", "-p", required=True, help="匹配模式")
    parser.add_argument("--replacement", "-r", required=True, help="替换内容")
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        default=True,
        help="预览模式，不实际执行（默认）"
    )
    parser.add_argument(
        "--execute", "-x",
        action="store_true",
        help="实际执行"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./logs"),
        help="日志目录"
    )

    args = parser.parse_args()

    # --execute 覆盖 --dry-run
    dry_run = not args.execute

    batch_rename_workflow(
        directory=args.directory,
        pattern=args.pattern,
        replacement=args.replacement,
        dry_run=dry_run,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
```

使用：

```bash
# 预览（默认）
python batch_rename.py ./docs --pattern "old_" --replacement "new_"

# 执行
python batch_rename.py ./docs --pattern "old_" --replacement "new_" --execute
```

## 小结

| 功能 | 作用 |
|------|------|
| Dry-run 模式 | 预览变更，不实际执行 |
| 预览报告 | 清晰展示将要发生的变更 |
| 确认流程 | 用户明确同意后才执行 |
| 日志记录 | 完整的操作审计轨迹 |

下一节我们将学习失败处理和回滚机制。

