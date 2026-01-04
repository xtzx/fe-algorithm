#!/usr/bin/env python3
"""
文件自动化工具使用示例
"""

from pathlib import Path
import tempfile
import sys

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from file_automation import (
    Operation,
    OpType,
    RenamePlanner,
    OrganizePlanner,
    Executor,
    StateManager,
)
from file_automation.executor import preview_operations


def example_rename():
    """示例：批量重命名"""
    print("=" * 60)
    print("示例 1: 批量重命名")
    print("=" * 60)

    # 创建临时目录和文件
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建示例文件
        for i in range(5):
            (temp_path / f"report_2024010{i + 1}.txt").write_text(f"Report {i + 1}")

        print(f"原始文件: {list(temp_path.glob('*.txt'))}")

        # 创建计划
        planner = RenamePlanner(temp_path)
        operations = planner.plan_regex_rename(
            pattern=r"report_(\d{4})(\d{2})(\d{2})",
            replacement=r"\1-\2-\3_report",
            file_glob="*.txt",
        )

        # 预览
        print("\n计划:")
        for op in operations:
            print(f"  {op}")

        # 执行（dry-run）
        executor = Executor(dry_run=True)
        summary = executor.execute(operations)
        print(f"\nDry-run 结果: {summary.to_dict()}")

        # 实际执行
        executor = Executor(dry_run=False)
        summary = executor.execute(operations)
        print(f"执行结果: {summary.to_dict()}")

        print(f"重命名后: {list(temp_path.glob('*.txt'))}")


def example_organize():
    """示例：文件分类"""
    print("\n" + "=" * 60)
    print("示例 2: 文件分类")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建不同类型的文件
        (temp_path / "photo.jpg").write_text("image")
        (temp_path / "video.mp4").write_text("video")
        (temp_path / "document.pdf").write_text("doc")
        (temp_path / "script.py").write_text("code")
        (temp_path / "data.json").write_text("{}")

        print(f"原始文件: {[f.name for f in temp_path.iterdir()]}")

        # 创建计划
        planner = OrganizePlanner(temp_path)
        operations = planner.plan_by_extension()

        # 预览
        print(f"\n将创建 {len(operations)} 个操作")

        # 执行
        executor = Executor(dry_run=False)
        summary = executor.execute(operations)
        print(f"执行结果: {summary.to_dict()}")

        # 显示结果
        print("\n分类后的目录结构:")
        for item in sorted(temp_path.rglob("*")):
            if item.is_file():
                print(f"  {item.relative_to(temp_path)}")


def example_resumable():
    """示例：断点续跑"""
    print("\n" + "=" * 60)
    print("示例 3: 断点续跑")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        state_file = temp_path / "state.json"

        # 创建文件
        for i in range(5):
            (temp_path / f"file_{i}.txt").write_text(f"Content {i}")

        operations = [
            Operation(OpType.MKDIR, temp_path / "backup"),
        ] + [
            Operation(
                OpType.COPY,
                temp_path / f"file_{i}.txt",
                temp_path / "backup" / f"file_{i}.txt",
            )
            for i in range(5)
        ]

        # 第一次执行（假设中途中断）
        print("第一次执行...")
        executor1 = Executor(dry_run=False, state_file=state_file)

        # 手动只执行前 3 个
        for i, op in enumerate(operations[:3]):
            from file_automation.operations import OperationExecutor

            OperationExecutor.execute(op)
            if executor1.state_manager:
                executor1.state_manager.mark_completed(i)

        state = StateManager(state_file).load()
        if state:
            print(f"状态: {state.get_summary()}")

        # 第二次执行（恢复）
        print("\n恢复执行...")
        executor2 = Executor(dry_run=False, state_file=state_file)
        summary = executor2.execute(operations)
        print(f"执行结果: {summary.to_dict()}")
        print(f"跳过了 {summary.skipped} 个已完成的操作")


def example_rollback():
    """示例：回滚"""
    print("\n" + "=" * 60)
    print("示例 4: 回滚操作")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        rollback_file = temp_path / "rollback.json"

        # 创建文件
        (temp_path / "original.txt").write_text("content")

        print(f"执行前: {[f.name for f in temp_path.glob('*.txt')]}")

        # 执行重命名
        operations = [
            Operation(
                OpType.RENAME,
                temp_path / "original.txt",
                temp_path / "renamed.txt",
            )
        ]

        executor = Executor(dry_run=False, rollback_file=rollback_file)
        executor.execute(operations)
        print(f"执行后: {[f.name for f in temp_path.glob('*.txt')]}")

        # 回滚
        executor.rollback()
        print(f"回滚后: {[f.name for f in temp_path.glob('*.txt')]}")


def main():
    """运行所有示例"""
    example_rename()
    example_organize()
    example_resumable()
    example_rollback()

    print("\n" + "=" * 60)
    print("所有示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

