"""
测试计划器模块
"""

import pytest
from pathlib import Path

from file_automation.planner import (
    RenamePlanner,
    OrganizePlanner,
    CleanupPlanner,
    get_category,
    analyze_plan,
)
from file_automation.operations import OpType


class TestGetCategory:
    """测试文件分类"""

    def test_image_category(self):
        assert get_category(".jpg") == "images"
        assert get_category(".PNG") == "images"
        assert get_category(".gif") == "images"

    def test_document_category(self):
        assert get_category(".pdf") == "documents"
        assert get_category(".docx") == "documents"
        assert get_category(".txt") == "documents"

    def test_code_category(self):
        assert get_category(".py") == "code"
        assert get_category(".js") == "code"
        assert get_category(".ts") == "code"

    def test_unknown_category(self):
        assert get_category(".xyz") == "others"
        assert get_category(".unknown") == "others"


class TestRenamePlanner:
    """测试重命名计划器"""

    def test_regex_rename(self, sample_files: Path):
        """测试正则重命名"""
        planner = RenamePlanner(sample_files)
        operations = planner.plan_regex_rename(
            pattern=r"report_(\d{4})(\d{2})(\d{2})",
            replacement=r"\1-\2-\3_report",
            file_glob="*.txt",
        )

        assert len(operations) == 2

        for op in operations:
            assert op.op_type == OpType.RENAME
            assert "report" in str(op.target)

    def test_sequential_rename(self, sample_files: Path):
        """测试序号重命名"""
        planner = RenamePlanner(sample_files)
        operations = planner.plan_sequential_rename(
            prefix="file_",
            file_glob="*.txt",
            start=1,
            width=3,
        )

        assert len(operations) >= 1

        for op in operations:
            assert op.op_type == OpType.RENAME
            assert op.target is not None
            assert "file_" in op.target.name

    def test_datetime_rename(self, sample_files: Path):
        """测试日期时间重命名"""
        planner = RenamePlanner(sample_files)
        operations = planner.plan_datetime_rename(
            file_glob="*.txt",
            date_format="%Y%m%d",
        )

        assert len(operations) >= 1

        for op in operations:
            assert op.op_type == OpType.RENAME

    def test_no_matches(self, sample_files: Path):
        """测试无匹配文件"""
        planner = RenamePlanner(sample_files)
        operations = planner.plan_regex_rename(
            pattern="nonexistent",
            replacement="new",
            file_glob="*.xyz",
        )

        assert len(operations) == 0


class TestOrganizePlanner:
    """测试整理计划器"""

    def test_organize_by_extension(self, sample_files: Path):
        """测试按扩展名分类"""
        planner = OrganizePlanner(sample_files)
        operations = planner.plan_by_extension()

        # 应该有 MKDIR 和 MOVE 操作
        mkdir_ops = [op for op in operations if op.op_type == OpType.MKDIR]
        move_ops = [op for op in operations if op.op_type == OpType.MOVE]

        assert len(mkdir_ops) > 0
        assert len(move_ops) > 0

    def test_organize_by_size(self, sample_files: Path):
        """测试按大小分类"""
        planner = OrganizePlanner(sample_files)
        operations = planner.plan_by_size()

        move_ops = [op for op in operations if op.op_type == OpType.MOVE]
        assert len(move_ops) > 0

        # 检查元数据中有大小信息
        for op in move_ops:
            assert "size" in op.metadata
            assert "category" in op.metadata


class TestCleanupPlanner:
    """测试清理计划器"""

    def test_delete_empty_dirs(self, nested_files: Path):
        """测试删除空目录"""
        # 创建空目录
        empty_dir = nested_files / "empty_folder"
        empty_dir.mkdir()

        planner = CleanupPlanner(nested_files)
        operations = planner.plan_delete_empty_dirs()

        assert len(operations) == 1
        assert operations[0].op_type == OpType.DELETE
        assert operations[0].source == empty_dir

    def test_delete_by_pattern(self, sample_files: Path):
        """测试按模式删除"""
        planner = CleanupPlanner(sample_files)
        operations = planner.plan_delete_by_pattern(["*.txt"])

        assert len(operations) >= 1

        for op in operations:
            assert op.op_type == OpType.DELETE
            assert op.source.suffix == ".txt"


class TestAnalyzePlan:
    """测试计划分析"""

    def test_analyze_plan(self, sample_files: Path):
        """测试分析计划"""
        planner = OrganizePlanner(sample_files)
        operations = planner.plan_by_extension()

        analysis = analyze_plan(operations)

        assert "total" in analysis
        assert "by_type" in analysis
        assert "operations" in analysis
        assert analysis["total"] == len(operations)

