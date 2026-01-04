"""
文件自动化工具库

支持批量重命名、文件分类、清理等操作
特性：dry-run 预览、断点续跑、回滚支持
"""

from file_automation.operations import Operation, OpType
from file_automation.planner import Planner, RenamePlanner, OrganizePlanner
from file_automation.executor import Executor, ExecutionResult
from file_automation.state import StateManager, BatchState, TaskStatus

__version__ = "0.1.0"

__all__ = [
    # 操作
    "Operation",
    "OpType",
    # 计划器
    "Planner",
    "RenamePlanner",
    "OrganizePlanner",
    # 执行器
    "Executor",
    "ExecutionResult",
    # 状态管理
    "StateManager",
    "BatchState",
    "TaskStatus",
]

