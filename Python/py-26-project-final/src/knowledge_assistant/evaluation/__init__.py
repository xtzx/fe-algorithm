"""
评测模块

提供:
- 评测数据集
- 评测指标
- 评测运行器
"""

from knowledge_assistant.evaluation.dataset import EvaluationDataset, TestCase
from knowledge_assistant.evaluation.metrics import Metrics, MetricResult, MetricType
from knowledge_assistant.evaluation.runner import EvaluationRunner, EvaluationResults, TestResult

__all__ = [
    "EvaluationDataset",
    "TestCase",
    "Metrics",
    "MetricResult",
    "MetricType",
    "EvaluationRunner",
    "EvaluationResults",
    "TestResult",
]


