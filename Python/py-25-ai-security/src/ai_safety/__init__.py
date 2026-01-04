"""
AI 服务安全与评测工具包

提供:
- 提示注入防护
- 输出安全过滤
- 评测体系
- 监控系统
"""

__version__ = "1.0.0"

from ai_safety.guards import (
    InjectionDetector,
    InputFilter,
    OutputFilter,
)
from ai_safety.evaluation import (
    EvaluationRunner,
    Metrics,
)

__all__ = [
    "InjectionDetector",
    "InputFilter",
    "OutputFilter",
    "EvaluationRunner",
    "Metrics",
]


