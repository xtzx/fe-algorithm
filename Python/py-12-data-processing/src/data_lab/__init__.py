"""Data Lab - 数据处理与模型化实验室

这个包提供数据清洗、验证和转换工具。
"""

__version__ = "1.0.0"

from data_lab.models import User, Product, Order, OrderItem, CleaningResult
from data_lab.parsers import read_csv, read_json, read_jsonl, write_jsonl
from data_lab.cleaners import DataCleaner, clean_string, clean_email, safe_int, safe_float
from data_lab.validators import validate_batch
from data_lab.reporters import generate_report, DataQualityReport

__all__ = [
    # 模型
    "User",
    "Product",
    "Order",
    "OrderItem",
    "CleaningResult",
    # 解析器
    "read_csv",
    "read_json",
    "read_jsonl",
    "write_jsonl",
    # 清洗器
    "DataCleaner",
    "clean_string",
    "clean_email",
    "safe_int",
    "safe_float",
    # 验证器
    "validate_batch",
    # 报告器
    "generate_report",
    "DataQualityReport",
]

