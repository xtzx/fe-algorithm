"""数据报告生成器

生成数据质量报告和统计信息。
"""

import json
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldReport:
    """字段报告"""

    name: str
    total: int
    non_null: int
    null_count: int
    null_rate: float
    unique_count: int
    data_types: dict[str, int]
    sample_values: list[Any]
    # 数值字段统计
    numeric_stats: dict[str, float] | None = None


@dataclass
class DataQualityReport:
    """数据质量报告"""

    total_records: int
    total_fields: int
    fields: list[FieldReport] = field(default_factory=list)
    duplicate_count: int = 0
    completeness_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "summary": {
                "total_records": self.total_records,
                "total_fields": self.total_fields,
                "duplicate_count": self.duplicate_count,
                "completeness_rate": self.completeness_rate,
            },
            "fields": [
                {
                    "name": f.name,
                    "total": f.total,
                    "non_null": f.non_null,
                    "null_count": f.null_count,
                    "null_rate": round(f.null_rate, 4),
                    "unique_count": f.unique_count,
                    "data_types": f.data_types,
                    "sample_values": f.sample_values,
                    "numeric_stats": f.numeric_stats,
                }
                for f in self.fields
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """转换为 Markdown"""
        lines = [
            "# Data Quality Report",
            "",
            "## Summary",
            "",
            f"- **Total Records**: {self.total_records:,}",
            f"- **Total Fields**: {self.total_fields}",
            f"- **Duplicate Count**: {self.duplicate_count:,}",
            f"- **Completeness Rate**: {self.completeness_rate:.2%}",
            "",
            "## Fields",
            "",
            "| Field | Non-Null | Null Rate | Unique | Types |",
            "|-------|----------|-----------|--------|-------|",
        ]

        for f in self.fields:
            types = ", ".join(f"{k}:{v}" for k, v in f.data_types.items())
            lines.append(
                f"| {f.name} | {f.non_null:,} | {f.null_rate:.2%} | {f.unique_count:,} | {types} |"
            )

        return "\n".join(lines)


def generate_report(items: list[dict[str, Any]]) -> DataQualityReport:
    """生成数据质量报告

    Args:
        items: 数据列表

    Returns:
        DataQualityReport 对象
    """
    if not items:
        return DataQualityReport(total_records=0, total_fields=0)

    total = len(items)

    # 收集所有字段
    all_fields: set[str] = set()
    for item in items:
        all_fields.update(item.keys())

    fields = []
    total_non_null = 0
    total_cells = total * len(all_fields)

    for field_name in sorted(all_fields):
        values = [item.get(field_name) for item in items]

        # 空值统计
        non_null_values = [v for v in values if v is not None and v != ""]
        null_count = total - len(non_null_values)
        null_rate = null_count / total if total > 0 else 0

        total_non_null += len(non_null_values)

        # 唯一值
        unique_values = set(str(v) for v in non_null_values)

        # 数据类型统计
        type_counts: dict[str, int] = {}
        for v in non_null_values:
            type_name = type(v).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # 数值统计
        numeric_stats = None
        numeric_values = [
            float(v) for v in non_null_values if isinstance(v, (int, float))
        ]
        if numeric_values:
            numeric_stats = {
                "count": len(numeric_values),
                "min": min(numeric_values),
                "max": max(numeric_values),
                "mean": statistics.mean(numeric_values),
                "median": statistics.median(numeric_values),
            }
            if len(numeric_values) > 1:
                numeric_stats["stdev"] = statistics.stdev(numeric_values)

        # 采样值
        sample_values = list(unique_values)[:5]

        fields.append(FieldReport(
            name=field_name,
            total=total,
            non_null=len(non_null_values),
            null_count=null_count,
            null_rate=null_rate,
            unique_count=len(unique_values),
            data_types=type_counts,
            sample_values=sample_values,
            numeric_stats=numeric_stats,
        ))

    # 完整性
    completeness_rate = total_non_null / total_cells if total_cells > 0 else 0

    return DataQualityReport(
        total_records=total,
        total_fields=len(all_fields),
        fields=fields,
        completeness_rate=completeness_rate,
    )


def generate_cleaning_report(
    original: list[dict],
    cleaned: list[dict],
    errors: list[dict],
) -> dict[str, Any]:
    """生成清洗报告

    Args:
        original: 原始数据
        cleaned: 清洗后的数据
        errors: 错误列表

    Returns:
        报告字典
    """
    return {
        "summary": {
            "original_count": len(original),
            "cleaned_count": len(cleaned),
            "error_count": len(errors),
            "success_rate": len(cleaned) / len(original) if original else 0,
        },
        "errors": errors[:100],  # 最多 100 个错误
    }


def basic_stats(values: list[float]) -> dict[str, float]:
    """计算基础统计

    Args:
        values: 数值列表

    Returns:
        统计字典
    """
    if not values:
        return {"count": 0}

    result = {
        "count": len(values),
        "sum": sum(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }

    if len(values) > 1:
        result["stdev"] = statistics.stdev(values)

    return result


def field_stats(items: list[dict], field_name: str) -> dict[str, Any]:
    """计算字段统计

    Args:
        items: 数据列表
        field_name: 字段名

    Returns:
        统计字典
    """
    values = [item.get(field_name) for item in items if field_name in item]

    # 基础统计
    total = len(values)
    non_null = [v for v in values if v is not None and v != ""]

    result = {
        "total": total,
        "non_null": len(non_null),
        "null_count": total - len(non_null),
        "null_rate": (total - len(non_null)) / total if total > 0 else 0,
        "unique_count": len(set(str(v) for v in non_null)),
    }

    # 数值统计
    numeric = [float(v) for v in non_null if isinstance(v, (int, float))]
    if numeric:
        result["numeric_stats"] = basic_stats(numeric)

    return result

