"""数据清洗器

提供数据清洗和转换功能。
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable


# ============================================================
# 基础清洗函数
# ============================================================


def clean_string(value: Any, default: str = "") -> str:
    """清洗字符串

    - 转换为字符串
    - 去除首尾空白
    - Unicode 规范化
    - 合并多个空格
    """
    if value is None:
        return default

    s = str(value).strip()

    # 处理空值表示
    if s.lower() in ("null", "none", "n/a", "na", "-", ""):
        return default

    # Unicode 规范化
    s = unicodedata.normalize("NFKC", s)

    # 合并多个空格
    s = re.sub(r"\s+", " ", s)

    return s


def clean_email(value: Any, default: str = "") -> str:
    """清洗邮箱

    - 转小写
    - 去空白
    - 验证格式
    """
    s = clean_string(value)
    if not s:
        return default

    s = s.lower()

    # 简单验证
    if "@" not in s or "." not in s.split("@")[-1]:
        return default

    return s


def clean_phone(value: Any, default: str = "") -> str:
    """清洗电话号码

    - 只保留数字
    """
    s = clean_string(value)
    if not s:
        return default

    digits = re.sub(r"\D", "", s)
    return digits if len(digits) >= 10 else default


def clean_name(value: Any, default: str = "") -> str:
    """清洗姓名

    - 去空白
    - 标题化
    """
    s = clean_string(value)
    if not s:
        return default

    return s.title()


# ============================================================
# 类型转换
# ============================================================


def safe_int(value: Any, default: int = 0) -> int:
    """安全转换为整数"""
    if value is None:
        return default

    try:
        # 处理带小数的字符串
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """安全转换为浮点数"""
    if value is None:
        return default

    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    """安全转换为布尔值"""
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1", "on", "y")

    return bool(value)


# ============================================================
# 清洗规则
# ============================================================


@dataclass
class CleaningRule:
    """清洗规则"""

    field: str
    cleaner: Callable[[Any], Any]
    default: Any = None
    required: bool = False


# ============================================================
# 数据清洗器
# ============================================================


@dataclass
class CleaningStats:
    """清洗统计"""

    total: int = 0
    success: int = 0
    failed: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)


class DataCleaner:
    """数据清洗器

    使用规则对数据进行清洗。
    """

    def __init__(self, rules: list[CleaningRule] | None = None):
        """初始化清洗器

        Args:
            rules: 清洗规则列表
        """
        self.rules: dict[str, CleaningRule] = {}
        if rules:
            for rule in rules:
                self.add_rule(rule)

    def add_rule(self, rule: CleaningRule) -> "DataCleaner":
        """添加规则"""
        self.rules[rule.field] = rule
        return self

    def add(
        self,
        field: str,
        cleaner: Callable[[Any], Any],
        default: Any = None,
        required: bool = False,
    ) -> "DataCleaner":
        """添加规则（便捷方法）"""
        return self.add_rule(CleaningRule(field, cleaner, default, required))

    def clean(self, data: dict[str, Any]) -> dict[str, Any]:
        """清洗单条数据

        Args:
            data: 原始数据

        Returns:
            清洗后的数据
        """
        result = data.copy()

        for field_name, rule in self.rules.items():
            value = data.get(field_name, rule.default)
            try:
                result[field_name] = rule.cleaner(value)
            except Exception:
                result[field_name] = rule.default

        return result

    def clean_batch(
        self,
        items: list[dict[str, Any]],
        skip_errors: bool = True,
    ) -> tuple[list[dict[str, Any]], CleaningStats]:
        """批量清洗

        Args:
            items: 数据列表
            skip_errors: 是否跳过错误

        Returns:
            (清洗后的数据, 统计信息)
        """
        stats = CleaningStats(total=len(items))
        cleaned = []

        for i, item in enumerate(items):
            try:
                result = self.clean(item)

                # 检查必填字段
                for field_name, rule in self.rules.items():
                    if rule.required and not result.get(field_name):
                        raise ValueError(f"Required field '{field_name}' is empty")

                cleaned.append(result)
                stats.success += 1
            except Exception as e:
                stats.failed += 1
                stats.errors.append({
                    "index": i,
                    "error": str(e),
                    "data": item,
                })
                if not skip_errors:
                    raise

        return cleaned, stats


# ============================================================
# 预定义清洗器
# ============================================================


def create_user_cleaner() -> DataCleaner:
    """创建用户数据清洗器"""
    return DataCleaner([
        CleaningRule("name", clean_name, "", required=True),
        CleaningRule("email", clean_email, "", required=True),
        CleaningRule("age", safe_int, None),
        CleaningRule("phone", clean_phone, ""),
    ])


def create_product_cleaner() -> DataCleaner:
    """创建产品数据清洗器"""
    return DataCleaner([
        CleaningRule("name", clean_string, "", required=True),
        CleaningRule("price", safe_float, 0.0, required=True),
        CleaningRule("quantity", safe_int, 0),
        CleaningRule("category", clean_string, ""),
    ])


# ============================================================
# 去重
# ============================================================


def dedupe_by_field(items: list[dict], field: str) -> list[dict]:
    """按字段去重，保留第一个"""
    seen: set = set()
    result = []
    for item in items:
        key = item.get(field)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def dedupe_by_fields(items: list[dict], fields: list[str]) -> list[dict]:
    """按多字段去重"""
    seen: set = set()
    result = []
    for item in items:
        key = tuple(item.get(f) for f in fields)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result

