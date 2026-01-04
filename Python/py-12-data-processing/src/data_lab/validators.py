"""数据验证器

使用 pydantic 进行数据验证。
"""

import logging
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from data_lab.models import CleaningResult, User

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def validate_item(
    data: dict[str, Any],
    model: Type[T],
) -> tuple[T | None, list[dict[str, Any]]]:
    """验证单条数据

    Args:
        data: 原始数据
        model: pydantic 模型类

    Returns:
        (验证后的模型实例或 None, 错误列表)
    """
    errors = []
    try:
        instance = model.model_validate(data)
        return instance, []
    except ValidationError as e:
        for error in e.errors():
            errors.append({
                "loc": ".".join(str(x) for x in error["loc"]),
                "msg": error["msg"],
                "type": error["type"],
                "input": error.get("input"),
            })
        return None, errors


def validate_batch(
    items: list[dict[str, Any]],
    model: Type[T],
    skip_errors: bool = True,
) -> CleaningResult:
    """批量验证数据

    Args:
        items: 数据列表
        model: pydantic 模型类
        skip_errors: 是否跳过错误

    Returns:
        CleaningResult 对象
    """
    result = CleaningResult(total=len(items))

    for i, item in enumerate(items):
        instance, errors = validate_item(item, model)

        if instance:
            result.success += 1
            result.cleaned_data.append(instance.model_dump())
        else:
            result.failed += 1
            result.errors.append({
                "index": i,
                "data": item,
                "errors": errors,
            })

            if not skip_errors:
                raise ValueError(f"Validation failed at index {i}: {errors}")

    return result


def validate_users(
    items: list[dict[str, Any]],
    skip_errors: bool = True,
) -> CleaningResult:
    """验证用户数据

    Args:
        items: 用户数据列表
        skip_errors: 是否跳过错误

    Returns:
        CleaningResult 对象
    """
    return validate_batch(items, User, skip_errors)


def format_validation_errors(errors: list[dict[str, Any]]) -> str:
    """格式化验证错误

    Args:
        errors: 错误列表

    Returns:
        格式化的错误消息
    """
    lines = []
    for error in errors:
        index = error.get("index", "?")
        field_errors = error.get("errors", [])
        for field_error in field_errors:
            loc = field_error.get("loc", "unknown")
            msg = field_error.get("msg", "unknown error")
            lines.append(f"  Row {index}: {loc} - {msg}")
    return "\n".join(lines)


class DataValidator:
    """数据验证器类

    封装验证逻辑，支持自定义规则。
    """

    def __init__(self, model: Type[T]):
        """初始化验证器

        Args:
            model: pydantic 模型类
        """
        self.model = model
        self.pre_validators: list = []
        self.post_validators: list = []

    def add_pre_validator(self, validator) -> "DataValidator":
        """添加预验证器（在 pydantic 验证前）"""
        self.pre_validators.append(validator)
        return self

    def add_post_validator(self, validator) -> "DataValidator":
        """添加后验证器（在 pydantic 验证后）"""
        self.post_validators.append(validator)
        return self

    def validate(self, data: dict[str, Any]) -> tuple[T | None, list[str]]:
        """验证单条数据

        Returns:
            (验证后的模型实例或 None, 错误消息列表)
        """
        errors = []

        # 预验证
        for validator in self.pre_validators:
            try:
                data = validator(data)
            except Exception as e:
                errors.append(f"Pre-validation error: {e}")
                return None, errors

        # pydantic 验证
        try:
            instance = self.model.model_validate(data)
        except ValidationError as e:
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                errors.append(f"{loc}: {error['msg']}")
            return None, errors

        # 后验证
        for validator in self.post_validators:
            try:
                instance = validator(instance)
            except Exception as e:
                errors.append(f"Post-validation error: {e}")
                return None, errors

        return instance, []

    def validate_batch(
        self,
        items: list[dict[str, Any]],
    ) -> CleaningResult:
        """批量验证"""
        result = CleaningResult(total=len(items))

        for i, item in enumerate(items):
            instance, errors = self.validate(item)

            if instance:
                result.success += 1
                result.cleaned_data.append(instance.model_dump())
            else:
                result.failed += 1
                result.errors.append({
                    "index": i,
                    "errors": errors,
                })

        return result

