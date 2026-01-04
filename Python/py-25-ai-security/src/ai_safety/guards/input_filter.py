"""
输入过滤器

提供输入安全检查和过滤
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import structlog

from ai_safety.guards.injection import InjectionDetector, InjectionResult, RiskLevel

logger = structlog.get_logger()


@dataclass
class InputCheckResult:
    """输入检查结果"""
    is_safe: bool
    input_text: str
    sanitized_text: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    injection_result: Optional[InjectionResult] = None

    @property
    def should_process(self) -> bool:
        """是否应该处理"""
        return self.is_safe or (
            self.injection_result is not None and
            not self.injection_result.should_block
        )


class InputFilter:
    """
    输入过滤器
    
    提供多层次的输入安全检查
    
    Usage:
        filter = InputFilter()
        
        result = filter.check("Hello, how are you?")
        if result.is_safe:
            process(result.input_text)
        else:
            handle_unsafe(result.issues)
    """

    def __init__(
        self,
        max_length: int = 10000,
        min_length: int = 1,
        enable_injection_detection: bool = True,
        enable_sanitization: bool = True,
    ):
        self.max_length = max_length
        self.min_length = min_length
        self.enable_injection_detection = enable_injection_detection
        self.enable_sanitization = enable_sanitization
        
        self._injection_detector = InjectionDetector()
        self._custom_validators: List[Callable[[str], tuple[bool, str]]] = []

    def check(self, text: str) -> InputCheckResult:
        """
        检查输入
        
        Args:
            text: 输入文本
        
        Returns:
            InputCheckResult
        """
        issues = []
        is_safe = True
        sanitized = text
        injection_result = None
        
        # 1. 长度检查
        if len(text) < self.min_length:
            issues.append(f"Input too short (min: {self.min_length})")
            is_safe = False
        
        if len(text) > self.max_length:
            issues.append(f"Input too long (max: {self.max_length})")
            is_safe = False
        
        # 2. 注入检测
        if self.enable_injection_detection:
            injection_result = self._injection_detector.detect(text)
            if injection_result.is_injection:
                issues.append(f"Injection detected: {injection_result.injection_type.value}")
                if injection_result.should_block:
                    is_safe = False
        
        # 3. 净化处理
        if self.enable_sanitization:
            sanitized = self._sanitize(text)
        
        # 4. 自定义验证
        for validator in self._custom_validators:
            valid, message = validator(text)
            if not valid:
                issues.append(message)
                is_safe = False
        
        if not is_safe:
            logger.warning("input_check_failed", issues=issues)
        
        return InputCheckResult(
            is_safe=is_safe,
            input_text=text,
            sanitized_text=sanitized,
            issues=issues,
            injection_result=injection_result,
        )

    def _sanitize(self, text: str) -> str:
        """净化输入"""
        sanitized = text
        
        # 移除零宽字符
        sanitized = re.sub(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f]', '', sanitized)
        
        # 移除控制字符（保留换行和制表符）
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
        
        # 规范化空白
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized

    def add_validator(self, validator: Callable[[str], tuple[bool, str]]):
        """添加自定义验证器"""
        self._custom_validators.append(validator)


class RateLimiter:
    """
    速率限制器
    
    基于滑动窗口的速率限制
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = {}

    def check(self, user_id: str) -> tuple[bool, int]:
        """
        检查速率限制
        
        Args:
            user_id: 用户标识
        
        Returns:
            (is_allowed, remaining_requests)
        """
        import time
        
        now = time.time()
        window_start = now - self.window_seconds
        
        if user_id not in self._requests:
            self._requests[user_id] = []
        
        # 清理过期请求
        self._requests[user_id] = [
            t for t in self._requests[user_id] if t > window_start
        ]
        
        # 检查是否超限
        current_count = len(self._requests[user_id])
        remaining = max(0, self.max_requests - current_count)
        
        if current_count >= self.max_requests:
            return False, remaining
        
        # 记录请求
        self._requests[user_id].append(now)
        return True, remaining - 1


class TokenBudgetManager:
    """
    Token 预算管理
    
    限制用户的 token 使用量
    """

    def __init__(self, daily_limit: int = 100000):
        self.daily_limit = daily_limit
        self._usage: Dict[str, Dict[str, int]] = {}

    def check_budget(self, user_id: str, estimated_tokens: int) -> tuple[bool, int]:
        """
        检查预算
        
        Args:
            user_id: 用户标识
            estimated_tokens: 预估 token 数
        
        Returns:
            (is_within_budget, remaining_tokens)
        """
        from datetime import date
        
        today = str(date.today())
        
        if user_id not in self._usage:
            self._usage[user_id] = {}
        
        if today not in self._usage[user_id]:
            self._usage[user_id] = {today: 0}  # 重置旧数据
        
        used = self._usage[user_id].get(today, 0)
        remaining = max(0, self.daily_limit - used)
        
        if estimated_tokens > remaining:
            return False, remaining
        
        return True, remaining

    def record_usage(self, user_id: str, tokens: int):
        """记录使用量"""
        from datetime import date
        
        today = str(date.today())
        
        if user_id not in self._usage:
            self._usage[user_id] = {}
        
        self._usage[user_id][today] = self._usage[user_id].get(today, 0) + tokens

    def get_usage(self, user_id: str) -> int:
        """获取今日使用量"""
        from datetime import date
        
        today = str(date.today())
        return self._usage.get(user_id, {}).get(today, 0)


