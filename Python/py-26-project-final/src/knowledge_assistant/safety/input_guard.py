"""
输入安全守卫

提供:
- 输入检查
- 注入检测
- 长度验证
- 净化处理
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

import structlog

logger = structlog.get_logger()


class InjectionType(str, Enum):
    """注入类型"""
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    ROLE_PLAY = "role_play"
    SYSTEM_OVERRIDE = "system_override"


class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InjectionResult:
    """注入检测结果"""
    is_injection: bool
    injection_type: Optional[InjectionType] = None
    risk_level: RiskLevel = RiskLevel.LOW
    confidence: float = 0.0
    matched_pattern: Optional[str] = None

    @property
    def should_block(self) -> bool:
        return self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)


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
        return self.is_safe or (
            self.injection_result is not None and
            not self.injection_result.should_block
        )


class InjectionDetector:
    """
    注入检测器
    
    检测 prompt injection 和 jailbreak 尝试
    """
    
    # 高风险模式
    HIGH_RISK_PATTERNS = [
        # 系统提示覆盖
        (r'ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)', InjectionType.SYSTEM_OVERRIDE),
        (r'disregard\s+(all\s+)?(previous|above|prior)', InjectionType.SYSTEM_OVERRIDE),
        (r'forget\s+(everything|all)\s+(you|that)', InjectionType.SYSTEM_OVERRIDE),
        (r'new\s+system\s+prompt', InjectionType.SYSTEM_OVERRIDE),
        (r'override\s+(system|instructions?)', InjectionType.SYSTEM_OVERRIDE),
        
        # 角色扮演尝试
        (r'you\s+are\s+now\s+(?:a|an|the)\s+\w+', InjectionType.ROLE_PLAY),
        (r'pretend\s+(?:to\s+be|you\s+are)', InjectionType.ROLE_PLAY),
        (r'act\s+as\s+(?:if\s+you|a|an)', InjectionType.ROLE_PLAY),
        (r'roleplay\s+as', InjectionType.ROLE_PLAY),
        
        # Jailbreak 尝试
        (r'do\s+anything\s+now', InjectionType.JAILBREAK),
        (r'DAN\s+mode', InjectionType.JAILBREAK),
        (r'developer\s+mode', InjectionType.JAILBREAK),
        (r'bypass\s+(?:safety|restrictions?|filters?)', InjectionType.JAILBREAK),
    ]
    
    # 中等风险模式
    MEDIUM_RISK_PATTERNS = [
        (r'\[system\]', InjectionType.SYSTEM_OVERRIDE),
        (r'<<SYS>>', InjectionType.SYSTEM_OVERRIDE),
        (r'<\|im_start\|>system', InjectionType.SYSTEM_OVERRIDE),
        (r'respond\s+only\s+with', InjectionType.PROMPT_INJECTION),
        (r'output\s+only', InjectionType.PROMPT_INJECTION),
    ]

    def detect(self, text: str) -> InjectionResult:
        """
        检测注入
        
        Args:
            text: 输入文本
        
        Returns:
            InjectionResult
        """
        text_lower = text.lower()
        
        # 检查高风险模式
        for pattern, injection_type in self.HIGH_RISK_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return InjectionResult(
                    is_injection=True,
                    injection_type=injection_type,
                    risk_level=RiskLevel.HIGH,
                    confidence=0.9,
                    matched_pattern=pattern,
                )
        
        # 检查中等风险模式
        for pattern, injection_type in self.MEDIUM_RISK_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return InjectionResult(
                    is_injection=True,
                    injection_type=injection_type,
                    risk_level=RiskLevel.MEDIUM,
                    confidence=0.7,
                    matched_pattern=pattern,
                )
        
        return InjectionResult(
            is_injection=False,
            risk_level=RiskLevel.LOW,
        )


class InputGuard:
    """
    输入安全守卫
    
    提供多层次的输入安全检查
    
    Usage:
        guard = InputGuard()
        
        result = guard.check("Hello, how are you?")
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
        self._custom_validators: List[Callable[[str], Tuple[bool, str]]] = []

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
            issues.append(f"输入太短（最少 {self.min_length} 字符）")
            is_safe = False
        
        if len(text) > self.max_length:
            issues.append(f"输入太长（最多 {self.max_length} 字符）")
            is_safe = False
        
        # 2. 注入检测
        if self.enable_injection_detection:
            injection_result = self._injection_detector.detect(text)
            if injection_result.is_injection:
                issues.append(f"检测到潜在的注入尝试: {injection_result.injection_type.value}")
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
        
        return sanitized

    def add_validator(self, validator: Callable[[str], Tuple[bool, str]]):
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
        self._requests: dict[str, List[float]] = {}

    def check(self, user_id: str) -> Tuple[bool, int]:
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


