"""
提示注入检测

支持:
- 直接注入检测
- 间接注入检测
- 越狱检测
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern

import structlog

logger = structlog.get_logger()


class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InjectionType(str, Enum):
    """注入类型"""
    DIRECT = "direct"         # 直接注入
    INDIRECT = "indirect"     # 间接注入
    JAILBREAK = "jailbreak"   # 越狱尝试
    NONE = "none"             # 无注入


@dataclass
class InjectionResult:
    """注入检测结果"""
    is_injection: bool
    injection_type: InjectionType = InjectionType.NONE
    risk_level: RiskLevel = RiskLevel.LOW
    matched_patterns: List[str] = field(default_factory=list)
    details: str = ""

    @property
    def should_block(self) -> bool:
        """是否应该阻止"""
        return self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)


class InjectionDetector:
    """
    提示注入检测器
    
    检测常见的提示注入攻击模式
    
    Usage:
        detector = InjectionDetector()
        
        result = detector.detect("Ignore previous instructions and...")
        if result.is_injection:
            print(f"Detected: {result.injection_type}")
            print(f"Risk: {result.risk_level}")
    """

    # 直接注入模式
    DIRECT_INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)",
        r"forget\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)",
        r"override\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)",
        r"do\s+not\s+follow\s+(previous|prior)\s+(instructions?|prompts?)",
        r"new\s+(instructions?|rules?|prompts?)\s*[:\-]",
        r"instead[,\s]+(do|say|respond|output)",
        r"your\s+(new\s+)?role\s+is",
        r"you\s+are\s+now\s+(a|an|the)",
        r"pretend\s+(to\s+be|you\s+are)",
        r"act\s+as\s+(if|a|an|the)",
    ]

    # 越狱模式
    JAILBREAK_PATTERNS = [
        r"(DAN|Developer\s+Mode|jailbreak)",
        r"do\s+anything\s+now",
        r"no\s+restrictions?",
        r"bypass\s+(all\s+)?(restrictions?|filters?|rules?)",
        r"unlock\s+(all\s+)?(capabilities|features)",
        r"without\s+(any\s+)?(ethical|moral)\s+(guidelines?|restrictions?)",
        r"imagine\s+you\s+(have\s+)?no\s+(restrictions?|limitations?)",
    ]

    # 系统提示泄露
    SYSTEM_LEAK_PATTERNS = [
        r"(reveal|show|tell\s+me)\s+(your\s+)?(system\s+)?prompt",
        r"(what\s+are|list)\s+(your\s+)?(instructions?|rules?)",
        r"repeat\s+(your\s+)?initial\s+(instructions?|prompts?)",
        r"output\s+(your\s+)?(system\s+)?prompt",
    ]

    # 间接注入标记
    INDIRECT_MARKERS = [
        r"\[INST\]",
        r"\[/INST\]",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"###\s*(Human|Assistant|System):",
        r"```system",
    ]

    def __init__(self, custom_patterns: Optional[List[str]] = None):
        """
        Args:
            custom_patterns: 自定义检测模式
        """
        self._patterns: Dict[str, List[Pattern]] = {
            "direct": [re.compile(p, re.IGNORECASE) for p in self.DIRECT_INJECTION_PATTERNS],
            "jailbreak": [re.compile(p, re.IGNORECASE) for p in self.JAILBREAK_PATTERNS],
            "system_leak": [re.compile(p, re.IGNORECASE) for p in self.SYSTEM_LEAK_PATTERNS],
            "indirect": [re.compile(p, re.IGNORECASE) for p in self.INDIRECT_MARKERS],
        }
        
        if custom_patterns:
            self._patterns["custom"] = [
                re.compile(p, re.IGNORECASE) for p in custom_patterns
            ]

    def detect(self, text: str) -> InjectionResult:
        """
        检测提示注入
        
        Args:
            text: 输入文本
        
        Returns:
            InjectionResult
        """
        matched = []
        injection_type = InjectionType.NONE
        risk_level = RiskLevel.LOW
        
        # 检测直接注入
        for pattern in self._patterns["direct"]:
            if pattern.search(text):
                matched.append(f"direct:{pattern.pattern}")
                injection_type = InjectionType.DIRECT
                risk_level = RiskLevel.HIGH
        
        # 检测越狱
        for pattern in self._patterns["jailbreak"]:
            if pattern.search(text):
                matched.append(f"jailbreak:{pattern.pattern}")
                injection_type = InjectionType.JAILBREAK
                risk_level = RiskLevel.CRITICAL
        
        # 检测系统提示泄露
        for pattern in self._patterns["system_leak"]:
            if pattern.search(text):
                matched.append(f"system_leak:{pattern.pattern}")
                if risk_level.value < RiskLevel.MEDIUM.value:
                    risk_level = RiskLevel.MEDIUM

        is_injection = len(matched) > 0
        
        if is_injection:
            logger.warning(
                "injection_detected",
                type=injection_type.value,
                risk=risk_level.value,
                patterns=matched[:3],
            )
        
        return InjectionResult(
            is_injection=is_injection,
            injection_type=injection_type,
            risk_level=risk_level,
            matched_patterns=matched,
            details=f"Matched {len(matched)} patterns" if matched else "",
        )

    def detect_in_context(self, context: str) -> InjectionResult:
        """
        检测上下文中的间接注入
        
        用于检测来自外部数据（如 RAG 检索结果）的注入
        
        Args:
            context: 上下文文本
        
        Returns:
            InjectionResult
        """
        matched = []
        
        # 检测间接注入标记
        for pattern in self._patterns["indirect"]:
            if pattern.search(context):
                matched.append(f"indirect:{pattern.pattern}")
        
        # 同时检测常规注入
        regular_result = self.detect(context)
        matched.extend(regular_result.matched_patterns)
        
        is_injection = len(matched) > 0
        
        return InjectionResult(
            is_injection=is_injection,
            injection_type=InjectionType.INDIRECT if is_injection else InjectionType.NONE,
            risk_level=RiskLevel.HIGH if is_injection else RiskLevel.LOW,
            matched_patterns=matched,
            details="Potential injection in external data" if is_injection else "",
        )

    def detect_jailbreak(self, text: str) -> InjectionResult:
        """
        专门检测越狱尝试
        
        Args:
            text: 输入文本
        
        Returns:
            InjectionResult
        """
        matched = []
        
        for pattern in self._patterns["jailbreak"]:
            if pattern.search(text):
                matched.append(pattern.pattern)
        
        is_jailbreak = len(matched) > 0
        
        return InjectionResult(
            is_injection=is_jailbreak,
            injection_type=InjectionType.JAILBREAK if is_jailbreak else InjectionType.NONE,
            risk_level=RiskLevel.CRITICAL if is_jailbreak else RiskLevel.LOW,
            matched_patterns=matched,
            details="Jailbreak attempt detected" if is_jailbreak else "",
        )

    def add_pattern(self, pattern: str, category: str = "custom"):
        """添加自定义检测模式"""
        if category not in self._patterns:
            self._patterns[category] = []
        self._patterns[category].append(re.compile(pattern, re.IGNORECASE))


class ContentFilter:
    """
    内容过滤器
    
    过滤敏感内容
    """

    SENSITIVE_TOPICS = [
        r"(make|create|build)\s+(a\s+)?(bomb|weapon|explosive)",
        r"(how\s+to|steps\s+to)\s+(hack|exploit|attack)",
        r"(illegal|illicit)\s+(drugs?|substances?)",
        r"(child|minor)\s+(abuse|exploitation)",
    ]

    def __init__(self):
        self._patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SENSITIVE_TOPICS
        ]

    def filter(self, text: str) -> tuple[bool, List[str]]:
        """
        过滤敏感内容
        
        Returns:
            (is_safe, matched_patterns)
        """
        matched = []
        for pattern in self._patterns:
            if pattern.search(text):
                matched.append(pattern.pattern)
        
        return len(matched) == 0, matched


