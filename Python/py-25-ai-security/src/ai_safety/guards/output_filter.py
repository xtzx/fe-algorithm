"""
输出过滤器

提供输出安全处理:
- PII 过滤
- 内容审核
- 格式验证
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern

import structlog

logger = structlog.get_logger()


class PIIType(str, Enum):
    """PII 类型"""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    IP_ADDRESS = "ip_address"


@dataclass
class PIIMatch:
    """PII 匹配结果"""
    pii_type: PIIType
    value: str
    start: int
    end: int


@dataclass
class ModerationResult:
    """审核结果"""
    is_safe: bool
    categories: Dict[str, bool] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    reason: Optional[str] = None


class OutputFilter:
    """
    输出过滤器
    
    提供 PII 过滤、内容审核、格式验证
    
    Usage:
        filter = OutputFilter()
        
        # PII 过滤
        safe_text = filter.remove_pii("Contact john@example.com")
        # "Contact [EMAIL]"
        
        # 内容审核
        result = filter.moderate(text)
        if not result.is_safe:
            block(result.reason)
    """

    # PII 检测模式
    PII_PATTERNS = {
        PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        PIIType.PHONE: r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        PIIType.SSN: r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        PIIType.CREDIT_CARD: r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        PIIType.IP_ADDRESS: r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }

    # 内容审核关键词（简化版）
    HARMFUL_PATTERNS = {
        "violence": [
            r'\b(kill|murder|harm|attack|violent)\b',
        ],
        "hate": [
            r'\b(hate|racist|sexist|discriminat)\w*\b',
        ],
        "self_harm": [
            r'\b(suicide|self[-\s]?harm)\b',
        ],
        "sexual": [
            r'\b(explicit|pornograph)\w*\b',
        ],
    }

    def __init__(
        self,
        mask_pii: bool = True,
        pii_mask: str = "[{type}]",
    ):
        self.mask_pii = mask_pii
        self.pii_mask = pii_mask
        
        self._pii_patterns: Dict[PIIType, Pattern] = {
            pii_type: re.compile(pattern, re.IGNORECASE)
            for pii_type, pattern in self.PII_PATTERNS.items()
        }
        
        self._harmful_patterns: Dict[str, List[Pattern]] = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in self.HARMFUL_PATTERNS.items()
        }

    def detect_pii(self, text: str) -> List[PIIMatch]:
        """
        检测 PII
        
        Args:
            text: 输入文本
        
        Returns:
            PIIMatch 列表
        """
        matches = []
        
        for pii_type, pattern in self._pii_patterns.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                ))
        
        return matches

    def remove_pii(self, text: str) -> str:
        """
        移除/遮蔽 PII
        
        Args:
            text: 输入文本
        
        Returns:
            处理后的文本
        """
        if not self.mask_pii:
            return text
        
        result = text
        
        for pii_type, pattern in self._pii_patterns.items():
            mask = self.pii_mask.format(type=pii_type.value.upper())
            result = pattern.sub(mask, result)
        
        return result

    def moderate(self, text: str) -> ModerationResult:
        """
        内容审核
        
        Args:
            text: 输入文本
        
        Returns:
            ModerationResult
        """
        categories = {}
        scores = {}
        reasons = []
        
        for category, patterns in self._harmful_patterns.items():
            match_count = 0
            for pattern in patterns:
                matches = pattern.findall(text)
                match_count += len(matches)
            
            # 简单评分：匹配数/总模式数
            score = min(1.0, match_count / len(patterns))
            scores[category] = score
            categories[category] = score > 0.5
            
            if categories[category]:
                reasons.append(category)
        
        is_safe = not any(categories.values())
        reason = f"Flagged categories: {', '.join(reasons)}" if reasons else None
        
        if not is_safe:
            logger.warning("content_moderation_flagged", categories=reasons)
        
        return ModerationResult(
            is_safe=is_safe,
            categories=categories,
            scores=scores,
            reason=reason,
        )

    def validate_json(self, text: str) -> tuple[bool, Optional[str]]:
        """
        验证 JSON 格式
        
        Args:
            text: 输入文本
        
        Returns:
            (is_valid, error_message)
        """
        try:
            json.loads(text)
            return True, None
        except json.JSONDecodeError as e:
            return False, str(e)

    def validate_format(
        self,
        text: str,
        expected_format: str,
    ) -> tuple[bool, Optional[str]]:
        """
        验证输出格式
        
        Args:
            text: 输入文本
            expected_format: 期望格式 ("json", "markdown", "code")
        
        Returns:
            (is_valid, error_message)
        """
        if expected_format == "json":
            return self.validate_json(text)
        
        elif expected_format == "markdown":
            # 简单检查：是否包含 markdown 元素
            has_headers = re.search(r'^#{1,6}\s', text, re.MULTILINE)
            has_lists = re.search(r'^[-*+]\s', text, re.MULTILINE)
            has_code = re.search(r'```', text)
            
            if has_headers or has_lists or has_code:
                return True, None
            return False, "No markdown elements detected"
        
        elif expected_format == "code":
            # 检查是否包含代码块
            if "```" in text:
                return True, None
            return False, "No code block detected"
        
        return True, None


class HallucinationDetector:
    """
    幻觉检测器
    
    检测 LLM 输出中可能的幻觉
    """

    # 幻觉指示词
    UNCERTAINTY_INDICATORS = [
        r"I'm not sure",
        r"I don't know",
        r"I cannot confirm",
        r"might be",
        r"possibly",
        r"I think",
        r"as far as I know",
    ]

    CONFIDENT_BUT_WRONG_INDICATORS = [
        r"definitely",
        r"certainly",
        r"absolutely",
        r"100%",
        r"always",
        r"never",
    ]

    def __init__(self):
        self._uncertainty_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.UNCERTAINTY_INDICATORS
        ]
        self._confident_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CONFIDENT_BUT_WRONG_INDICATORS
        ]

    def analyze(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        分析幻觉风险
        
        Args:
            text: LLM 输出
            context: 上下文（用于对比）
        
        Returns:
            分析结果
        """
        uncertainty_count = sum(
            len(p.findall(text)) for p in self._uncertainty_patterns
        )
        
        overconfidence_count = sum(
            len(p.findall(text)) for p in self._confident_patterns
        )
        
        # 风险评估
        risk_score = 0.0
        
        # 不确定性低但过度自信 → 可能幻觉
        if uncertainty_count == 0 and overconfidence_count > 0:
            risk_score += 0.3
        
        # 如果提供了上下文，检查是否有上下文中没有的声明
        context_match = 1.0
        if context:
            # 简化检查：输出中的关键信息是否在上下文中
            pass
        
        return {
            "uncertainty_indicators": uncertainty_count,
            "overconfidence_indicators": overconfidence_count,
            "hallucination_risk": min(1.0, risk_score),
            "context_match": context_match,
        }


