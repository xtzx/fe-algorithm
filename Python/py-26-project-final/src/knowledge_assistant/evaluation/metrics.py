"""
评测指标

提供多种评测指标
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()


class MetricType(str, Enum):
    """指标类型"""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    FAITHFULNESS = "faithfulness"
    HARMLESSNESS = "harmlessness"
    FLUENCY = "fluency"
    COHERENCE = "coherence"


@dataclass
class MetricResult:
    """指标结果"""
    metric_type: MetricType
    score: float
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class Metrics:
    """
    评测指标计算
    
    提供多种评测指标:
    - 准确性: 与预期输出的匹配度
    - 相关性: 回答与问题的相关程度
    - 忠实度: 回答是否基于给定上下文
    - 无害性: 回答是否安全无害
    
    Usage:
        metrics = Metrics()
        
        # 计算准确性
        result = metrics.accuracy(output, expected)
        
        # 计算相关性
        result = metrics.relevance(question, answer)
    """

    def accuracy(
        self,
        output: str,
        expected: str,
        method: str = "fuzzy",
    ) -> MetricResult:
        """
        计算准确性
        
        Args:
            output: 模型输出
            expected: 预期输出
            method: 匹配方法 ("exact", "fuzzy", "contains")
        
        Returns:
            MetricResult
        """
        score = 0.0
        details = {"method": method}
        
        if method == "exact":
            score = 1.0 if output.strip() == expected.strip() else 0.0
        
        elif method == "contains":
            # 检查预期输出是否包含在输出中
            score = 1.0 if expected.strip().lower() in output.lower() else 0.0
        
        elif method == "fuzzy":
            # 模糊匹配：基于词重叠
            output_words = set(self._tokenize(output))
            expected_words = set(self._tokenize(expected))
            
            if not expected_words:
                score = 1.0 if not output_words else 0.0
            else:
                intersection = output_words & expected_words
                score = len(intersection) / len(expected_words)
            
            details["overlap_ratio"] = score
        
        return MetricResult(
            metric_type=MetricType.ACCURACY,
            score=score,
            details=details,
        )

    def relevance(self, question: str, answer: str) -> MetricResult:
        """
        计算相关性
        
        基于关键词重叠和语义相似度
        
        Args:
            question: 问题
            answer: 回答
        
        Returns:
            MetricResult
        """
        # 简单实现：基于关键词重叠
        question_words = set(self._tokenize(question))
        answer_words = set(self._tokenize(answer))
        
        if not question_words:
            return MetricResult(
                metric_type=MetricType.RELEVANCE,
                score=1.0,
                details={"method": "keyword_overlap"},
            )
        
        # 计算问题词在回答中的覆盖率
        covered = question_words & answer_words
        coverage = len(covered) / len(question_words)
        
        # 检查是否包含"不知道"等回避性回答
        evasive_patterns = [
            r"不知道",
            r"没有.*信息",
            r"无法.*回答",
            r"i don't know",
            r"not sure",
        ]
        
        is_evasive = any(
            re.search(p, answer, re.IGNORECASE)
            for p in evasive_patterns
        )
        
        # 如果是回避性回答，相关性降低
        if is_evasive:
            coverage *= 0.5
        
        return MetricResult(
            metric_type=MetricType.RELEVANCE,
            score=min(1.0, coverage),
            details={
                "method": "keyword_overlap",
                "coverage": coverage,
                "is_evasive": is_evasive,
            },
        )

    def faithfulness(self, answer: str, context: str) -> MetricResult:
        """
        计算忠实度
        
        检查回答是否基于给定上下文
        
        Args:
            answer: 回答
            context: 上下文
        
        Returns:
            MetricResult
        """
        answer_sentences = self._split_sentences(answer)
        
        if not answer_sentences:
            return MetricResult(
                metric_type=MetricType.FAITHFULNESS,
                score=1.0,
                details={"method": "sentence_grounding"},
            )
        
        # 检查每个句子是否有上下文支持
        grounded_count = 0
        context_lower = context.lower()
        
        for sentence in answer_sentences:
            # 检查句子中的关键词是否在上下文中
            words = self._tokenize(sentence)
            if not words:
                continue
            
            # 计算词在上下文中的覆盖率
            covered = sum(1 for w in words if w.lower() in context_lower)
            if covered / len(words) > 0.3:  # 30% 以上的词在上下文中
                grounded_count += 1
        
        score = grounded_count / len(answer_sentences) if answer_sentences else 1.0
        
        return MetricResult(
            metric_type=MetricType.FAITHFULNESS,
            score=score,
            details={
                "method": "sentence_grounding",
                "grounded_sentences": grounded_count,
                "total_sentences": len(answer_sentences),
            },
        )

    def harmlessness(self, text: str) -> MetricResult:
        """
        计算无害性
        
        检测文本中的有害内容
        
        Args:
            text: 文本
        
        Returns:
            MetricResult
        """
        from knowledge_assistant.safety.output_guard import OutputGuard
        
        guard = OutputGuard()
        result = guard.moderate(text)
        
        # 计算分数（无有害内容 = 1.0）
        if result.is_safe:
            score = 1.0
        else:
            # 根据类别数量降低分数
            flagged_count = sum(1 for v in result.categories.values() if v)
            score = max(0.0, 1.0 - (flagged_count * 0.25))
        
        return MetricResult(
            metric_type=MetricType.HARMLESSNESS,
            score=score,
            details={
                "is_safe": result.is_safe,
                "categories": result.categories,
                "reason": result.reason,
            },
        )

    def fluency(self, text: str) -> MetricResult:
        """
        计算流畅度
        
        基于简单启发式规则
        
        Args:
            text: 文本
        
        Returns:
            MetricResult
        """
        score = 1.0
        issues = []
        
        # 检查文本长度
        if len(text) < 10:
            score -= 0.3
            issues.append("太短")
        
        # 检查重复
        sentences = self._split_sentences(text)
        if sentences:
            unique_sentences = set(s.lower().strip() for s in sentences)
            repetition_ratio = 1 - (len(unique_sentences) / len(sentences))
            if repetition_ratio > 0.3:
                score -= 0.2
                issues.append("重复内容过多")
        
        # 检查是否有未完成的句子
        if text and text[-1] not in ".!?。！？":
            score -= 0.1
            issues.append("句子未完成")
        
        return MetricResult(
            metric_type=MetricType.FLUENCY,
            score=max(0.0, score),
            details={"issues": issues},
        )

    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 中英文混合分词
        words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text.lower())
        # 过滤停用词
        stop_words = {"的", "是", "在", "了", "和", "a", "the", "is", "are", "to"}
        return [w for w in words if w not in stop_words and len(w) > 1]

    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        sentences = re.split(r'[.!?。！？]+', text)
        return [s.strip() for s in sentences if s.strip()]


