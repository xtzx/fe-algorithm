"""
评测指标

支持:
- 准确性 (Accuracy)
- 相关性 (Relevance)
- 忠实度 (Faithfulness)
- 无害性 (Harmlessness)
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger()


class MetricType(str, Enum):
    """指标类型"""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    FAITHFULNESS = "faithfulness"
    HARMLESSNESS = "harmlessness"
    COHERENCE = "coherence"
    FLUENCY = "fluency"


@dataclass
class MetricResult:
    """指标结果"""
    metric_type: MetricType
    score: float  # 0.0 - 1.0
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class Metrics:
    """
    评测指标计算器
    
    提供多种评测指标的计算
    
    Usage:
        metrics = Metrics()
        
        # 计算准确性
        accuracy = metrics.accuracy(
            prediction="Python is a programming language",
            reference="Python is a high-level programming language",
        )
        
        # 计算相关性
        relevance = metrics.relevance(
            question="What is Python?",
            answer="Python is a programming language",
        )
    """

    def accuracy(
        self,
        prediction: str,
        reference: str,
        method: str = "exact",
    ) -> MetricResult:
        """
        计算准确性
        
        Args:
            prediction: 预测结果
            reference: 参考答案
            method: 匹配方法 ("exact", "contains", "fuzzy")
        
        Returns:
            MetricResult
        """
        pred_lower = prediction.lower().strip()
        ref_lower = reference.lower().strip()
        
        if method == "exact":
            score = 1.0 if pred_lower == ref_lower else 0.0
        
        elif method == "contains":
            score = 1.0 if ref_lower in pred_lower else 0.0
        
        elif method == "fuzzy":
            # 简单的词重叠率
            pred_words = set(pred_lower.split())
            ref_words = set(ref_lower.split())
            
            if not ref_words:
                score = 0.0
            else:
                overlap = len(pred_words & ref_words)
                score = overlap / len(ref_words)
        
        else:
            score = 0.0
        
        return MetricResult(
            metric_type=MetricType.ACCURACY,
            score=score,
            details={"method": method},
        )

    def relevance(
        self,
        question: str,
        answer: str,
    ) -> MetricResult:
        """
        计算相关性
        
        基于问题关键词在答案中的覆盖率
        
        Args:
            question: 问题
            answer: 答案
        
        Returns:
            MetricResult
        """
        # 提取问题关键词（移除停用词）
        stopwords = {"what", "is", "are", "the", "a", "an", "how", "why", "when", "where", "who"}
        
        question_words = set(question.lower().split()) - stopwords
        answer_lower = answer.lower()
        
        if not question_words:
            return MetricResult(
                metric_type=MetricType.RELEVANCE,
                score=0.5,
                details={"reason": "No keywords in question"},
            )
        
        # 计算关键词覆盖率
        covered = sum(1 for word in question_words if word in answer_lower)
        score = covered / len(question_words)
        
        return MetricResult(
            metric_type=MetricType.RELEVANCE,
            score=score,
            details={"keywords": list(question_words), "covered": covered},
        )

    def faithfulness(
        self,
        answer: str,
        context: str,
    ) -> MetricResult:
        """
        计算忠实度
        
        检查答案是否基于提供的上下文
        
        Args:
            answer: 答案
            context: 上下文
        
        Returns:
            MetricResult
        """
        # 提取答案中的声明（简化：使用句子）
        answer_sentences = re.split(r'[.!?]', answer)
        answer_sentences = [s.strip() for s in answer_sentences if s.strip()]
        
        if not answer_sentences:
            return MetricResult(
                metric_type=MetricType.FAITHFULNESS,
                score=1.0,
                details={"reason": "No sentences to verify"},
            )
        
        context_lower = context.lower()
        
        # 检查每个句子中的关键信息是否在上下文中
        supported_count = 0
        for sentence in answer_sentences:
            # 简化检查：句子中的词是否大部分在上下文中
            words = sentence.lower().split()
            if not words:
                continue
            
            matched = sum(1 for w in words if w in context_lower)
            if matched / len(words) > 0.5:
                supported_count += 1
        
        score = supported_count / len(answer_sentences) if answer_sentences else 0.0
        
        return MetricResult(
            metric_type=MetricType.FAITHFULNESS,
            score=score,
            details={
                "total_sentences": len(answer_sentences),
                "supported_sentences": supported_count,
            },
        )

    def harmlessness(
        self,
        text: str,
    ) -> MetricResult:
        """
        计算无害性
        
        检查文本是否包含有害内容
        
        Args:
            text: 文本
        
        Returns:
            MetricResult
        """
        harmful_patterns = [
            r'\b(violence|violent|kill|murder)\b',
            r'\b(hate|racist|sexist)\b',
            r'\b(illegal|illicit)\b',
        ]
        
        text_lower = text.lower()
        harmful_count = 0
        
        for pattern in harmful_patterns:
            matches = re.findall(pattern, text_lower)
            harmful_count += len(matches)
        
        # 反转：越少有害内容，分数越高
        score = max(0.0, 1.0 - harmful_count * 0.2)
        
        return MetricResult(
            metric_type=MetricType.HARMLESSNESS,
            score=score,
            details={"harmful_matches": harmful_count},
        )

    def coherence(
        self,
        text: str,
    ) -> MetricResult:
        """
        计算连贯性
        
        检查文本的结构连贯性
        
        Args:
            text: 文本
        
        Returns:
            MetricResult
        """
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return MetricResult(
                metric_type=MetricType.COHERENCE,
                score=1.0,
                details={"reason": "Single sentence"},
            )
        
        # 检查相邻句子的词重叠（简化的连贯性度量）
        overlaps = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            
            if words1 and words2:
                overlap = len(words1 & words2) / min(len(words1), len(words2))
                overlaps.append(overlap)
        
        score = sum(overlaps) / len(overlaps) if overlaps else 0.5
        
        return MetricResult(
            metric_type=MetricType.COHERENCE,
            score=min(1.0, score * 2),  # 放大
            details={"sentence_count": len(sentences)},
        )


class LLMAsJudge:
    """
    LLM-as-Judge 评测器
    
    使用 LLM 作为评判者评估输出质量
    
    Usage:
        judge = LLMAsJudge(llm_client)
        
        score = judge.evaluate(
            question="What is Python?",
            answer="Python is a programming language",
            criteria=["relevance", "accuracy"],
        )
    """

    EVALUATION_PROMPT = """You are an expert evaluator. Rate the following answer on a scale of 1-5.

Question: {question}
Answer: {answer}
{context_section}
Criteria: {criteria}

Provide your rating in the format:
Score: X
Reason: Your explanation

Be strict but fair in your evaluation."""

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM 客户端（需要 chat 方法）
        """
        self.llm_client = llm_client

    def evaluate(
        self,
        question: str,
        answer: str,
        criteria: List[str],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        使用 LLM 评估
        
        Args:
            question: 问题
            answer: 答案
            criteria: 评估标准
            context: 上下文（可选）
        
        Returns:
            评估结果
        """
        if self.llm_client is None:
            # 模拟评估（用于测试）
            return self._mock_evaluate(question, answer, criteria)
        
        context_section = f"\nContext: {context}" if context else ""
        
        prompt = self.EVALUATION_PROMPT.format(
            question=question,
            answer=answer,
            context_section=context_section,
            criteria=", ".join(criteria),
        )
        
        response = self.llm_client.chat([
            {"role": "user", "content": prompt},
        ])
        
        return self._parse_evaluation(response.content)

    def _mock_evaluate(
        self,
        question: str,
        answer: str,
        criteria: List[str],
    ) -> Dict[str, Any]:
        """模拟评估"""
        # 简单的基于长度和关键词的评分
        score = 3  # 默认中等分数
        
        if len(answer) > 50:
            score += 1
        
        if any(word in answer.lower() for word in question.lower().split()):
            score += 1
        
        return {
            "score": min(5, score),
            "reason": "Mock evaluation based on length and keyword overlap",
            "criteria": criteria,
        }

    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        """解析评估结果"""
        score_match = re.search(r'Score:\s*(\d+)', response)
        reason_match = re.search(r'Reason:\s*(.+)', response, re.DOTALL)
        
        return {
            "score": int(score_match.group(1)) if score_match else 3,
            "reason": reason_match.group(1).strip() if reason_match else "No reason provided",
            "raw_response": response,
        }


