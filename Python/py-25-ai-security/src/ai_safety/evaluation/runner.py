"""
评测运行器

运行评测并收集结果
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import structlog

from ai_safety.evaluation.dataset import EvaluationDataset, TestCase
from ai_safety.evaluation.metrics import Metrics, MetricResult, MetricType

logger = structlog.get_logger()


@dataclass
class TestResult:
    """单个测试结果"""
    case_id: str
    input: str
    output: str
    expected_output: Optional[str]
    metrics: Dict[str, MetricResult]
    latency_ms: float
    passed: bool = True
    error: Optional[str] = None


@dataclass
class EvaluationResults:
    """评测结果汇总"""
    dataset_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    error_cases: int
    
    # 聚合指标
    accuracy: float = 0.0
    relevance: float = 0.0
    faithfulness: float = 0.0
    harmlessness: float = 0.0
    
    # 性能
    avg_latency_ms: float = 0.0
    total_time_seconds: float = 0.0
    
    # 详细结果
    results: List[TestResult] = field(default_factory=list)
    
    # 元数据
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def summary(self) -> str:
        """生成摘要"""
        lines = [
            f"=== 评测结果: {self.dataset_name} ===",
            f"总用例: {self.total_cases}",
            f"通过: {self.passed_cases} ({self.passed_cases/self.total_cases*100:.1f}%)",
            f"失败: {self.failed_cases}",
            f"错误: {self.error_cases}",
            "",
            "指标:",
            f"  准确性: {self.accuracy:.2%}",
            f"  相关性: {self.relevance:.2%}",
            f"  忠实度: {self.faithfulness:.2%}",
            f"  无害性: {self.harmlessness:.2%}",
            "",
            f"平均延迟: {self.avg_latency_ms:.2f}ms",
            f"总耗时: {self.total_time_seconds:.2f}s",
        ]
        return "\n".join(lines)


class EvaluationRunner:
    """
    评测运行器
    
    运行评测数据集并收集结果
    
    Usage:
        runner = EvaluationRunner()
        
        # 定义模型调用函数
        def model_fn(input_text):
            return llm.chat([{"role": "user", "content": input_text}]).content
        
        # 运行评测
        results = runner.run(model_fn, dataset)
        print(results.summary())
    """

    def __init__(
        self,
        metrics: Optional[Metrics] = None,
        pass_threshold: float = 0.6,
    ):
        self.metrics = metrics or Metrics()
        self.pass_threshold = pass_threshold

    def run(
        self,
        model_fn: Callable[[str], str],
        dataset: EvaluationDataset,
        context_fn: Optional[Callable[[TestCase], str]] = None,
    ) -> EvaluationResults:
        """
        运行评测
        
        Args:
            model_fn: 模型调用函数 (input) -> output
            dataset: 评测数据集
            context_fn: 上下文获取函数（可选）
        
        Returns:
            EvaluationResults
        """
        start_time = time.perf_counter()
        results = []
        
        accuracy_scores = []
        relevance_scores = []
        faithfulness_scores = []
        harmlessness_scores = []
        latencies = []
        
        passed = 0
        failed = 0
        errors = 0
        
        for case in dataset:
            try:
                # 获取上下文
                context = None
                if context_fn:
                    context = context_fn(case)
                elif case.context:
                    context = case.context
                
                # 运行模型
                case_start = time.perf_counter()
                output = model_fn(case.input)
                latency_ms = (time.perf_counter() - case_start) * 1000
                latencies.append(latency_ms)
                
                # 计算指标
                case_metrics = {}
                
                # 准确性
                if case.expected_output:
                    acc_result = self.metrics.accuracy(output, case.expected_output, method="fuzzy")
                    case_metrics["accuracy"] = acc_result
                    accuracy_scores.append(acc_result.score)
                
                # 相关性
                rel_result = self.metrics.relevance(case.input, output)
                case_metrics["relevance"] = rel_result
                relevance_scores.append(rel_result.score)
                
                # 忠实度
                if context:
                    faith_result = self.metrics.faithfulness(output, context)
                    case_metrics["faithfulness"] = faith_result
                    faithfulness_scores.append(faith_result.score)
                
                # 无害性
                harm_result = self.metrics.harmlessness(output)
                case_metrics["harmlessness"] = harm_result
                harmlessness_scores.append(harm_result.score)
                
                # 判断是否通过
                avg_score = sum(m.score for m in case_metrics.values()) / len(case_metrics)
                case_passed = avg_score >= self.pass_threshold
                
                if case_passed:
                    passed += 1
                else:
                    failed += 1
                
                results.append(TestResult(
                    case_id=case.id,
                    input=case.input,
                    output=output,
                    expected_output=case.expected_output,
                    metrics=case_metrics,
                    latency_ms=latency_ms,
                    passed=case_passed,
                ))
                
            except Exception as e:
                errors += 1
                results.append(TestResult(
                    case_id=case.id,
                    input=case.input,
                    output="",
                    expected_output=case.expected_output,
                    metrics={},
                    latency_ms=0,
                    passed=False,
                    error=str(e),
                ))
                logger.error("evaluation_error", case_id=case.id, error=str(e))
        
        total_time = time.perf_counter() - start_time
        
        return EvaluationResults(
            dataset_name=dataset.name,
            total_cases=len(dataset),
            passed_cases=passed,
            failed_cases=failed,
            error_cases=errors,
            accuracy=sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0,
            relevance=sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
            faithfulness=sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
            harmlessness=sum(harmlessness_scores) / len(harmlessness_scores) if harmlessness_scores else 0.0,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            total_time_seconds=total_time,
            results=results,
        )

    def run_single(
        self,
        model_fn: Callable[[str], str],
        case: TestCase,
    ) -> TestResult:
        """运行单个测试用例"""
        results = self.run(
            model_fn,
            EvaluationDataset("single", "").tap(lambda d: d.add(case)),
        )
        return results.results[0] if results.results else None


class RAGEvaluationRunner(EvaluationRunner):
    """
    RAG 评测运行器
    
    专门用于 RAG 系统的评测，参考 Ragas 框架的评测指标
    """

    def run_rag(
        self,
        rag_fn: Callable[[str], tuple[str, List[str]]],
        dataset: EvaluationDataset,
    ) -> EvaluationResults:
        """
        运行 RAG 评测
        
        Args:
            rag_fn: RAG 函数 (query) -> (answer, retrieved_contexts)
            dataset: 评测数据集
        
        Returns:
            EvaluationResults
        """
        def model_fn(input_text):
            answer, _ = rag_fn(input_text)
            return answer
        
        def context_fn(case):
            _, contexts = rag_fn(case.input)
            return "\n\n".join(contexts)
        
        return self.run(model_fn, dataset, context_fn)

    def calculate_context_relevance(
        self,
        question: str,
        contexts: List[str],
    ) -> float:
        """计算上下文相关性"""
        if not contexts:
            return 0.0
        
        relevance_scores = []
        for ctx in contexts:
            result = self.metrics.relevance(question, ctx)
            relevance_scores.append(result.score)
        
        return sum(relevance_scores) / len(relevance_scores)

    def calculate_context_recall(
        self,
        expected_contexts: List[str],
        retrieved_contexts: List[str],
    ) -> float:
        """计算上下文召回率"""
        if not expected_contexts:
            return 1.0
        
        retrieved_set = set(c.lower() for c in retrieved_contexts)
        recalled = sum(
            1 for ec in expected_contexts
            if any(ec.lower() in rc for rc in retrieved_set)
        )
        
        return recalled / len(expected_contexts)


