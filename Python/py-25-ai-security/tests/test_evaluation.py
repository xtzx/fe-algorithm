"""
评测模块测试
"""

import pytest

from ai_safety.evaluation import (
    EvaluationDataset,
    EvaluationRunner,
    Metrics,
    TestCase,
)
from ai_safety.evaluation.metrics import MetricType


class TestMetrics:
    """评测指标测试"""

    @pytest.fixture
    def metrics(self):
        return Metrics()

    def test_accuracy_exact(self, metrics):
        """测试精确匹配准确性"""
        result = metrics.accuracy("hello", "hello", method="exact")
        assert result.score == 1.0
        
        result = metrics.accuracy("hello", "world", method="exact")
        assert result.score == 0.0

    def test_accuracy_contains(self, metrics):
        """测试包含匹配准确性"""
        result = metrics.accuracy(
            "Python is a programming language",
            "programming language",
            method="contains",
        )
        assert result.score == 1.0

    def test_accuracy_fuzzy(self, metrics):
        """测试模糊匹配准确性"""
        result = metrics.accuracy(
            "Python is great",
            "Python is wonderful",
            method="fuzzy",
        )
        # "Python" 和 "is" 匹配
        assert result.score > 0

    def test_relevance(self, metrics):
        """测试相关性"""
        result = metrics.relevance(
            question="What is Python?",
            answer="Python is a programming language",
        )
        assert result.metric_type == MetricType.RELEVANCE
        assert result.score > 0

    def test_faithfulness(self, metrics):
        """测试忠实度"""
        result = metrics.faithfulness(
            answer="Python was created by Guido.",
            context="Python is a language created by Guido van Rossum.",
        )
        assert result.metric_type == MetricType.FAITHFULNESS
        assert result.score > 0

    def test_harmlessness(self, metrics):
        """测试无害性"""
        safe_result = metrics.harmlessness("Python is a programming language")
        assert safe_result.score == 1.0
        
        # 包含潜在有害词
        risky_result = metrics.harmlessness("This is violent content")
        assert risky_result.score < 1.0


class TestEvaluationDataset:
    """评测数据集测试"""

    def test_add_case(self):
        """测试添加测试用例"""
        dataset = EvaluationDataset("test")
        
        dataset.add(TestCase(
            id="test_1",
            input="What is Python?",
            expected_output="programming language",
        ))
        
        assert len(dataset) == 1

    def test_get_case(self):
        """测试获取测试用例"""
        dataset = EvaluationDataset("test")
        
        case = TestCase(id="test_1", input="question")
        dataset.add(case)
        
        retrieved = dataset.get("test_1")
        assert retrieved is not None
        assert retrieved.input == "question"

    def test_filter(self):
        """测试过滤"""
        dataset = EvaluationDataset("test")
        
        dataset.add(TestCase(id="1", input="q1", metadata={"type": "basic"}))
        dataset.add(TestCase(id="2", input="q2", metadata={"type": "advanced"}))
        
        basic = dataset.filter(type="basic")
        assert len(basic) == 1
        assert basic[0].id == "1"

    def test_sample(self):
        """测试采样"""
        dataset = EvaluationDataset("test")
        
        for i in range(10):
            dataset.add(TestCase(id=f"test_{i}", input=f"question {i}"))
        
        sample = dataset.sample(3)
        assert len(sample) == 3

    def test_save_and_load(self, tmp_path):
        """测试保存和加载"""
        dataset = EvaluationDataset("test", "Test dataset")
        dataset.add(TestCase(
            id="test_1",
            input="What is Python?",
            expected_output="language",
        ))
        
        # 保存
        save_path = tmp_path / "dataset.json"
        dataset.save(save_path)
        
        # 加载
        loaded = EvaluationDataset.load(save_path)
        
        assert loaded.name == "test"
        assert len(loaded) == 1


class TestEvaluationRunner:
    """评测运行器测试"""

    @pytest.fixture
    def dataset(self):
        dataset = EvaluationDataset("test")
        dataset.add(TestCase(
            id="test_1",
            input="What is Python?",
            expected_output="programming language",
        ))
        dataset.add(TestCase(
            id="test_2",
            input="What is 2+2?",
            expected_output="4",
        ))
        return dataset

    def test_run(self, dataset):
        """测试运行评测"""
        runner = EvaluationRunner()
        
        # 简单的模拟模型
        def model_fn(input_text):
            if "Python" in input_text:
                return "Python is a programming language"
            return "The answer is 4"
        
        results = runner.run(model_fn, dataset)
        
        assert results.total_cases == 2
        assert results.passed_cases + results.failed_cases + results.error_cases == 2

    def test_results_summary(self, dataset):
        """测试结果摘要"""
        runner = EvaluationRunner()
        
        def model_fn(input_text):
            return "response"
        
        results = runner.run(model_fn, dataset)
        summary = results.summary()
        
        assert "评测结果" in summary
        assert "准确性" in summary


