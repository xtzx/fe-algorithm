"""
评测系统演示

展示评测功能
"""

from ai_safety.evaluation import (
    EvaluationDataset,
    EvaluationRunner,
    Metrics,
    TestCase,
)


def demo_metrics():
    """评测指标演示"""
    print("=== 评测指标演示 ===\n")
    
    metrics = Metrics()
    
    # 准确性
    print("1. 准确性评测:")
    test_cases = [
        ("Python", "Python", "exact"),
        ("Python is great", "programming language", "contains"),
        ("Python is wonderful", "Python is great", "fuzzy"),
    ]
    
    for pred, ref, method in test_cases:
        result = metrics.accuracy(pred, ref, method=method)
        print(f"  预测: {pred}")
        print(f"  参考: {ref}")
        print(f"  方法: {method}")
        print(f"  分数: {result.score:.2%}")
        print()
    
    # 相关性
    print("2. 相关性评测:")
    result = metrics.relevance(
        question="What is Python?",
        answer="Python is a high-level programming language known for its simplicity.",
    )
    print(f"  问题: What is Python?")
    print(f"  答案: Python is a high-level programming language...")
    print(f"  分数: {result.score:.2%}")
    print()
    
    # 忠实度
    print("3. 忠实度评测:")
    result = metrics.faithfulness(
        answer="Python was created by Guido van Rossum in 1991.",
        context="Python is a programming language created by Guido van Rossum. It was first released in 1991.",
    )
    print(f"  答案: Python was created by Guido van Rossum in 1991.")
    print(f"  上下文: Python is a programming language created by...")
    print(f"  分数: {result.score:.2%}")
    print()


def demo_evaluation_runner():
    """评测运行器演示"""
    print("=== 评测运行器演示 ===\n")
    
    # 创建数据集
    dataset = EvaluationDataset("qa_test", "问答测试数据集")
    
    dataset.add(TestCase(
        id="q1",
        input="What is Python?",
        expected_output="programming language",
    ))
    dataset.add(TestCase(
        id="q2",
        input="What is 2+2?",
        expected_output="4",
    ))
    dataset.add(TestCase(
        id="q3",
        input="What is the capital of France?",
        expected_output="Paris",
    ))
    
    print(f"数据集: {dataset.name}")
    print(f"用例数: {len(dataset)}")
    print()
    
    # 模拟模型
    def mock_model(input_text: str) -> str:
        responses = {
            "What is Python?": "Python is a popular programming language used for web development and data science.",
            "What is 2+2?": "The answer is 4.",
            "What is the capital of France?": "The capital of France is Paris.",
        }
        return responses.get(input_text, "I don't know.")
    
    # 运行评测
    runner = EvaluationRunner(pass_threshold=0.5)
    results = runner.run(mock_model, dataset)
    
    # 输出结果
    print(results.summary())
    print()
    
    # 详细结果
    print("详细结果:")
    for r in results.results:
        status = "✓ 通过" if r.passed else "✗ 失败"
        print(f"  {r.case_id}: {status}")
        print(f"    输入: {r.input}")
        print(f"    输出: {r.output[:50]}...")
        print()


def main():
    print("=" * 50)
    print("评测系统演示")
    print("=" * 50)
    print()
    
    demo_metrics()
    demo_evaluation_runner()
    
    print("演示完成！")


if __name__ == "__main__":
    main()


