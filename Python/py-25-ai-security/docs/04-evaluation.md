# 评测体系

## 概述

系统化评测 AI 输出质量，确保服务可靠性。

## 1. 评测指标

### 1.1 准确性（Accuracy）

答案是否正确：

```python
from ai_safety.evaluation import Metrics

metrics = Metrics()

result = metrics.accuracy(
    prediction="Python is a programming language",
    reference="Python is a high-level programming language",
    method="fuzzy",  # exact, contains, fuzzy
)

print(f"准确性: {result.score:.2%}")
```

### 1.2 相关性（Relevance）

答案是否相关：

```python
result = metrics.relevance(
    question="What is Python?",
    answer="Python is a programming language...",
)

print(f"相关性: {result.score:.2%}")
```

### 1.3 忠实度（Faithfulness）

答案是否基于上下文：

```python
result = metrics.faithfulness(
    answer="Python was created by Guido...",
    context="Python is a programming language created by Guido van Rossum...",
)

print(f"忠实度: {result.score:.2%}")
```

### 1.4 无害性（Harmlessness）

内容是否安全：

```python
result = metrics.harmlessness(text)
print(f"无害性: {result.score:.2%}")
```

## 2. 评测数据集

### 2.1 创建数据集

```python
from ai_safety.evaluation import EvaluationDataset, TestCase

dataset = EvaluationDataset("qa_test", "QA 测试数据集")

dataset.add(TestCase(
    id="test_1",
    input="What is Python?",
    expected_output="programming language",
    metadata={"category": "basic"},
))

dataset.add(TestCase(
    id="test_2",
    input="How to create a list in Python?",
    expected_output="[]",
    context="Python lists are created using square brackets...",
))
```

### 2.2 保存/加载

```python
# 保存
dataset.save("./test_data.json")

# 加载
dataset = EvaluationDataset.load("./test_data.json")
```

### 2.3 过滤和采样

```python
# 过滤
basic_cases = dataset.filter(category="basic")

# 随机采样
sample = dataset.sample(10)
```

## 3. 评测运行器

### 3.1 运行评测

```python
from ai_safety.evaluation import EvaluationRunner

runner = EvaluationRunner(pass_threshold=0.6)

# 定义模型函数
def model_fn(input_text):
    return llm.chat([{"role": "user", "content": input_text}]).content

# 运行评测
results = runner.run(model_fn, dataset)

print(results.summary())
```

### 3.2 结果分析

```python
# 汇总
print(f"通过率: {results.passed_cases}/{results.total_cases}")
print(f"准确性: {results.accuracy:.2%}")
print(f"相关性: {results.relevance:.2%}")

# 详细结果
for r in results.results:
    if not r.passed:
        print(f"失败: {r.case_id}")
        print(f"  输入: {r.input}")
        print(f"  输出: {r.output}")
```

## 4. LLM-as-Judge

### 4.1 概念

使用 LLM 作为评判者评估输出质量。

### 4.2 使用

```python
from ai_safety.evaluation.metrics import LLMAsJudge

judge = LLMAsJudge(llm_client)

evaluation = judge.evaluate(
    question="What is Python?",
    answer="Python is a programming language...",
    criteria=["relevance", "accuracy", "completeness"],
)

print(f"分数: {evaluation['score']}/5")
print(f"原因: {evaluation['reason']}")
```

### 4.3 评估提示

```python
EVALUATION_PROMPT = """
You are an expert evaluator. Rate the following answer on a scale of 1-5.

Question: {question}
Answer: {answer}

Criteria: {criteria}

Provide your rating in the format:
Score: X
Reason: Your explanation
"""
```

## 5. RAG 评测

### 5.1 Ragas 指标

- **Context Relevance**: 检索的上下文是否相关
- **Context Recall**: 是否检索到所有相关内容
- **Faithfulness**: 答案是否基于检索内容
- **Answer Relevance**: 答案是否回答了问题

### 5.2 RAG 评测

```python
from ai_safety.evaluation.runner import RAGEvaluationRunner

runner = RAGEvaluationRunner()

def rag_fn(query):
    # 返回 (answer, [contexts])
    contexts = retriever.search(query)
    answer = generate(query, contexts)
    return answer, contexts

results = runner.run_rag(rag_fn, dataset)
```

## 6. 持续评测

### 6.1 A/B 测试

```python
# 随机分流
import random

def get_model(user_id: str) -> str:
    if hash(user_id) % 100 < 10:
        return "model_b"  # 10% 流量
    return "model_a"
```

### 6.2 线上采样评测

```python
# 采样评测
if random.random() < 0.01:  # 1% 采样
    evaluate_and_log(input, output)
```

## 7. 最佳实践

1. **多维度评测**：不只看准确性
2. **定期评测**：监控质量变化
3. **真实数据**：使用生产数据
4. **人工抽检**：补充自动评测
5. **对比基准**：建立质量基线


