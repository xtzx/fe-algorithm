# 📈 RAG 评估

> 使用 Ragas 评估 RAG 系统质量

---

## RAG 评估指标

### 核心指标

```
1. Faithfulness（忠实度）
   - 答案是否忠实于检索到的上下文
   - 是否存在幻觉

2. Answer Relevance（答案相关性）
   - 答案是否相关于问题
   - 是否回答了用户的实际问题

3. Context Precision（上下文精确度）
   - 检索的上下文是否精确
   - 是否包含无关信息

4. Context Recall（上下文召回率）
   - 检索是否覆盖了回答问题所需的信息
   - 是否遗漏重要内容

5. Answer Correctness（答案正确性）
   - 答案是否正确
   - 与标准答案的一致性
```

---

## Ragas 框架

### 安装

```bash
pip install ragas datasets
```

### 基础使用

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from datasets import Dataset

# 准备评估数据
eval_data = {
    "question": [
        "什么是机器学习？",
        "Python 有什么特点？"
    ],
    "answer": [
        "机器学习是人工智能的一个分支，通过数据训练模型来做预测。",
        "Python 是一种简洁易读的编程语言，广泛用于数据科学和 AI。"
    ],
    "contexts": [
        ["机器学习是AI的子领域，专注于让计算机从数据中学习。"],
        ["Python 以简洁的语法著称，是数据科学首选语言。"]
    ],
    "ground_truth": [
        "机器学习是人工智能的一个分支，使计算机能够从数据中学习并改进。",
        "Python 是一种高级编程语言，以简洁和易读性著称。"
    ]
}

dataset = Dataset.from_dict(eval_data)

# 运行评估
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness
    ]
)

print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.88, ...}
```

### 评估单个样本

```python
from ragas.metrics import faithfulness, answer_relevancy

# 单样本评估
sample = {
    "question": "什么是深度学习？",
    "answer": "深度学习是机器学习的子集，使用多层神经网络。",
    "contexts": ["深度学习使用多层神经网络处理复杂任务。"]
}

# 计算忠实度
faith_score = faithfulness.score(sample)
print(f"Faithfulness: {faith_score}")

# 计算答案相关性
relevancy_score = answer_relevancy.score(sample)
print(f"Answer Relevancy: {relevancy_score}")
```

---

## 完整评估流程

### 评估脚本

```python
"""rag_evaluation.py - RAG 系统评估"""
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
import json
from typing import List, Dict
from datetime import datetime

class RAGEvaluator:
    """RAG 评估器"""

    def __init__(self, rag_system):
        self.rag = rag_system
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]

    def prepare_dataset(self, test_cases: List[Dict]) -> Dataset:
        """准备评估数据集"""
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for case in test_cases:
            # 调用 RAG 系统
            result = self.rag.query(case["question"])

            questions.append(case["question"])
            answers.append(result["answer"])
            contexts.append([s["content"] for s in result["sources"]])
            ground_truths.append(case.get("ground_truth", ""))

        return Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })

    def evaluate(self, test_cases: List[Dict]) -> Dict:
        """运行评估"""
        print(f"评估 {len(test_cases)} 个测试用例...")

        dataset = self.prepare_dataset(test_cases)
        results = evaluate(dataset, metrics=self.metrics)

        return {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(test_cases),
            "scores": dict(results),
            "per_sample": results.to_pandas().to_dict("records")
        }

    def save_results(self, results: Dict, path: str):
        """保存评估结果"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


# 使用示例
if __name__ == "__main__":
    from rag_engine import RAGEngine

    # 测试用例
    test_cases = [
        {
            "question": "什么是 RAG？",
            "ground_truth": "RAG 是检索增强生成，通过检索外部知识来增强 LLM 的回答。"
        },
        {
            "question": "LangChain 是什么？",
            "ground_truth": "LangChain 是一个用于构建 LLM 应用的开发框架。"
        },
        # ... 更多测试用例
    ]

    # 评估
    rag = RAGEngine()
    evaluator = RAGEvaluator(rag)
    results = evaluator.evaluate(test_cases)

    # 输出结果
    print("\n评估结果：")
    for metric, score in results["scores"].items():
        print(f"  {metric}: {score:.4f}")

    # 保存
    evaluator.save_results(results, "eval_results.json")
```

### 批量评估

```python
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def batch_evaluate(rag, test_cases: List[Dict], batch_size: int = 10):
    """批量评估"""
    results = []

    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i:i+batch_size]

        with ThreadPoolExecutor(max_workers=5) as executor:
            batch_results = list(executor.map(
                lambda case: {
                    "question": case["question"],
                    **rag.query(case["question"])
                },
                batch
            ))

        results.extend(batch_results)
        print(f"进度: {min(i+batch_size, len(test_cases))}/{len(test_cases)}")

    return results
```

---

## 自定义评估指标

### 自定义指标

```python
from ragas.metrics.base import MetricWithLLM
from dataclasses import dataclass

@dataclass
class CustomMetric(MetricWithLLM):
    name: str = "custom_metric"

    def score(self, sample: Dict) -> float:
        """自定义评分逻辑"""
        question = sample["question"]
        answer = sample["answer"]

        # 示例：检查答案长度
        if len(answer) < 10:
            return 0.0
        elif len(answer) > 500:
            return 0.5
        else:
            return 1.0

# 简单的评估函数
def evaluate_answer_length(answer: str) -> float:
    """评估答案长度是否合适"""
    if len(answer) < 20:
        return 0.3
    elif len(answer) < 50:
        return 0.7
    elif len(answer) < 500:
        return 1.0
    else:
        return 0.8

def evaluate_has_source(answer: str, sources: list) -> float:
    """评估是否引用了来源"""
    return 1.0 if sources else 0.0
```

### 基于 LLM 的评估

```python
from openai import OpenAI

client = OpenAI()

def llm_evaluate(question: str, answer: str, criteria: str) -> float:
    """使用 LLM 进行评估"""

    prompt = f"""请评估以下回答的质量。

问题：{question}
回答：{answer}

评估标准：{criteria}

请给出 1-10 的分数，只输出数字。
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        score = float(response.choices[0].message.content.strip())
        return score / 10
    except:
        return 0.5

# 使用
score = llm_evaluate(
    question="什么是机器学习？",
    answer="机器学习是 AI 的一个分支...",
    criteria="回答是否准确、完整、易懂"
)
```

---

## 评估报告

```python
import pandas as pd
import matplotlib.pyplot as plt

def generate_report(results: Dict, output_path: str = "eval_report.html"):
    """生成评估报告"""

    # 总体分数
    overall_scores = results["scores"]

    # 每个样本的分数
    per_sample = pd.DataFrame(results["per_sample"])

    # 生成 HTML 报告
    html = f"""
    <html>
    <head>
        <title>RAG 评估报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            .metric {{
                display: inline-block;
                margin: 10px;
                padding: 20px;
                background: #f5f5f5;
                border-radius: 8px;
            }}
            .score {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>RAG 评估报告</h1>
        <p>时间: {results['timestamp']}</p>
        <p>样本数: {results['num_samples']}</p>

        <h2>总体分数</h2>
        <div>
    """

    for metric, score in overall_scores.items():
        html += f"""
            <div class="metric">
                <div>{metric}</div>
                <div class="score">{score:.2%}</div>
            </div>
        """

    html += """
        </div>

        <h2>详细结果</h2>
        <table>
            <tr>
                <th>问题</th>
                <th>Faithfulness</th>
                <th>Relevancy</th>
                <th>Precision</th>
                <th>Recall</th>
            </tr>
    """

    for _, row in per_sample.iterrows():
        html += f"""
            <tr>
                <td>{row.get('question', '')[:50]}...</td>
                <td>{row.get('faithfulness', 0):.2f}</td>
                <td>{row.get('answer_relevancy', 0):.2f}</td>
                <td>{row.get('context_precision', 0):.2f}</td>
                <td>{row.get('context_recall', 0):.2f}</td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"报告已保存到 {output_path}")
```

---

## 持续评估

```python
"""定期评估 RAG 系统"""
import schedule
import time

def daily_evaluation():
    """每日评估任务"""
    # 加载测试集
    test_cases = load_test_cases("test_cases.json")

    # 运行评估
    evaluator = RAGEvaluator(rag_system)
    results = evaluator.evaluate(test_cases)

    # 保存结果
    date_str = datetime.now().strftime("%Y%m%d")
    evaluator.save_results(results, f"eval_{date_str}.json")

    # 检查质量下降
    for metric, score in results["scores"].items():
        if score < 0.7:
            send_alert(f"警告: {metric} 分数过低: {score:.2f}")

# 每天凌晨 2 点运行
schedule.every().day.at("02:00").do(daily_evaluation)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## ➡️ 下一步

继续 [11-安全防护.md](./11-安全防护.md)

