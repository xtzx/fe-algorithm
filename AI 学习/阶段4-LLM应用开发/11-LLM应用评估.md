# 📊 11 - LLM 应用评估

> 如何评估和监控 LLM 应用的质量、性能和成本

---

## 目录

1. [为什么需要评估](#1-为什么需要评估)
2. [评估指标体系](#2-评估指标体系)
3. [自动化评估方法](#3-自动化评估方法)
4. [人工评估](#4-人工评估)
5. [RAG 专项评估](#5-rag-专项评估)
6. [生产监控](#6-生产监控)
7. [A/B 测试](#7-ab-测试)
8. [练习题](#8-练习题)

---

## 1. 为什么需要评估

### 1.1 评估的价值

```
为什么要评估 LLM 应用？

1. 质量保障
   └── 确保输出符合预期

2. 迭代改进
   └── 量化改进效果

3. 问题诊断
   └── 发现薄弱环节

4. 成本优化
   └── 平衡质量与成本

5. 合规要求
   └── 满足业务/法规标准
```

### 1.2 评估的挑战

```
LLM 评估为什么难？

1. 开放性输出
   └── 同一问题有多个正确答案

2. 主观性强
   └── 质量判断因人而异

3. 长尾问题
   └── 边缘场景难以覆盖

4. 动态性
   └── 模型升级可能导致行为变化
```

---

## 2. 评估指标体系

### 2.1 通用指标

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class QualityDimension(Enum):
    RELEVANCE = "相关性"      # 回答是否切题
    ACCURACY = "准确性"       # 事实是否正确
    COHERENCE = "连贯性"      # 逻辑是否通顺
    FLUENCY = "流畅性"        # 语言是否自然
    HELPFULNESS = "有用性"    # 是否解决用户问题
    SAFETY = "安全性"         # 是否包含有害内容

@dataclass
class EvaluationResult:
    """评估结果"""
    question: str
    answer: str
    reference: Optional[str] = None

    # 各维度得分 (0-1)
    relevance: float = 0.0
    accuracy: float = 0.0
    coherence: float = 0.0
    fluency: float = 0.0
    helpfulness: float = 0.0
    safety: float = 0.0

    # 综合得分
    overall: float = 0.0

    # 元信息
    latency_ms: float = 0.0
    token_count: int = 0
    cost_usd: float = 0.0
```

### 2.2 任务特定指标

```python
# 1. 分类任务
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_classification(predictions, labels):
    return {
        "precision": precision_score(labels, predictions, average='weighted'),
        "recall": recall_score(labels, predictions, average='weighted'),
        "f1": f1_score(labels, predictions, average='weighted'),
    }

# 2. 生成任务
from rouge_score import rouge_scorer

def evaluate_generation(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure,
    }

# 3. QA 任务
def evaluate_qa(predicted, expected):
    """精确匹配和 F1"""
    # 精确匹配
    exact_match = predicted.strip().lower() == expected.strip().lower()

    # Token-level F1
    pred_tokens = set(predicted.lower().split())
    exp_tokens = set(expected.lower().split())

    if len(pred_tokens) == 0 or len(exp_tokens) == 0:
        f1 = 0.0
    else:
        precision = len(pred_tokens & exp_tokens) / len(pred_tokens)
        recall = len(pred_tokens & exp_tokens) / len(exp_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"exact_match": exact_match, "f1": f1}
```

---

## 3. 自动化评估方法

### 3.1 LLM-as-Judge

```python
from openai import OpenAI
import json

client = OpenAI()

def llm_judge(question: str, answer: str, criteria: str = None) -> dict:
    """使用 LLM 作为评判者"""

    if criteria is None:
        criteria = """
1. 相关性 (relevance): 回答是否切题
2. 准确性 (accuracy): 信息是否正确
3. 完整性 (completeness): 是否充分回答问题
4. 清晰度 (clarity): 表达是否清楚
"""

    prompt = f"""
你是一个专业的回答质量评估员。请评估以下问答的质量。

问题：{question}

回答：{answer}

评估维度：
{criteria}

请为每个维度打分（1-5分），并给出简短理由。
返回 JSON 格式：
{{
    "relevance": {{"score": 1-5, "reason": "..."}},
    "accuracy": {{"score": 1-5, "reason": "..."}},
    "completeness": {{"score": 1-5, "reason": "..."}},
    "clarity": {{"score": 1-5, "reason": "..."}},
    "overall": 1-5,
    "feedback": "总体评价..."
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# 使用
result = llm_judge(
    question="什么是机器学习？",
    answer="机器学习是人工智能的一个分支，让计算机能从数据中学习规律，而无需显式编程。"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
```

### 3.2 成对比较 (Pairwise Comparison)

```python
def pairwise_compare(question: str, answer_a: str, answer_b: str) -> dict:
    """成对比较两个回答"""

    prompt = f"""
请比较以下两个回答的质量。

问题：{question}

回答 A：
{answer_a}

回答 B：
{answer_b}

请判断哪个回答更好，返回 JSON：
{{
    "winner": "A" 或 "B" 或 "tie",
    "reason": "选择理由...",
    "a_strengths": ["A 的优点..."],
    "a_weaknesses": ["A 的缺点..."],
    "b_strengths": ["B 的优点..."],
    "b_weaknesses": ["B 的缺点..."]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# 使用：比较不同 prompt 策略的效果
result = pairwise_compare(
    question="如何学习 Python？",
    answer_a="首先安装 Python，然后学习基础语法。",
    answer_b="学习 Python 建议分几步：1) 安装环境 2) 学习基础语法 3) 做小项目 4) 深入框架。推荐从官方教程开始。"
)
print(f"胜者: {result['winner']}, 原因: {result['reason']}")
```

### 3.3 基于参考答案的评估

```python
def evaluate_with_reference(question: str, answer: str, reference: str) -> dict:
    """与参考答案对比评估"""

    prompt = f"""
请评估回答与参考答案的一致性和质量。

问题：{question}

参考答案：{reference}

待评估回答：{answer}

评估要点：
1. 事实一致性：回答中的事实是否与参考答案一致
2. 信息覆盖度：参考答案中的要点覆盖了多少
3. 额外信息：是否有超出参考答案的正确/错误信息

返回 JSON：
{{
    "factual_consistency": 0-1,
    "coverage": 0-1,
    "extra_correct": ["正确的额外信息..."],
    "extra_incorrect": ["错误的额外信息..."],
    "missing": ["缺失的要点..."],
    "overall_score": 0-1
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

---

## 4. 人工评估

### 4.1 评估界面设计

```python
import streamlit as st
from datetime import datetime
import json

def create_evaluation_ui():
    """创建人工评估界面"""

    st.title("LLM 回答质量评估")

    # 加载待评估数据
    if 'eval_data' not in st.session_state:
        st.session_state.eval_data = load_eval_samples()
        st.session_state.current_idx = 0
        st.session_state.results = []

    data = st.session_state.eval_data
    idx = st.session_state.current_idx

    if idx >= len(data):
        st.success("评估完成！")
        st.json(st.session_state.results)
        return

    sample = data[idx]

    # 显示问答
    st.subheader(f"问题 {idx + 1}/{len(data)}")
    st.write(f"**问题：** {sample['question']}")
    st.write(f"**回答：** {sample['answer']}")

    # 评分
    st.subheader("质量评分")

    relevance = st.slider("相关性", 1, 5, 3, help="回答是否切题")
    accuracy = st.slider("准确性", 1, 5, 3, help="信息是否正确")
    helpfulness = st.slider("有用性", 1, 5, 3, help="是否解决用户问题")

    # 文字反馈
    feedback = st.text_area("补充反馈（可选）")

    # 提交
    if st.button("提交并下一个"):
        result = {
            "sample_id": sample.get("id", idx),
            "question": sample["question"],
            "answer": sample["answer"],
            "relevance": relevance,
            "accuracy": accuracy,
            "helpfulness": helpfulness,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
            "evaluator": "human"
        }

        st.session_state.results.append(result)
        st.session_state.current_idx += 1
        st.rerun()

def load_eval_samples():
    """加载评估样本"""
    # 实际应用中从数据库或文件加载
    return [
        {"id": 1, "question": "什么是 Python？", "answer": "Python 是一种编程语言..."},
        {"id": 2, "question": "如何学习机器学习？", "answer": "首先学习数学基础..."},
    ]
```

### 4.2 评估者一致性

```python
from scipy.stats import pearsonr, spearmanr
import numpy as np

def calculate_inter_rater_agreement(ratings_a: list, ratings_b: list) -> dict:
    """计算评估者间一致性"""

    # Pearson 相关系数
    pearson_r, pearson_p = pearsonr(ratings_a, ratings_b)

    # Spearman 秩相关
    spearman_r, spearman_p = spearmanr(ratings_a, ratings_b)

    # 精确一致率
    exact_agreement = sum(a == b for a, b in zip(ratings_a, ratings_b)) / len(ratings_a)

    # 允许1分差异的一致率
    near_agreement = sum(abs(a - b) <= 1 for a, b in zip(ratings_a, ratings_b)) / len(ratings_a)

    return {
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "exact_agreement": exact_agreement,
        "near_agreement": near_agreement,
    }

# 使用
evaluator_1 = [4, 3, 5, 2, 4, 3, 5, 4]
evaluator_2 = [5, 3, 4, 2, 4, 4, 5, 3]

agreement = calculate_inter_rater_agreement(evaluator_1, evaluator_2)
print(f"Pearson 相关: {agreement['pearson_r']:.3f}")
print(f"精确一致率: {agreement['exact_agreement']:.2%}")
```

---

## 5. RAG 专项评估

### 5.1 RAG 评估维度

```python
@dataclass
class RAGEvaluationResult:
    """RAG 系统评估结果"""

    # 检索质量
    retrieval_precision: float    # 检索结果的相关性
    retrieval_recall: float       # 是否召回所有相关文档
    retrieval_mrr: float          # 平均倒数排名

    # 生成质量
    answer_relevance: float       # 回答相关性
    answer_faithfulness: float    # 回答忠实度（是否基于检索内容）
    answer_correctness: float     # 回答正确性

    # 整体质量
    context_utilization: float    # 上下文利用率
    hallucination_rate: float     # 幻觉率
```

### 5.2 忠实度评估

```python
def evaluate_faithfulness(answer: str, contexts: list[str]) -> dict:
    """评估回答对检索内容的忠实度"""

    prompt = f"""
请评估回答是否忠实于给定的上下文。

上下文：
{chr(10).join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])}

回答：
{answer}

分析：
1. 回答中的每个陈述是否都能在上下文中找到依据？
2. 是否有凭空编造的内容（幻觉）？

返回 JSON：
{{
    "faithfulness_score": 0-1,
    "supported_claims": ["有依据的陈述..."],
    "unsupported_claims": ["无依据的陈述..."],
    "hallucinations": ["幻觉内容..."],
    "analysis": "详细分析..."
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)
```

### 5.3 检索评估

```python
from typing import List

def evaluate_retrieval(
    query: str,
    retrieved_docs: List[str],
    relevant_docs: List[str]  # 真实相关文档
) -> dict:
    """评估检索质量"""

    # 计算 Precision@K
    k = len(retrieved_docs)
    relevant_retrieved = set(retrieved_docs) & set(relevant_docs)
    precision_at_k = len(relevant_retrieved) / k if k > 0 else 0

    # 计算 Recall
    recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0

    # 计算 MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            mrr = 1.0 / (i + 1)
            break

    # 计算 NDCG
    def dcg(relevances, k):
        return sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))

    relevances = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
    ideal_relevances = sorted(relevances, reverse=True)

    dcg_score = dcg(relevances, k)
    idcg_score = dcg(ideal_relevances, k)
    ndcg = dcg_score / idcg_score if idcg_score > 0 else 0

    return {
        "precision_at_k": precision_at_k,
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg,
        "relevant_count": len(relevant_retrieved),
        "total_retrieved": k,
        "total_relevant": len(relevant_docs)
    }
```

---

## 6. 生产监控

### 6.1 核心监控指标

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
import time

@dataclass
class RequestMetrics:
    """单次请求指标"""
    request_id: str
    timestamp: datetime

    # 性能
    latency_ms: float
    time_to_first_token_ms: float = 0.0

    # 成本
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    # 质量
    success: bool = True
    error_type: str = None

    # 业务
    user_id: str = None
    session_id: str = None
    feedback: int = None  # 1=positive, -1=negative, 0=neutral

class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics: List[RequestMetrics] = []

    def record(self, metrics: RequestMetrics):
        self.metrics.append(metrics)

    def get_summary(self, window_minutes: int = 60) -> dict:
        """获取指标摘要"""
        cutoff = datetime.now().timestamp() - window_minutes * 60
        recent = [m for m in self.metrics if m.timestamp.timestamp() > cutoff]

        if not recent:
            return {"message": "No data in window"}

        latencies = [m.latency_ms for m in recent]
        costs = [m.cost_usd for m in recent]
        success_rate = sum(1 for m in recent if m.success) / len(recent)

        return {
            "request_count": len(recent),
            "success_rate": success_rate,
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            "total_cost_usd": sum(costs),
            "avg_cost_per_request": np.mean(costs),
        }

# 使用装饰器自动收集指标
collector = MetricsCollector()

def track_request(func):
    def wrapper(*args, **kwargs):
        request_id = str(time.time())
        start = time.time()

        try:
            result = func(*args, **kwargs)
            success = True
            error_type = None
        except Exception as e:
            success = False
            error_type = type(e).__name__
            raise
        finally:
            latency = (time.time() - start) * 1000

            metrics = RequestMetrics(
                request_id=request_id,
                timestamp=datetime.now(),
                latency_ms=latency,
                success=success,
                error_type=error_type
            )
            collector.record(metrics)

        return result
    return wrapper
```

### 6.2 告警规则

```python
from typing import Callable
from dataclasses import dataclass

@dataclass
class AlertRule:
    name: str
    condition: Callable[[dict], bool]
    severity: str  # "critical", "warning", "info"
    message_template: str

class AlertManager:
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.triggered_alerts: List[dict] = []

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def check(self, metrics_summary: dict):
        for rule in self.rules:
            if rule.condition(metrics_summary):
                alert = {
                    "rule": rule.name,
                    "severity": rule.severity,
                    "message": rule.message_template.format(**metrics_summary),
                    "timestamp": datetime.now().isoformat()
                }
                self.triggered_alerts.append(alert)
                self._notify(alert)

    def _notify(self, alert: dict):
        # 发送通知（邮件、Slack、PagerDuty 等）
        print(f"[{alert['severity'].upper()}] {alert['message']}")

# 配置告警规则
alert_manager = AlertManager()

alert_manager.add_rule(AlertRule(
    name="high_error_rate",
    condition=lambda m: m.get("success_rate", 1) < 0.95,
    severity="critical",
    message_template="错误率过高：{success_rate:.1%}"
))

alert_manager.add_rule(AlertRule(
    name="high_latency",
    condition=lambda m: m.get("latency_p95", 0) > 5000,
    severity="warning",
    message_template="P95 延迟过高：{latency_p95:.0f}ms"
))

alert_manager.add_rule(AlertRule(
    name="cost_spike",
    condition=lambda m: m.get("avg_cost_per_request", 0) > 0.1,
    severity="warning",
    message_template="单次请求成本过高：${avg_cost_per_request:.3f}"
))
```

### 6.3 用户反馈收集

```python
from enum import IntEnum

class FeedbackType(IntEnum):
    THUMBS_UP = 1
    THUMBS_DOWN = -1
    NEUTRAL = 0

class FeedbackCollector:
    def __init__(self):
        self.feedbacks = []

    def record(self, request_id: str, feedback: FeedbackType,
               comment: str = None, categories: List[str] = None):
        """记录用户反馈"""
        self.feedbacks.append({
            "request_id": request_id,
            "feedback": feedback.value,
            "comment": comment,
            "categories": categories or [],
            "timestamp": datetime.now().isoformat()
        })

    def get_satisfaction_rate(self, window_hours: int = 24) -> dict:
        """计算满意度"""
        cutoff = datetime.now().timestamp() - window_hours * 3600
        recent = [f for f in self.feedbacks
                  if datetime.fromisoformat(f["timestamp"]).timestamp() > cutoff]

        if not recent:
            return {"message": "No feedback in window"}

        positive = sum(1 for f in recent if f["feedback"] == 1)
        negative = sum(1 for f in recent if f["feedback"] == -1)

        return {
            "total_feedback": len(recent),
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": positive / len(recent),
            "nps": (positive - negative) / len(recent) * 100  # Net Promoter Score
        }
```

---

## 7. A/B 测试

### 7.1 实验框架

```python
import random
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class Experiment:
    name: str
    variants: Dict[str, Any]  # variant_name -> config
    traffic_split: Dict[str, float]  # variant_name -> 流量比例

class ABTestManager:
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.assignments: Dict[str, str] = {}  # user_id -> variant

    def create_experiment(self, experiment: Experiment):
        # 验证流量分配总和为 1
        assert abs(sum(experiment.traffic_split.values()) - 1.0) < 0.001
        self.experiments[experiment.name] = experiment

    def get_variant(self, experiment_name: str, user_id: str) -> tuple[str, Any]:
        """获取用户分配的变体"""
        exp = self.experiments[experiment_name]

        # 检查是否已分配
        key = f"{experiment_name}:{user_id}"
        if key in self.assignments:
            variant_name = self.assignments[key]
            return variant_name, exp.variants[variant_name]

        # 基于用户 ID 的确定性分配（保证同一用户始终看到同一变体）
        hash_value = hash(key) % 10000 / 10000

        cumulative = 0.0
        for variant_name, proportion in exp.traffic_split.items():
            cumulative += proportion
            if hash_value < cumulative:
                self.assignments[key] = variant_name
                return variant_name, exp.variants[variant_name]

        # 默认返回第一个
        first_variant = list(exp.variants.keys())[0]
        self.assignments[key] = first_variant
        return first_variant, exp.variants[first_variant]

# 使用
ab_manager = ABTestManager()

# 创建实验：测试不同的 prompt
ab_manager.create_experiment(Experiment(
    name="prompt_optimization",
    variants={
        "control": {"prompt": "请回答以下问题："},
        "treatment_a": {"prompt": "你是专家，请详细回答："},
        "treatment_b": {"prompt": "请用简洁的语言回答："}
    },
    traffic_split={
        "control": 0.34,
        "treatment_a": 0.33,
        "treatment_b": 0.33
    }
))

# 获取用户分配
user_id = "user_123"
variant_name, config = ab_manager.get_variant("prompt_optimization", user_id)
print(f"用户 {user_id} 分配到：{variant_name}")
print(f"使用 prompt：{config['prompt']}")
```

### 7.2 统计显著性检验

```python
from scipy import stats
import numpy as np

def calculate_significance(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int
) -> dict:
    """计算 A/B 测试的统计显著性"""

    # 转化率
    control_rate = control_conversions / control_total
    treatment_rate = treatment_conversions / treatment_total

    # 相对提升
    relative_lift = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0

    # Z 检验
    pooled_rate = (control_conversions + treatment_conversions) / (control_total + treatment_total)
    se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_total + 1/treatment_total))
    z_score = (treatment_rate - control_rate) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # 双尾检验

    # 置信区间
    ci_95 = 1.96 * se

    return {
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "relative_lift": relative_lift,
        "z_score": z_score,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "confidence_interval": (treatment_rate - control_rate - ci_95,
                                 treatment_rate - control_rate + ci_95)
    }

# 使用
result = calculate_significance(
    control_conversions=120,
    control_total=1000,
    treatment_conversions=150,
    treatment_total=1000
)

print(f"Control 转化率: {result['control_rate']:.2%}")
print(f"Treatment 转化率: {result['treatment_rate']:.2%}")
print(f"相对提升: {result['relative_lift']:.2%}")
print(f"P-value: {result['p_value']:.4f}")
print(f"统计显著: {result['is_significant']}")
```

---

## 8. 练习题

### 基础练习

1. 实现一个 LLM-as-Judge 评估函数
2. 计算 ROUGE 分数评估生成质量
3. 实现用户反馈收集和满意度计算

### 进阶练习

4. 设计一个完整的 RAG 评估 Pipeline
5. 实现 A/B 测试框架和统计显著性检验

### 参考答案

<details>
<summary>练习 1 参考答案</summary>

```python
from openai import OpenAI
import json

client = OpenAI()

def llm_judge_simple(question: str, answer: str) -> dict:
    """简单的 LLM 评估"""

    prompt = f"""
评估以下问答的质量，返回 JSON 格式评分。

问题：{question}
回答：{answer}

返回格式：
{{"score": 1-5, "reason": "评分理由"}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# 测试
result = llm_judge_simple(
    question="Python 和 Java 的区别是什么？",
    answer="Python 是动态类型语言，语法简洁；Java 是静态类型语言，需要编译。"
)
print(f"评分: {result['score']}, 理由: {result['reason']}")
```

</details>

---

## ➡️ 下一步

学完本节后，继续实战项目 [12-项目-RAG知识库.md](./12-项目-RAG知识库.md)

