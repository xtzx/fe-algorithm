# 面试高频题

## 1. 什么是提示注入？如何防护？

### 答案

**提示注入**是攻击者通过精心构造的输入，试图操纵 LLM 行为的攻击技术。

**类型**：
- **直接注入**：用户输入中包含恶意指令
- **间接注入**：恶意指令隐藏在外部数据中
- **越狱**：试图绕过安全限制

**防护措施**：

1. **输入过滤**
```python
detector = InjectionDetector()
result = detector.detect(user_input)
if result.is_injection:
    reject_request()
```

2. **分隔符隔离**
```
<user_input>
{用户输入}
</user_input>
```

3. **角色强化**：在系统提示中明确规则

4. **输出验证**：检查是否泄露系统提示

---

## 2. 如何评估 RAG 系统的质量？

### 答案

**评测指标**：

| 指标 | 描述 |
|------|------|
| Context Relevance | 检索的上下文是否相关 |
| Context Recall | 是否检索到所有相关内容 |
| Faithfulness | 答案是否基于检索内容 |
| Answer Relevance | 答案是否回答了问题 |

**评测方法**：

```python
# 使用 Ragas 框架
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
)
```

**最佳实践**：
- 构建专门的评测数据集
- 包含真实场景的问答对
- 标注相关文档
- 定期评测并对比基准

---

## 3. 什么是 LLM-as-Judge？

### 答案

**LLM-as-Judge** 是使用 LLM 作为评判者来评估另一个 LLM 输出质量的技术。

**优势**：
- 比规则评估更灵活
- 可以评估开放式问题
- 更接近人类判断

**实现**：

```python
JUDGE_PROMPT = """
Rate the following answer on a scale of 1-5:

Question: {question}
Answer: {answer}

Criteria: accuracy, relevance, completeness

Score: 
Reason:
"""

result = llm.chat(JUDGE_PROMPT.format(question=q, answer=a))
```

**注意事项**：
- 选择能力强的模型作为 Judge
- 提供清晰的评判标准
- 人工抽检验证

---

## 4. 如何处理 LLM 的幻觉问题？

### 答案

**幻觉**指 LLM 生成看似正确但实际错误的信息。

**预防措施**：

1. **RAG**：基于检索的真实文档回答
2. **温度调低**：降低随机性
3. **提示约束**：要求基于上下文回答

**检测方法**：

```python
# 检测过度自信
overconfidence = ["definitely", "certainly", "100%"]
if any(word in response for word in overconfidence):
    flag_for_review()

# 检测不确定性
uncertainty = ["I'm not sure", "might be"]
# 正常的不确定性表达是好的
```

**处理策略**：
- 添加免责声明
- 要求引用来源
- 人工审核高风险内容

---

## 5. 如何保护用户隐私？

### 答案

**措施**：

1. **PII 过滤**
```python
filter = OutputFilter()
safe_text = filter.remove_pii(response)
```

2. **数据隔离**：用户数据不互相访问

3. **最小化数据**：不存储不必要的数据

4. **加密**：传输和存储加密

5. **审计日志**：记录数据访问

**合规要求**：
- GDPR（欧盟）
- CCPA（加州）
- 数据本地化

---

## 6. 如何监控 LLM 应用的成本？

### 答案

**监控指标**：

```python
monitor = CostMonitor(daily_budget=100.0)

# 记录每次调用
monitor.record_usage(
    model="gpt-4",
    input_tokens=100,
    output_tokens=200,
)

# 检查预算
if monitor.is_over_budget():
    switch_to_cheaper_model()
```

**优化策略**：

| 策略 | 效果 |
|------|------|
| 模型选择 | 简单任务用便宜模型 |
| 缓存 | 相同查询复用结果 |
| 截断上下文 | 减少输入 token |
| 批处理 | 合并请求 |
| 设置限额 | 用户/日限制 |

---

## 7. 如何设计 AI 应用的回退策略？

### 答案

**多层回退**：

```python
async def call_with_fallback(prompt: str):
    try:
        # 主模型
        return await call_gpt4(prompt)
    except RateLimitError:
        # 备用模型
        return await call_gpt35(prompt)
    except TimeoutError:
        # 缓存响应
        return get_cached_response(prompt)
    except Exception:
        # 预设响应
        return "服务暂时不可用"
```

**设计原则**：
- 分级降级
- 快速失败
- 用户透明
- 监控记录

---

## 8. 如何处理 LLM 的不确定性输出？

### 答案

**处理策略**：

1. **结构化输出**：强制 JSON Schema
```python
result = client.generate(
    prompt="...",
    schema=MyModel,
    max_retries=3,
)
```

2. **重试机制**：格式错误自动重试

3. **后处理**：
```python
try:
    data = json.loads(response)
except JSONDecodeError:
    # 尝试修复
    data = extract_json(response)
```

4. **回退默认值**：
```python
value = parsed.get("field", default_value)
```

5. **人工兜底**：置信度低时转人工

---

## 附加题

### 9. 如何实现 AI 应用的审计追踪？

**关键信息**：
- 用户 ID
- 请求内容（脱敏）
- 响应内容
- 模型/参数
- 时间戳
- request_id

**存储**：
- 结构化日志
- 支持查询
- 保留期限

### 10. RAG vs Fine-tuning 在安全性方面的对比？

| 方面 | RAG | Fine-tuning |
|------|-----|-------------|
| 数据泄露 | 低风险（不修改模型） | 高风险（数据嵌入模型） |
| 内容控制 | 可控（检索结果） | 难控（模型内部） |
| 更新 | 更新文档即可 | 需要重新训练 |
| 追溯 | 可追溯来源 | 难以追溯 |


