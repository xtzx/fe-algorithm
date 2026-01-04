# 输出安全

## 概述

确保 LLM 输出不包含敏感信息、有害内容，且符合预期格式。

## 1. PII 过滤

### 1.1 检测 PII

```python
from ai_safety.guards import OutputFilter

filter = OutputFilter()

text = "Contact John at john@example.com or call 123-456-7890"
pii_matches = filter.detect_pii(text)

for match in pii_matches:
    print(f"类型: {match.pii_type}")
    print(f"值: {match.value}")
    print(f"位置: {match.start}-{match.end}")
```

### 1.2 移除/遮蔽 PII

```python
safe_text = filter.remove_pii(text)
# "Contact [NAME] at [EMAIL] or call [PHONE]"
```

### 1.3 支持的 PII 类型

| 类型 | 示例 |
|------|------|
| EMAIL | john@example.com |
| PHONE | 123-456-7890 |
| SSN | 123-45-6789 |
| CREDIT_CARD | 1234-5678-9012-3456 |
| IP_ADDRESS | 192.168.1.1 |

## 2. 内容审核

### 2.1 基础审核

```python
result = filter.moderate(text)

if not result.is_safe:
    print(f"原因: {result.reason}")
    print(f"类别: {result.categories}")
```

### 2.2 审核类别

- **violence**: 暴力内容
- **hate**: 仇恨言论
- **self_harm**: 自残相关
- **sexual**: 性相关内容

### 2.3 自定义审核

```python
# 添加自定义检测模式
filter.add_pattern(r"company_secret", category="confidential")
```

## 3. 格式验证

### 3.1 JSON 验证

```python
is_valid, error = filter.validate_json(response)

if not is_valid:
    print(f"JSON 无效: {error}")
```

### 3.2 格式检查

```python
# 检查 Markdown 格式
is_valid, error = filter.validate_format(response, "markdown")

# 检查代码格式
is_valid, error = filter.validate_format(response, "code")
```

## 4. 幻觉检测

### 4.1 分析输出

```python
from ai_safety.guards.output_filter import HallucinationDetector

detector = HallucinationDetector()

analysis = detector.analyze(
    text=response,
    context=retrieved_context,
)

print(f"幻觉风险: {analysis['hallucination_risk']}")
print(f"不确定性指标: {analysis['uncertainty_indicators']}")
print(f"过度自信指标: {analysis['overconfidence_indicators']}")
```

### 4.2 处理幻觉

```python
if analysis['hallucination_risk'] > 0.5:
    # 添加免责声明
    response += "\n\n注意：此回答可能不完全准确，建议核实。"
```

## 5. 安全响应模板

```python
# 安全的错误响应
def safe_error_response(error: str) -> str:
    # 不暴露内部错误详情
    return "抱歉，处理您的请求时出现了问题。请稍后重试。"

# 安全的拒绝响应
def safe_decline_response(reason: str) -> str:
    return "抱歉，我无法帮助处理这个请求。"
```

## 6. 最佳实践

1. **始终过滤 PII**：保护用户隐私
2. **分级审核**：根据场景调整严格度
3. **记录可疑内容**：用于改进
4. **格式验证**：确保输出可用
5. **回退策略**：验证失败时有备选方案


