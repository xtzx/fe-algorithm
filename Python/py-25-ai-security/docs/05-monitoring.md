# 生产监控

## 概述

实时监控 AI 服务的质量、成本和异常。

## 1. 质量监控

### 1.1 响应质量

```python
from ai_safety.monitoring import QualityMonitor

monitor = QualityMonitor()

# 记录每次响应
monitor.record_response(
    response=output,
    latency_ms=latency,
    model="gpt-4o-mini",
    tokens=usage.total_tokens,
    was_blocked=False,
)

# 获取质量报告
report = monitor.get_quality_report()
print(f"错误率: {report['error_rate']:.2%}")
print(f"阻止率: {report['block_rate']:.2%}")
print(f"平均延迟: {report['latency']['avg']:.2f}ms")
```

### 1.2 关键指标

| 指标 | 描述 | 阈值建议 |
|------|------|----------|
| 错误率 | 失败请求比例 | < 1% |
| 延迟 P95 | 95% 请求延迟 | < 3s |
| 阻止率 | 被安全过滤的比例 | < 5% |

## 2. 成本监控

### 2.1 使用追踪

```python
from ai_safety.monitoring import CostMonitor

monitor = CostMonitor(daily_budget=100.0)

# 记录每次调用
monitor.record_usage(
    model="gpt-4o-mini",
    input_tokens=100,
    output_tokens=200,
)

# 检查预算
if monitor.is_over_budget():
    alert("Daily budget exceeded!")
```

### 2.2 成本报告

```python
report = monitor.get_cost_report()
print(f"今日花费: ${report['today_cost']:.2f}")
print(f"预算使用: {report['budget_used_percent']:.1f}%")
```

### 2.3 成本优化

1. **模型选择**：简单任务用便宜模型
2. **缓存**：相同查询复用结果
3. **截断**：限制上下文长度
4. **批处理**：合并请求

## 3. 异常告警

### 3.1 配置告警

```python
from ai_safety.monitoring import AlertManager, AlertLevel

alert_mgr = AlertManager()

# 延迟告警
alert_mgr.add_rule(
    name="high_latency",
    metric="latency_ms",
    threshold=3000,
    level=AlertLevel.WARNING,
)

# 错误率告警
alert_mgr.add_rule(
    name="high_error_rate",
    metric="error_rate",
    threshold=0.05,
    level=AlertLevel.ERROR,
)

# 成本告警
alert_mgr.add_rule(
    name="budget_warning",
    metric="daily_cost",
    threshold=80,
    level=AlertLevel.WARNING,
)
```

### 3.2 告警处理

```python
# 添加告警处理器
def send_slack_alert(alert):
    # 发送 Slack 通知
    pass

alert_mgr.add_handler(send_slack_alert)

# 检查告警
alerts = alert_mgr.check(monitor)
```

## 4. 日志记录

### 4.1 结构化日志

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "llm_request",
    user_id=user_id,
    model=model,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    latency_ms=latency_ms,
    request_id=request_id,
)
```

### 4.2 日志级别

| 级别 | 用途 |
|------|------|
| DEBUG | 开发调试 |
| INFO | 正常请求 |
| WARNING | 可疑活动、性能问题 |
| ERROR | 错误、失败 |

## 5. 仪表盘

### 5.1 关键看板

```
┌────────────────────────────────────────────────────┐
│  AI 服务监控仪表盘                                  │
├────────────────────────────────────────────────────┤
│                                                    │
│  请求量     延迟 P95    错误率      日成本         │
│  ████████   ████████   ████████   ████████        │
│  12.3k/h    1.2s       0.5%       $45.20          │
│                                                    │
├────────────────────────────────────────────────────┤
│  最近告警                                          │
│  • [WARNING] high_latency - 2 min ago             │
│  • [INFO] budget at 80% - 1 hour ago              │
│                                                    │
└────────────────────────────────────────────────────┘
```

### 5.2 Prometheus 指标

```python
# 暴露 Prometheus 指标
from prometheus_client import Counter, Histogram

llm_requests = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["model", "status"],
)

llm_latency = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency",
    ["model"],
)
```

## 6. 最佳实践

1. **实时监控**：及时发现问题
2. **历史分析**：识别趋势
3. **自动告警**：减少人工干预
4. **分级响应**：不同级别不同处理
5. **定期审查**：优化阈值


