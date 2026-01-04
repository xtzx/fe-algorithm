# P25: AI 服务安全与评测

> 构建安全可靠的 AI 服务

## 🎯 学完后能做

- 防护提示注入
- 实现内容安全
- 评测 AI 系统

## 📁 目录结构

```
py-25-ai-security/
├── README.md
├── pyproject.toml
├── docs/
│   ├── 01-prompt-injection.md    # 提示注入防护
│   ├── 02-output-safety.md       # 输出安全
│   ├── 03-system-design.md       # 系统设计
│   ├── 04-evaluation.md          # 评测体系
│   ├── 05-monitoring.md          # 生产监控
│   ├── 06-exercises.md           # 练习题
│   └── 07-interview.md           # 面试题
├── src/ai_safety/
│   ├── __init__.py
│   ├── guards/
│   │   ├── __init__.py
│   │   ├── input_filter.py       # 输入过滤
│   │   ├── output_filter.py      # 输出过滤
│   │   └── injection.py          # 注入检测
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # 评测指标
│   │   ├── dataset.py            # 评测数据集
│   │   └── runner.py             # 评测运行器
│   └── monitoring/
│       ├── __init__.py
│       └── monitor.py            # 监控系统
├── tests/
├── examples/
└── scripts/
```

## 🚀 快速开始

### 安装

```bash
cd py-25-ai-security
pip install -e ".[dev]"
```

### 输入过滤

```python
from ai_safety.guards import InputFilter, InjectionDetector

# 创建过滤器
filter = InputFilter()

# 检查输入
result = filter.check("Tell me about Python")
if result.is_safe:
    # 安全，继续处理
    pass

# 注入检测
detector = InjectionDetector()
result = detector.detect("Ignore previous instructions and...")
print(result.is_injection)  # True
print(result.risk_level)    # "high"
```

### 输出安全

```python
from ai_safety.guards import OutputFilter

filter = OutputFilter()

# PII 过滤
output = "Contact John at john@example.com"
safe_output = filter.remove_pii(output)
# "Contact [NAME] at [EMAIL]"

# 内容审核
result = filter.moderate(content)
if not result.is_safe:
    print(f"Blocked: {result.reason}")
```

### 评测系统

```python
from ai_safety.evaluation import EvaluationRunner, Metrics

# 创建评测器
runner = EvaluationRunner()

# 定义测试用例
test_cases = [
    {"input": "What is Python?", "expected": "programming language"},
]

# 运行评测
results = runner.run(model, test_cases)
print(f"Accuracy: {results.accuracy:.2%}")
print(f"Relevance: {results.relevance:.2f}")
```

## 🔧 核心概念

### 1. 提示注入防护

```python
# 直接注入检测
detector.detect("Ignore previous instructions")

# 间接注入检测（来自外部数据）
detector.detect_in_context(user_data)

# 越狱检测
detector.detect_jailbreak("DAN prompt...")
```

### 2. 输出安全

```python
# PII 检测
filter.detect_pii(text)  # 检测 PII

# 内容审核
filter.moderate(text)  # 审核内容

# 格式验证
filter.validate_json(text)  # 验证 JSON
```

### 3. 评测体系

```
评测指标:
├── 准确性 (Accuracy) - 答案是否正确
├── 相关性 (Relevance) - 答案是否相关
├── 忠实度 (Faithfulness) - 是否基于上下文
└── 无害性 (Harmlessness) - 是否安全
```

## 📚 学习路径

1. **提示注入** - 攻击类型、检测方法
2. **输出安全** - PII、内容审核
3. **系统设计** - 隔离、权限、审计
4. **评测体系** - 指标、数据集、LLM-as-Judge
5. **生产监控** - 质量、成本、告警

## ✅ 功能清单

- [x] 直接注入检测
- [x] 间接注入检测
- [x] 越狱防护
- [x] 输入过滤
- [x] PII 过滤
- [x] 内容审核
- [x] 格式验证
- [x] 隔离策略
- [x] 权限控制
- [x] 审计日志
- [x] 准确性评测
- [x] 相关性评测
- [x] 忠实度评测
- [x] LLM-as-Judge
- [x] RAG 评测
- [x] 质量监控
- [x] 成本监控
- [x] 异常告警


