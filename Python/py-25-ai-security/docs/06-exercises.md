# 练习题

## 练习 1: 注入检测

实现一个检测提示注入的函数：

```python
def detect_injection(text: str) -> dict:
    """
    检测提示注入
    
    返回:
    {
        "is_injection": bool,
        "type": str,  # direct, indirect, jailbreak, none
        "risk_level": str,  # low, medium, high, critical
    }
    """
    pass
```

---

## 练习 2: PII 过滤

实现 PII 检测和遮蔽：

**要求**：
1. 检测邮箱、电话、身份证号
2. 支持自定义遮蔽格式
3. 返回检测到的 PII 列表

---

## 练习 3: 输入过滤器

实现完整的输入过滤器：

**要求**：
1. 长度检查
2. 注入检测
3. 敏感词过滤
4. 返回详细的检查结果

---

## 练习 4: 评测数据集

设计一个 RAG 评测数据集：

**要求**：
1. 至少 20 个测试用例
2. 包含 question, answer, context
3. 覆盖不同难度和类型

---

## 练习 5: 评测指标

实现 BLEU 或 ROUGE 评测指标：

```python
def calculate_bleu(reference: str, prediction: str) -> float:
    """计算 BLEU 分数"""
    pass
```

---

## 练习 6: LLM-as-Judge

实现 LLM 评判器：

**要求**：
1. 设计评判提示
2. 解析评分结果
3. 支持多个评判标准

---

## 练习 7: 成本监控

实现详细的成本追踪：

```python
class CostTracker:
    def record(self, model, input_tokens, output_tokens): ...
    def get_daily_report(self) -> dict: ...
    def get_monthly_report(self) -> dict: ...
    def get_by_model(self) -> dict: ...
```

---

## 练习 8: 告警系统

实现告警管理器：

**要求**：
1. 支持多种告警规则
2. 告警冷却（避免重复告警）
3. 多种通知方式

---

## 练习 9: 速率限制

实现滑动窗口速率限制：

```python
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int): ...
    def allow(self, user_id: str) -> bool: ...
    def get_remaining(self, user_id: str) -> int: ...
```

---

## 练习 10: 安全 API

设计一个安全的 LLM API：

**要求**：
1. 输入过滤
2. 输出过滤
3. 速率限制
4. 成本控制
5. 审计日志

---

## 参考答案

### 练习 1 答案

```python
import re

INJECTION_PATTERNS = {
    "direct": [
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"forget\s+your\s+rules",
    ],
    "jailbreak": [
        r"(DAN|do\s+anything\s+now)",
        r"no\s+restrictions?",
    ],
}

def detect_injection(text: str) -> dict:
    text_lower = text.lower()
    
    for injection_type, patterns in INJECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return {
                    "is_injection": True,
                    "type": injection_type,
                    "risk_level": "high" if injection_type == "jailbreak" else "medium",
                }
    
    return {
        "is_injection": False,
        "type": "none",
        "risk_level": "low",
    }
```

### 练习 9 答案

```python
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = defaultdict(list)
    
    def allow(self, user_id: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        
        # 清理过期请求
        self._requests[user_id] = [
            t for t in self._requests[user_id] if t > cutoff
        ]
        
        if len(self._requests[user_id]) >= self.max_requests:
            return False
        
        self._requests[user_id].append(now)
        return True
    
    def get_remaining(self, user_id: str) -> int:
        now = time.time()
        cutoff = now - self.window_seconds
        
        recent = [t for t in self._requests.get(user_id, []) if t > cutoff]
        return max(0, self.max_requests - len(recent))
```


