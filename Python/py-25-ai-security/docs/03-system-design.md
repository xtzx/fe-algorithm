# 系统设计

## 概述

安全的 AI 服务需要从系统层面进行设计。

## 1. 隔离策略

### 1.1 权限分层

```
┌─────────────────────────────────────┐
│           用户层                     │
│  - 只能发送文本输入                  │
│  - 不能直接访问模型                  │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│           网关层                     │
│  - 输入过滤                         │
│  - 速率限制                         │
│  - 认证授权                         │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│           服务层                     │
│  - 业务逻辑                         │
│  - 上下文管理                       │
│  - 输出过滤                         │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│           模型层                     │
│  - LLM 调用                         │
│  - 成本控制                         │
└─────────────────────────────────────┘
```

### 1.2 数据隔离

```python
# 每个用户的上下文隔离
class UserSession:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.context = []
        self.token_budget = TokenBudgetManager(daily_limit=10000)
    
    def add_message(self, role: str, content: str):
        # 只能访问自己的上下文
        self.context.append({"role": role, "content": content})
```

## 2. 权限控制

### 2.1 用户权限

```python
from enum import Enum

class UserTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

TIER_LIMITS = {
    UserTier.FREE: {
        "daily_tokens": 10000,
        "max_context_length": 2000,
        "models": ["gpt-3.5-turbo"],
    },
    UserTier.PRO: {
        "daily_tokens": 100000,
        "max_context_length": 8000,
        "models": ["gpt-3.5-turbo", "gpt-4o-mini"],
    },
    UserTier.ENTERPRISE: {
        "daily_tokens": 1000000,
        "max_context_length": 32000,
        "models": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"],
    },
}
```

### 2.2 功能权限

```python
def check_permission(user: User, feature: str) -> bool:
    """检查用户是否有权限使用某功能"""
    user_features = TIER_FEATURES.get(user.tier, set())
    return feature in user_features
```

## 3. 审计日志

### 3.1 日志结构

```python
import structlog
from datetime import datetime

logger = structlog.get_logger()

def log_request(
    user_id: str,
    action: str,
    input_text: str,
    output_text: str,
    model: str,
    tokens: int,
    latency_ms: float,
):
    logger.info(
        "llm_request",
        user_id=user_id,
        action=action,
        input_length=len(input_text),
        output_length=len(output_text),
        model=model,
        tokens=tokens,
        latency_ms=latency_ms,
        timestamp=datetime.utcnow().isoformat(),
    )
```

### 3.2 敏感操作日志

```python
def log_security_event(
    event_type: str,
    severity: str,
    details: dict,
):
    logger.warning(
        "security_event",
        event_type=event_type,
        severity=severity,
        **details,
    )
```

## 4. 回退策略

### 4.1 分级回退

```python
async def call_with_fallback(prompt: str):
    """带回退的 LLM 调用"""
    try:
        # 主模型
        return await call_gpt4(prompt)
    except RateLimitError:
        # 回退到备用模型
        return await call_gpt35(prompt)
    except Exception:
        # 回退到预设响应
        return "抱歉，服务暂时不可用。"
```

### 4.2 超时回退

```python
import asyncio

async def call_with_timeout(prompt: str, timeout: float = 30.0):
    try:
        return await asyncio.wait_for(
            call_llm(prompt),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return "请求超时，请稍后重试。"
```

## 5. 配置管理

### 5.1 安全配置

```python
from pydantic_settings import BaseSettings

class SecuritySettings(BaseSettings):
    # 输入限制
    max_input_length: int = 10000
    max_context_messages: int = 20
    
    # 速率限制
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60
    
    # 内容过滤
    enable_pii_filter: bool = True
    enable_injection_detection: bool = True
    
    # 成本控制
    daily_budget_usd: float = 100.0
    
    class Config:
        env_prefix = "AI_SECURITY_"
```

### 5.2 动态配置

```python
# 支持热更新的配置
class DynamicConfig:
    def __init__(self):
        self._config = {}
        self._last_update = None
    
    def get(self, key: str, default=None):
        self._refresh_if_needed()
        return self._config.get(key, default)
```

## 6. 最佳实践

1. **最小权限原则**：只授予必要权限
2. **纵深防御**：多层安全措施
3. **完整审计**：记录所有操作
4. **故障隔离**：单点故障不影响整体
5. **定期审查**：检查安全配置


