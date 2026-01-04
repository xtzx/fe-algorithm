"""
LLM 客户端基础抽象

特性:
- 不绑定厂商 SDK
- timeout/retry/streaming
- 幂等 request_id
- 成本/耗时统计
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

import structlog

logger = structlog.get_logger()


class Role(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    """聊天消息"""
    role: Role | str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = {"role": str(self.role.value if isinstance(self.role, Role) else self.role)}
        if self.content:
            d["content"] = self.content
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class TokenUsage:
    """Token 使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def estimated_cost(self) -> float:
        """估算成本（基于 GPT-4o-mini 价格）"""
        # 简化的成本估算
        input_cost = self.prompt_tokens * 0.00015 / 1000
        output_cost = self.completion_tokens * 0.0006 / 1000
        return input_cost + output_cost


@dataclass
class ChatResponse:
    """聊天响应"""
    content: str
    model: str
    usage: TokenUsage
    request_id: str
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    # 统计信息
    latency_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        return f"ChatResponse(content='{self.content[:50]}...', tokens={self.usage.total_tokens})"


@dataclass
class StreamChunk:
    """流式输出块"""
    delta: str
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    @property
    def is_done(self) -> bool:
        return self.finish_reason is not None


class LLMError(Exception):
    """LLM 错误基类"""
    pass


class RateLimitError(LLMError):
    """速率限制错误"""
    pass


class AuthenticationError(LLMError):
    """认证错误"""
    pass


class InvalidRequestError(LLMError):
    """无效请求错误"""
    pass


class BaseLLMClient(ABC):
    """
    LLM 客户端基础抽象
    
    特性:
    - 不绑定厂商 SDK（使用 httpx）
    - timeout/retry 配置
    - 幂等 request_id
    - 成本/耗时统计
    
    Usage:
        class MyClient(BaseLLMClient):
            def chat(self, messages, **kwargs):
                # 实现具体逻辑
                ...
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 统计
        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0

    def generate_request_id(self) -> str:
        """生成幂等请求 ID"""
        return str(uuid.uuid4())

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage | Dict[str, Any]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        """
        聊天补全
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            request_id: 幂等请求 ID
            **kwargs: 其他参数
        
        Returns:
            ChatResponse
        """
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: List[ChatMessage | Dict[str, Any]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """
        流式聊天补全
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            request_id: 幂等请求 ID
            **kwargs: 其他参数
        
        Yields:
            StreamChunk
        """
        pass

    def _normalize_messages(
        self, messages: List[ChatMessage | Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """标准化消息格式"""
        result = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                result.append(msg.to_dict())
            elif isinstance(msg, dict):
                result.append(msg)
            else:
                raise InvalidRequestError(f"Invalid message type: {type(msg)}")
        return result

    def _record_usage(self, usage: TokenUsage):
        """记录使用统计"""
        self._total_requests += 1
        self._total_tokens += usage.total_tokens
        self._total_cost += usage.estimated_cost

    def get_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_cost": round(self._total_cost, 6),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(base_url={self.base_url})"


