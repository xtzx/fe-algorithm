"""
LLM 客户端

支持:
- OpenAI API
- Stub 模式（用于测试）
- 同步和异步调用
- 流式响应
"""

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"


@dataclass
class StreamChunk:
    """流式响应块"""
    delta: str = ""
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LLMClient:
    """
    LLM 客户端
    
    支持 OpenAI API 兼容接口
    
    Usage:
        client = LLMClient(api_key="sk-...", model="gpt-4o-mini")
        
        # 同步调用
        response = client.chat([{"role": "user", "content": "Hello"}])
        print(response.content)
        
        # 流式调用
        for chunk in client.chat_stream(messages):
            print(chunk.delta, end="")
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        provider: str = "openai",
        timeout: float = 60.0,
        max_tokens: int = 2048,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.timeout = timeout
        self.max_tokens = max_tokens
        
        self._client: Optional[Any] = None
        self._async_client: Optional[Any] = None
        
        if provider != "stub":
            self._init_client()

    def _init_client(self):
        """初始化 HTTP 客户端"""
        import httpx
        
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        
        self._async_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        同步聊天
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
        
        Returns:
            LLMResponse
        """
        if self.provider == "stub":
            return self._stub_response(messages)
        
        response = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or self.max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
        )

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """异步聊天"""
        if self.provider == "stub":
            return self._stub_response(messages)
        
        response = await self._async_client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or self.max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            usage=data.get("usage", {}),
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
        )

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Iterator[StreamChunk]:
        """
        流式聊天
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
        
        Yields:
            StreamChunk
        """
        if self.provider == "stub":
            yield from self._stub_stream(messages)
            return
        
        with self._client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line or line == "data: [DONE]":
                    continue
                
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        
                        yield StreamChunk(
                            delta=delta.get("content", ""),
                            finish_reason=choice.get("finish_reason"),
                            tool_calls=delta.get("tool_calls"),
                        )
                    except json.JSONDecodeError:
                        continue

    async def chat_stream_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        异步流式聊天
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
        
        Yields:
            StreamChunk
        """
        if self.provider == "stub":
            for chunk in self._stub_stream(messages):
                yield chunk
            return
        
        async with self._async_client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line or line == "data: [DONE]":
                    continue
                
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        
                        yield StreamChunk(
                            delta=delta.get("content", ""),
                            finish_reason=choice.get("finish_reason"),
                            tool_calls=delta.get("tool_calls"),
                        )
                    except json.JSONDecodeError:
                        continue

    def _stub_response(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Stub 模式响应"""
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        
        # 生成简单的测试响应
        content = f"这是对问题的回答。[1]\n\n您的问题是: {last_user_msg[:100]}\n\n基于提供的上下文，我的回答如下..."
        
        return LLMResponse(
            content=content,
            model="stub-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            finish_reason="stop",
        )

    def _stub_stream(self, messages: List[Dict[str, str]]) -> Iterator[StreamChunk]:
        """Stub 模式流式响应"""
        response = self._stub_response(messages)
        
        # 模拟流式输出
        words = response.content.split()
        for i, word in enumerate(words):
            yield StreamChunk(
                delta=word + (" " if i < len(words) - 1 else ""),
                finish_reason=None,
            )
        
        yield StreamChunk(delta="", finish_reason="stop")

    def close(self):
        """关闭客户端"""
        if self._client:
            self._client.close()

    async def aclose(self):
        """异步关闭客户端"""
        if self._async_client:
            await self._async_client.aclose()


def create_llm_client(
    provider: str = "stub",
    api_key: str = "",
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
) -> LLMClient:
    """
    工厂函数：创建 LLM 客户端
    
    Args:
        provider: 提供商 ("stub", "openai")
        api_key: API 密钥
        model: 模型名称
        base_url: API 基础 URL
    
    Returns:
        LLMClient
    """
    return LLMClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
    )


