"""
OpenAI 客户端实现

使用 httpx 而非官方 SDK，保持灵活性
"""

import json
import time
from typing import Any, Dict, Iterator, List, Optional

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_kit.client.base import (
    AuthenticationError,
    BaseLLMClient,
    ChatMessage,
    ChatResponse,
    InvalidRequestError,
    LLMError,
    RateLimitError,
    StreamChunk,
    TokenUsage,
)

logger = structlog.get_logger()


class OpenAIClient(BaseLLMClient):
    """
    OpenAI 兼容客户端
    
    支持:
    - OpenAI API
    - Azure OpenAI
    - 本地 LLM（如 Ollama、vLLM）
    - 其他兼容 API
    
    Usage:
        # OpenAI
        client = OpenAIClient(api_key="sk-...")
        
        # Azure
        client = OpenAIClient(
            api_key="...",
            base_url="https://xxx.openai.azure.com/openai/deployments/gpt-4",
        )
        
        # 本地
        client = OpenAIClient(
            api_key="not-needed",
            base_url="http://localhost:11434/v1",
        )
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        organization: Optional[str] = None,
    ):
        super().__init__(api_key, base_url, timeout, max_retries)
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.organization = organization
        
        # 创建 HTTP 客户端
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._get_headers(),
            timeout=httpx.Timeout(timeout),
        )

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    def _handle_error(self, response: httpx.Response):
        """处理错误响应"""
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text

        if response.status_code == 401:
            raise AuthenticationError(error_msg)
        elif response.status_code == 429:
            raise RateLimitError(error_msg)
        elif response.status_code >= 400:
            raise InvalidRequestError(f"Status {response.status_code}: {error_msg}")

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
    )
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
            temperature: 温度参数 (0-2)
            max_tokens: 最大输出 token 数
            request_id: 幂等请求 ID
            **kwargs: 其他参数 (top_p, presence_penalty, etc.)
        
        Returns:
            ChatResponse
        """
        request_id = request_id or self.generate_request_id()
        start_time = time.perf_counter()

        # 构建请求
        payload = {
            "model": model,
            "messages": self._normalize_messages(messages),
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        logger.info(
            "llm_request_started",
            request_id=request_id,
            model=model,
            message_count=len(messages),
        )

        try:
            response = self._client.post("/chat/completions", json=payload)
            self._handle_error(response)
            data = response.json()
        except httpx.TimeoutException:
            raise LLMError(f"Request timeout after {self.timeout}s")

        latency_ms = (time.perf_counter() - start_time) * 1000

        # 解析响应
        choice = data["choices"][0]
        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        self._record_usage(usage)

        result = ChatResponse(
            content=choice["message"].get("content", ""),
            model=data.get("model", model),
            usage=usage,
            request_id=request_id,
            finish_reason=choice.get("finish_reason"),
            tool_calls=choice["message"].get("tool_calls"),
            latency_ms=latency_ms,
        )

        logger.info(
            "llm_request_completed",
            request_id=request_id,
            model=model,
            tokens=usage.total_tokens,
            latency_ms=round(latency_ms, 2),
        )

        return result

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
        request_id = request_id or self.generate_request_id()

        # 构建请求
        payload = {
            "model": model,
            "messages": self._normalize_messages(messages),
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        logger.info(
            "llm_stream_started",
            request_id=request_id,
            model=model,
        )

        try:
            with self._client.stream("POST", "/chat/completions", json=payload) as response:
                self._handle_error(response)
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    # SSE 格式: data: {...}
                    if line.startswith("data: "):
                        data_str = line[6:]
                        
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            
                            yield StreamChunk(
                                delta=delta.get("content", ""),
                                finish_reason=choice.get("finish_reason"),
                                tool_calls=delta.get("tool_calls"),
                            )
                        except json.JSONDecodeError:
                            continue

        except httpx.TimeoutException:
            raise LLMError(f"Stream timeout after {self.timeout}s")

        logger.info("llm_stream_completed", request_id=request_id)

    def close(self):
        """关闭客户端"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


