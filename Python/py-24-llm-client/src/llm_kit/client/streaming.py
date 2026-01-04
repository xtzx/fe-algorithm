"""
流式处理

特性:
- SSE 解析
- 增量内容收集
- 中断处理
- 工具调用收集
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

import structlog

from llm_kit.client.base import StreamChunk

logger = structlog.get_logger()


@dataclass
class StreamState:
    """流式状态"""
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    chunk_count: int = 0
    
    @property
    def is_complete(self) -> bool:
        return self.finish_reason is not None


class StreamProcessor:
    """
    流式处理器
    
    用于收集和处理流式输出
    
    Usage:
        processor = StreamProcessor()
        
        for chunk in client.chat_stream(messages):
            # 处理增量内容
            processor.process(chunk)
            print(chunk.delta, end="", flush=True)
        
        # 获取完整结果
        result = processor.get_result()
        print(f"\\nTotal: {result.content}")
    """

    def __init__(self, on_chunk: Optional[Callable[[StreamChunk], None]] = None):
        self._state = StreamState()
        self._on_chunk = on_chunk
        self._tool_call_buffer: Dict[int, Dict] = {}

    def process(self, chunk: StreamChunk):
        """处理单个块"""
        self._state.chunk_count += 1
        
        # 收集内容
        if chunk.delta:
            self._state.content += chunk.delta
        
        # 收集工具调用
        if chunk.tool_calls:
            self._collect_tool_calls(chunk.tool_calls)
        
        # 完成状态
        if chunk.finish_reason:
            self._state.finish_reason = chunk.finish_reason
            self._finalize_tool_calls()
        
        # 回调
        if self._on_chunk:
            self._on_chunk(chunk)

    def _collect_tool_calls(self, tool_calls: List[Dict[str, Any]]):
        """增量收集工具调用"""
        for tc in tool_calls:
            idx = tc.get("index", 0)
            
            if idx not in self._tool_call_buffer:
                self._tool_call_buffer[idx] = {
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {"name": "", "arguments": ""},
                }
            
            buf = self._tool_call_buffer[idx]
            
            if tc.get("id"):
                buf["id"] = tc["id"]
            
            if "function" in tc:
                if tc["function"].get("name"):
                    buf["function"]["name"] = tc["function"]["name"]
                if tc["function"].get("arguments"):
                    buf["function"]["arguments"] += tc["function"]["arguments"]

    def _finalize_tool_calls(self):
        """完成工具调用收集"""
        if self._tool_call_buffer:
            self._state.tool_calls = [
                self._tool_call_buffer[i]
                for i in sorted(self._tool_call_buffer.keys())
            ]

    def get_result(self) -> StreamState:
        """获取完整结果"""
        return self._state

    def reset(self):
        """重置状态"""
        self._state = StreamState()
        self._tool_call_buffer = {}


class StreamCollector:
    """
    流式收集器
    
    简化的流式内容收集
    
    Usage:
        content = ""
        for chunk in StreamCollector.collect(client.chat_stream(messages)):
            content += chunk.delta
            print(chunk.delta, end="")
    """

    @staticmethod
    def collect(
        stream: Iterator[StreamChunk],
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> Iterator[StreamChunk]:
        """
        收集流式输出
        
        Args:
            stream: 流式迭代器
            on_delta: 增量回调
        
        Yields:
            StreamChunk
        """
        for chunk in stream:
            if on_delta and chunk.delta:
                on_delta(chunk.delta)
            yield chunk

    @staticmethod
    def to_string(stream: Iterator[StreamChunk]) -> str:
        """收集流式输出为字符串"""
        parts = []
        for chunk in stream:
            if chunk.delta:
                parts.append(chunk.delta)
        return "".join(parts)


class StreamInterrupter:
    """
    流式中断器
    
    支持在流式输出过程中中断
    
    Usage:
        interrupter = StreamInterrupter()
        
        def on_user_cancel():
            interrupter.interrupt()
        
        for chunk in interrupter.wrap(client.chat_stream(messages)):
            print(chunk.delta, end="")
            if should_stop:
                interrupter.interrupt()
    """

    def __init__(self):
        self._interrupted = False
        self._content_so_far = ""

    def interrupt(self):
        """中断流式输出"""
        self._interrupted = True

    def wrap(
        self,
        stream: Iterator[StreamChunk],
        collect_content: bool = True,
    ) -> Iterator[StreamChunk]:
        """
        包装流式迭代器，支持中断
        
        Args:
            stream: 原始流
            collect_content: 是否收集内容
        
        Yields:
            StreamChunk
        """
        try:
            for chunk in stream:
                if self._interrupted:
                    logger.info("stream_interrupted", content_length=len(self._content_so_far))
                    break
                
                if collect_content and chunk.delta:
                    self._content_so_far += chunk.delta
                
                yield chunk
        finally:
            pass

    @property
    def content(self) -> str:
        """获取已收集的内容"""
        return self._content_so_far

    @property
    def was_interrupted(self) -> bool:
        """是否被中断"""
        return self._interrupted

    def reset(self):
        """重置状态"""
        self._interrupted = False
        self._content_so_far = ""


