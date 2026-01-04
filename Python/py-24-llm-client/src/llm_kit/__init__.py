"""
LLM 客户端与 RAG 工具包

提供:
- LLM 客户端抽象
- 结构化输出
- 流式处理
- RAG 系统组件
- 提示工程工具
"""

__version__ = "1.0.0"

from llm_kit.client import BaseLLMClient, OpenAIClient
from llm_kit.rag import Chunker, DocumentLoader, Embedder, Retriever, VectorIndex

__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    "DocumentLoader",
    "Chunker",
    "Embedder",
    "VectorIndex",
    "Retriever",
]


