"""
查询相关模型
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """问答查询请求"""
    question: str = Field(..., min_length=1, max_length=10000, description="用户问题")
    conversation_id: Optional[str] = Field(None, description="会话 ID，用于多轮对话")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="检索数量")
    stream: bool = Field(False, description="是否流式响应")
    filters: Optional[Dict[str, Any]] = Field(None, description="元数据过滤器")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "什么是 RAG？",
                    "conversation_id": "conv_123",
                    "top_k": 5,
                    "stream": True,
                }
            ]
        }
    }


class CitationResponse(BaseModel):
    """引用响应"""
    index: int = Field(..., description="引用序号")
    text: str = Field(..., description="引用文本片段")
    source: str = Field(..., description="来源文件")
    chunk_id: str = Field(..., description="块 ID")
    doc_id: str = Field(..., description="文档 ID")
    page: Optional[int] = Field(None, description="页码")
    section: Optional[str] = Field(None, description="章节")


class QueryResponse(BaseModel):
    """问答查询响应"""
    answer: str = Field(..., description="回答内容")
    citations: List[CitationResponse] = Field(default_factory=list, description="引用列表")
    conversation_id: str = Field(..., description="会话 ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "RAG（检索增强生成）是一种结合检索和生成的 AI 技术...[1]",
                    "citations": [
                        {
                            "index": 1,
                            "text": "RAG 是一种技术...",
                            "source": "rag_intro.md",
                            "chunk_id": "rag_intro_chunk_0",
                            "doc_id": "rag_intro_abc123",
                        }
                    ],
                    "conversation_id": "conv_123",
                    "metadata": {"retrieval_count": 5},
                }
            ]
        }
    }


class ConversationMessage(BaseModel):
    """对话消息"""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    timestamp: datetime


class HistoryResponse(BaseModel):
    """对话历史响应"""
    conversation_id: str
    messages: List[ConversationMessage]
    created_at: datetime
    updated_at: datetime


