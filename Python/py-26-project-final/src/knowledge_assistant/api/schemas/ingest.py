"""
文档摄取相关模型
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IngestStatus(str, Enum):
    """摄取状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentInfo(BaseModel):
    """文档信息"""
    doc_id: str = Field(..., description="文档 ID")
    filename: str = Field(..., description="文件名")
    source: str = Field(..., description="来源路径")
    size: int = Field(..., description="文件大小（字节）")
    chunk_count: int = Field(0, description="分块数量")
    created_at: datetime = Field(..., description="创建时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class IngestResponse(BaseModel):
    """文档摄取响应"""
    status: IngestStatus = Field(..., description="处理状态")
    message: str = Field(..., description="状态消息")
    documents: List[DocumentInfo] = Field(default_factory=list, description="已处理的文档")
    total_chunks: int = Field(0, description="总块数")
    errors: List[str] = Field(default_factory=list, description="错误信息")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "completed",
                    "message": "成功处理 3 个文档",
                    "documents": [
                        {
                            "doc_id": "doc_abc123",
                            "filename": "guide.pdf",
                            "source": "/uploads/guide.pdf",
                            "size": 102400,
                            "chunk_count": 15,
                            "created_at": "2024-01-01T00:00:00Z",
                            "metadata": {"pages": 10},
                        }
                    ],
                    "total_chunks": 45,
                    "errors": [],
                }
            ]
        }
    }


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: List[DocumentInfo]
    total: int
    page: int
    page_size: int


class DeleteResponse(BaseModel):
    """删除响应"""
    success: bool
    message: str
    deleted_count: int = 0


