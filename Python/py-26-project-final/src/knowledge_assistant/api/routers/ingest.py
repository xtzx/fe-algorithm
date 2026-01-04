"""
文档摄取路由
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from knowledge_assistant.api.dependencies.auth import get_current_active_user, require_admin
from knowledge_assistant.api.dependencies.services import (
    get_embedder,
    get_vector_index,
    save_index,
)
from knowledge_assistant.api.schemas.auth import UserResponse
from knowledge_assistant.api.schemas.ingest import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    IngestResponse,
    IngestStatus,
)
from knowledge_assistant.config import get_settings
from knowledge_assistant.rag.chunker import Chunker
from knowledge_assistant.rag.loader import MultiFormatLoader

logger = structlog.get_logger()
router = APIRouter()


@router.post("/upload", response_model=IngestResponse)
async def upload_documents(
    files: List[UploadFile] = File(..., description="要上传的文档"),
    current_user: UserResponse = Depends(get_current_active_user),
) -> IngestResponse:
    """
    上传并处理文档
    
    支持:
    - PDF (.pdf)
    - Markdown (.md)
    - 文本文件 (.txt)
    """
    settings = get_settings()
    loader = MultiFormatLoader()
    chunker = Chunker(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    index = get_vector_index()
    
    documents_info = []
    errors = []
    total_chunks = 0
    
    for file in files:
        try:
            # 验证文件类型
            if not file.filename:
                errors.append("文件名为空")
                continue
            
            suffix = Path(file.filename).suffix.lower()
            if suffix not in {".pdf", ".md", ".txt", ".markdown"}:
                errors.append(f"不支持的文件类型: {file.filename}")
                continue
            
            # 保存文件
            upload_path = settings.upload_dir / file.filename
            settings.upload_dir.mkdir(parents=True, exist_ok=True)
            
            with open(upload_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # 加载文档
            doc = loader.load(upload_path)
            doc.metadata["uploaded_by"] = current_user.username
            doc.metadata["uploaded_at"] = datetime.now(timezone.utc).isoformat()
            
            # 分块
            chunks = chunker.split(doc)
            
            # 添加到索引
            index.add_chunks(chunks)
            
            documents_info.append(DocumentInfo(
                doc_id=doc.doc_id,
                filename=file.filename,
                source=str(upload_path),
                size=len(content),
                chunk_count=len(chunks),
                created_at=datetime.now(timezone.utc),
                metadata=doc.metadata,
            ))
            
            total_chunks += len(chunks)
            
            logger.info(
                "document_ingested",
                filename=file.filename,
                doc_id=doc.doc_id,
                chunks=len(chunks),
            )
            
        except Exception as e:
            errors.append(f"处理 {file.filename} 失败: {str(e)}")
            logger.error("ingest_error", filename=file.filename, error=str(e))
    
    # 保存索引
    if documents_info:
        save_index()
    
    status_val = IngestStatus.COMPLETED if not errors else (
        IngestStatus.FAILED if not documents_info else IngestStatus.COMPLETED
    )
    
    return IngestResponse(
        status=status_val,
        message=f"成功处理 {len(documents_info)} 个文档" + (
            f"，{len(errors)} 个失败" if errors else ""
        ),
        documents=documents_info,
        total_chunks=total_chunks,
        errors=errors,
    )


@router.post("/text", response_model=IngestResponse)
async def ingest_text(
    text: str,
    source: str = "direct_input",
    current_user: UserResponse = Depends(get_current_active_user),
) -> IngestResponse:
    """
    直接摄取文本内容
    """
    settings = get_settings()
    from knowledge_assistant.rag.loader import DocumentLoader
    
    loader = DocumentLoader()
    chunker = Chunker(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    index = get_vector_index()
    
    try:
        # 创建文档
        doc = loader.load_text(text, source=source)
        doc.metadata["uploaded_by"] = current_user.username
        doc.metadata["uploaded_at"] = datetime.now(timezone.utc).isoformat()
        
        # 分块
        chunks = chunker.split(doc)
        
        # 添加到索引
        index.add_chunks(chunks)
        
        # 保存索引
        save_index()
        
        return IngestResponse(
            status=IngestStatus.COMPLETED,
            message="文本摄取成功",
            documents=[DocumentInfo(
                doc_id=doc.doc_id,
                filename=source,
                source=source,
                size=len(text),
                chunk_count=len(chunks),
                created_at=datetime.now(timezone.utc),
                metadata=doc.metadata,
            )],
            total_chunks=len(chunks),
            errors=[],
        )
        
    except Exception as e:
        logger.error("text_ingest_error", error=str(e))
        return IngestResponse(
            status=IngestStatus.FAILED,
            message=f"摄取失败: {str(e)}",
            documents=[],
            total_chunks=0,
            errors=[str(e)],
        )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    current_user: UserResponse = Depends(get_current_active_user),
) -> DocumentListResponse:
    """
    列出已摄取的文档
    """
    index = get_vector_index()
    
    # 获取唯一文档
    doc_ids_seen = set()
    documents = []
    
    for entry in index._entries:
        doc_id = entry.metadata.get("doc_id", "")
        if doc_id and doc_id not in doc_ids_seen:
            doc_ids_seen.add(doc_id)
            
            # 统计该文档的块数
            chunk_count = sum(
                1 for e in index._entries
                if e.metadata.get("doc_id") == doc_id
            )
            
            documents.append(DocumentInfo(
                doc_id=doc_id,
                filename=entry.metadata.get("filename", "unknown"),
                source=entry.metadata.get("source", "unknown"),
                size=entry.metadata.get("size", 0),
                chunk_count=chunk_count,
                created_at=datetime.fromisoformat(
                    entry.metadata.get("uploaded_at", datetime.now(timezone.utc).isoformat())
                ),
                metadata=entry.metadata,
            ))
    
    # 分页
    total = len(documents)
    start = (page - 1) * page_size
    end = start + page_size
    documents = documents[start:end]
    
    return DocumentListResponse(
        documents=documents,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(
    doc_id: str,
    current_user: UserResponse = Depends(require_admin),
) -> DeleteResponse:
    """
    删除文档（需要管理员权限）
    """
    index = get_vector_index()
    
    # 计算要删除的块数
    chunks_to_delete = [
        entry.chunk_id for entry in index._entries
        if entry.metadata.get("doc_id") == doc_id
    ]
    
    if not chunks_to_delete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档不存在: {doc_id}",
        )
    
    # 删除
    index.delete_by_doc_id(doc_id)
    save_index()
    
    logger.info("document_deleted", doc_id=doc_id, chunks=len(chunks_to_delete))
    
    return DeleteResponse(
        success=True,
        message=f"已删除文档 {doc_id}",
        deleted_count=len(chunks_to_delete),
    )


@router.delete("/clear", response_model=DeleteResponse)
async def clear_all_documents(
    confirm: bool = False,
    current_user: UserResponse = Depends(require_admin),
) -> DeleteResponse:
    """
    清空所有文档（需要管理员权限）
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请设置 confirm=true 确认清空操作",
        )
    
    index = get_vector_index()
    count = len(index)
    index.clear()
    save_index()
    
    logger.warning("all_documents_cleared", count=count, user=current_user.username)
    
    return DeleteResponse(
        success=True,
        message="已清空所有文档",
        deleted_count=count,
    )


