"""
问答查询路由
"""

import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from knowledge_assistant.api.dependencies.auth import get_current_active_user, get_optional_user
from knowledge_assistant.api.dependencies.services import get_rag_service, get_retriever
from knowledge_assistant.api.schemas.auth import UserResponse
from knowledge_assistant.api.schemas.query import (
    CitationResponse,
    ConversationMessage,
    HistoryResponse,
    QueryRequest,
    QueryResponse,
)
from knowledge_assistant.safety.input_guard import InputGuard
from knowledge_assistant.safety.output_guard import OutputGuard

logger = structlog.get_logger()
router = APIRouter()

# 会话存储（简单内存实现，生产环境应使用 Redis）
_conversations: Dict[str, List[Dict[str, Any]]] = {}


def get_or_create_conversation(conversation_id: Optional[str]) -> tuple[str, List[Dict[str, Any]]]:
    """获取或创建会话"""
    if conversation_id and conversation_id in _conversations:
        return conversation_id, _conversations[conversation_id]
    
    new_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
    _conversations[new_id] = []
    return new_id, _conversations[new_id]


@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    current_user: Optional[UserResponse] = Depends(get_optional_user),
) -> QueryResponse:
    """
    问答查询
    
    支持多轮对话和引用来源
    """
    # 输入检查
    input_guard = InputGuard()
    check_result = input_guard.check(request.question)
    
    if not check_result.is_safe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"输入不合法: {', '.join(check_result.issues)}",
        )
    
    # 获取会话
    conv_id, history = get_or_create_conversation(request.conversation_id)
    
    # 转换历史格式
    history_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history[-6:]  # 最近 3 轮
    ]
    
    # 生成回答
    generator = get_rag_service()
    
    try:
        response = generator.generate(
            question=request.question,
            history=history_messages if history_messages else None,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.error("query_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成回答时发生错误: {str(e)}",
        )
    
    # 输出检查
    output_guard = OutputGuard()
    moderation_result = output_guard.moderate(response.answer)
    
    if not moderation_result.is_safe:
        logger.warning("output_moderated", reason=moderation_result.reason)
        response.answer = "抱歉，生成的回答包含不适当的内容，已被过滤。请尝试重新提问。"
        response.citations = []
    
    # 更新会话历史
    history.append({
        "role": "user",
        "content": request.question,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    history.append({
        "role": "assistant",
        "content": response.answer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    
    # 构建响应
    citations = [
        CitationResponse(
            index=i + 1,
            text=cite.text,
            source=cite.source,
            chunk_id=cite.chunk_id,
            doc_id=cite.doc_id,
            page=cite.page,
            section=cite.section,
        )
        for i, cite in enumerate(response.citations)
    ]
    
    return QueryResponse(
        answer=response.answer,
        citations=citations,
        conversation_id=conv_id,
        metadata=response.metadata,
    )


@router.post("/stream")
async def query_stream(
    request: QueryRequest,
    current_user: Optional[UserResponse] = Depends(get_optional_user),
) -> EventSourceResponse:
    """
    流式问答查询
    
    使用 Server-Sent Events (SSE) 返回流式响应
    """
    # 输入检查
    input_guard = InputGuard()
    check_result = input_guard.check(request.question)
    
    if not check_result.is_safe:
        async def error_stream():
            yield {
                "event": "error",
                "data": f"输入不合法: {', '.join(check_result.issues)}",
            }
        return EventSourceResponse(error_stream())
    
    # 获取会话
    conv_id, history = get_or_create_conversation(request.conversation_id)
    
    # 转换历史格式
    history_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history[-6:]
    ]
    
    generator = get_rag_service()
    retriever = get_retriever()
    
    async def event_generator() -> AsyncIterator[Dict[str, Any]]:
        full_answer = ""
        
        try:
            # 先发送引用信息
            _, citations = retriever.search_with_citations(request.question, request.top_k)
            
            citations_data = [
                {
                    "index": i + 1,
                    "text": cite.text,
                    "source": cite.source,
                    "chunk_id": cite.chunk_id,
                    "doc_id": cite.doc_id,
                    "page": cite.page,
                    "section": cite.section,
                }
                for i, cite in enumerate(citations)
            ]
            
            yield {
                "event": "citations",
                "data": {"citations": citations_data, "conversation_id": conv_id},
            }
            
            # 流式生成回答
            async for chunk in generator.generate_stream(
                question=request.question,
                history=history_messages if history_messages else None,
                top_k=request.top_k,
            ):
                full_answer += chunk
                yield {
                    "event": "content",
                    "data": {"delta": chunk},
                }
            
            # 更新会话历史
            history.append({
                "role": "user",
                "content": request.question,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            history.append({
                "role": "assistant",
                "content": full_answer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            # 发送完成信号
            yield {
                "event": "done",
                "data": {"conversation_id": conv_id},
            }
            
        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield {
                "event": "error",
                "data": {"message": str(e)},
            }
    
    return EventSourceResponse(event_generator())


@router.get("/history/{conversation_id}", response_model=HistoryResponse)
async def get_conversation_history(
    conversation_id: str,
    current_user: UserResponse = Depends(get_current_active_user),
) -> HistoryResponse:
    """
    获取对话历史
    """
    if conversation_id not in _conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话不存在: {conversation_id}",
        )
    
    history = _conversations[conversation_id]
    
    messages = [
        ConversationMessage(
            role=msg["role"],
            content=msg["content"],
            timestamp=datetime.fromisoformat(msg["timestamp"]),
        )
        for msg in history
    ]
    
    # 获取时间戳
    if history:
        created_at = datetime.fromisoformat(history[0]["timestamp"])
        updated_at = datetime.fromisoformat(history[-1]["timestamp"])
    else:
        created_at = updated_at = datetime.now(timezone.utc)
    
    return HistoryResponse(
        conversation_id=conversation_id,
        messages=messages,
        created_at=created_at,
        updated_at=updated_at,
    )


@router.delete("/history/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: UserResponse = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """
    删除对话历史
    """
    if conversation_id not in _conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话不存在: {conversation_id}",
        )
    
    del _conversations[conversation_id]
    
    return {"success": True, "message": f"已删除会话 {conversation_id}"}


@router.get("/search")
async def search_documents(
    q: str,
    top_k: int = 5,
    current_user: Optional[UserResponse] = Depends(get_optional_user),
) -> Dict[str, Any]:
    """
    简单搜索（不生成回答，只返回相关文档）
    """
    retriever = get_retriever()
    results = retriever.search(q, top_k=top_k)
    
    return {
        "query": q,
        "results": [
            {
                "content": r.content,
                "score": r.score,
                "source": r.source,
                "chunk_id": r.chunk_id,
                "metadata": r.metadata,
            }
            for r in results
        ],
    }


