"""
RAG 生成器

将检索结果与 LLM 结合生成回答
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import structlog

from knowledge_assistant.rag.retriever import Citation, Retriever, SearchResult

logger = structlog.get_logger()


@dataclass
class RAGResponse:
    """RAG 响应"""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    context_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "context_used": self.context_used,
            "metadata": self.metadata,
        }


# RAG 系统提示词
RAG_SYSTEM_PROMPT = """你是一个知识库助手。请根据提供的上下文信息回答用户的问题。

规则:
1. 仅基于提供的上下文回答问题
2. 如果上下文中没有相关信息，请明确告知用户
3. 在回答中引用来源，使用 [1], [2] 等标记
4. 保持回答简洁、准确
5. 使用与用户相同的语言回答"""


RAG_USER_PROMPT_TEMPLATE = """上下文信息:
{context}

---
来源列表:
{sources}

---
用户问题: {question}

请根据上下文回答问题，并在适当位置标注引用来源。"""


class RAGGenerator:
    """
    RAG 生成器
    
    将检索结果与 LLM 结合生成回答
    
    Usage:
        generator = RAGGenerator(retriever, llm_client)
        
        # 同步生成
        response = generator.generate("What is RAG?")
        
        # 流式生成
        async for chunk in generator.generate_stream("What is RAG?"):
            print(chunk, end="")
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: Any,  # LLMClient
        system_prompt: Optional[str] = None,
        max_context_tokens: int = 2000,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.system_prompt = system_prompt or RAG_SYSTEM_PROMPT
        self.max_context_tokens = max_context_tokens

    def generate(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        生成回答
        
        Args:
            question: 用户问题
            history: 对话历史
            top_k: 检索数量
        
        Returns:
            RAGResponse
        """
        # 检索相关内容
        results, citations = self.retriever.search_with_citations(question, top_k)
        
        if not results:
            return RAGResponse(
                answer="抱歉，我在知识库中没有找到与您问题相关的信息。请尝试用其他方式描述您的问题。",
                citations=[],
                context_used=[],
                metadata={"retrieval_count": 0},
            )
        
        # 构建上下文
        context = self._build_context(results)
        sources = self.retriever.format_citations(citations)
        
        # 构建消息
        messages = self._build_messages(question, context, sources, history)
        
        # 调用 LLM
        response = self.llm_client.chat(messages)
        
        return RAGResponse(
            answer=response.content,
            citations=citations,
            context_used=[r.content for r in results],
            metadata={
                "retrieval_count": len(results),
                "model": response.model if hasattr(response, "model") else "unknown",
            },
        )

    async def generate_stream(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        流式生成回答
        
        Args:
            question: 用户问题
            history: 对话历史
            top_k: 检索数量
        
        Yields:
            回答片段
        """
        # 检索相关内容
        results, citations = self.retriever.search_with_citations(question, top_k)
        
        if not results:
            yield "抱歉，我在知识库中没有找到与您问题相关的信息。请尝试用其他方式描述您的问题。"
            return
        
        # 构建上下文
        context = self._build_context(results)
        sources = self.retriever.format_citations(citations)
        
        # 构建消息
        messages = self._build_messages(question, context, sources, history)
        
        # 流式调用 LLM
        async for chunk in self.llm_client.chat_stream_async(messages):
            if chunk.delta:
                yield chunk.delta

    def _build_context(self, results: List[SearchResult]) -> str:
        """构建上下文"""
        context_parts = []
        total_length = 0
        max_chars = self.max_context_tokens * 4
        
        for i, result in enumerate(results):
            if total_length + len(result.content) > max_chars:
                break
            
            context_parts.append(f"[{i+1}] {result.content}")
            total_length += len(result.content)
        
        return "\n\n".join(context_parts)

    def _build_messages(
        self,
        question: str,
        context: str,
        sources: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 添加历史消息
        if history:
            for msg in history[-6:]:  # 只保留最近 3 轮
                messages.append(msg)
        
        # 添加当前问题
        user_content = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            sources=sources,
            question=question,
        )
        messages.append({"role": "user", "content": user_content})
        
        return messages

    def get_citations_for_response(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> List[Citation]:
        """仅获取引用（不生成回答）"""
        _, citations = self.retriever.search_with_citations(question, top_k)
        return citations


