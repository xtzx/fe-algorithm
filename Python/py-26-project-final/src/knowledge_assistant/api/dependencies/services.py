"""
服务依赖

提供 RAG 相关服务的依赖注入
"""

from functools import lru_cache
from typing import Optional

from knowledge_assistant.config import get_settings
from knowledge_assistant.llm.client import LLMClient, create_llm_client
from knowledge_assistant.rag.embedder import BaseEmbedder, create_embedder
from knowledge_assistant.rag.index import VectorIndex
from knowledge_assistant.rag.retriever import Retriever
from knowledge_assistant.rag.generator import RAGGenerator


# 全局单例
_vector_index: Optional[VectorIndex] = None
_embedder: Optional[BaseEmbedder] = None
_llm_client: Optional[LLMClient] = None
_retriever: Optional[Retriever] = None
_generator: Optional[RAGGenerator] = None


def get_embedder() -> BaseEmbedder:
    """获取嵌入器"""
    global _embedder
    
    if _embedder is None:
        settings = get_settings()
        _embedder = create_embedder(
            provider=settings.embedding_provider,
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            base_url=settings.openai_base_url,
            dimension=settings.embedding_dimension,
        )
    
    return _embedder


def get_vector_index() -> VectorIndex:
    """获取向量索引"""
    global _vector_index
    
    if _vector_index is None:
        settings = get_settings()
        embedder = get_embedder()
        
        # 尝试加载现有索引
        _vector_index = VectorIndex.load(settings.index_dir, embedder)
    
    return _vector_index


def get_llm_client() -> LLMClient:
    """获取 LLM 客户端"""
    global _llm_client
    
    if _llm_client is None:
        settings = get_settings()
        _llm_client = create_llm_client(
            provider=settings.llm_provider,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
        )
    
    return _llm_client


def get_retriever() -> Retriever:
    """获取检索器"""
    global _retriever
    
    if _retriever is None:
        settings = get_settings()
        index = get_vector_index()
        _retriever = Retriever(
            index=index,
            top_k=settings.top_k,
            score_threshold=settings.score_threshold,
        )
    
    return _retriever


def get_rag_service() -> RAGGenerator:
    """获取 RAG 生成器"""
    global _generator
    
    if _generator is None:
        retriever = get_retriever()
        llm_client = get_llm_client()
        _generator = RAGGenerator(
            retriever=retriever,
            llm_client=llm_client,
        )
    
    return _generator


def reset_services():
    """重置所有服务（用于测试）"""
    global _vector_index, _embedder, _llm_client, _retriever, _generator
    _vector_index = None
    _embedder = None
    _llm_client = None
    _retriever = None
    _generator = None


def save_index():
    """保存索引到磁盘"""
    global _vector_index
    
    if _vector_index is not None:
        settings = get_settings()
        _vector_index.save(settings.index_dir)


