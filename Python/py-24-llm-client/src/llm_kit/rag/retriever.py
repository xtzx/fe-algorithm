"""
检索器

从向量索引中检索相关内容
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from llm_kit.rag.index import IndexEntry, VectorIndex

logger = structlog.get_logger()


@dataclass
class SearchResult:
    """搜索结果"""
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")


@dataclass
class Citation:
    """引用"""
    text: str
    source: str
    chunk_id: str
    page: Optional[int] = None
    line: Optional[int] = None


class Retriever:
    """
    检索器
    
    从向量索引中检索相关内容
    
    Usage:
        retriever = Retriever(index, top_k=5)
        
        # 简单检索
        results = retriever.search("What is RAG?")
        
        # 带引用的检索
        results, citations = retriever.search_with_citations("What is RAG?")
    """

    def __init__(
        self,
        index: VectorIndex,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        self.index = index
        self.top_k = top_k
        self.score_threshold = score_threshold

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        搜索相关内容
        
        Args:
            query: 查询文本
            top_k: 返回数量（覆盖默认值）
            filters: 元数据过滤器
        
        Returns:
            SearchResult 列表
        """
        k = top_k or self.top_k
        
        raw_results = self.index.search(
            query=query,
            top_k=k * 2 if filters else k,  # 过滤时多获取一些
            threshold=self.score_threshold,
        )
        
        results = []
        for entry, score in raw_results:
            # 应用过滤器
            if filters and not self._match_filters(entry.metadata, filters):
                continue
            
            results.append(SearchResult(
                content=entry.content,
                score=score,
                metadata=entry.metadata,
                chunk_id=entry.chunk_id,
            ))
            
            if len(results) >= k:
                break
        
        logger.debug(
            "retriever_search",
            query=query[:50],
            results=len(results),
        )
        
        return results

    def search_with_citations(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> tuple[List[SearchResult], List[Citation]]:
        """
        搜索并返回引用
        
        Args:
            query: 查询文本
            top_k: 返回数量
        
        Returns:
            (SearchResult 列表, Citation 列表)
        """
        results = self.search(query, top_k)
        
        citations = []
        for result in results:
            citations.append(Citation(
                text=result.content[:200],  # 截取摘要
                source=result.source,
                chunk_id=result.chunk_id or "",
                page=result.metadata.get("page"),
                line=result.metadata.get("line"),
            ))
        
        return results, citations

    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """检查元数据是否匹配过滤器"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        
        return True

    def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        获取上下文字符串（用于 LLM 提示）
        
        Args:
            query: 查询文本
            max_tokens: 最大 token 数（近似）
            separator: 分隔符
        
        Returns:
            上下文字符串
        """
        results = self.search(query)
        
        context_parts = []
        total_length = 0
        max_chars = max_tokens * 4  # 近似 token 到字符
        
        for result in results:
            if total_length + len(result.content) > max_chars:
                break
            
            context_parts.append(f"[Source: {result.source}]\n{result.content}")
            total_length += len(result.content)
        
        return separator.join(context_parts)

    def format_citations(self, citations: List[Citation]) -> str:
        """格式化引用"""
        lines = []
        for i, cite in enumerate(citations, 1):
            lines.append(f"[{i}] {cite.source}")
            if cite.page:
                lines[-1] += f", page {cite.page}"
        return "\n".join(lines)


class HybridRetriever:
    """
    混合检索器
    
    结合多个检索源
    
    Usage:
        retriever = HybridRetriever([
            (vector_retriever, 0.7),
            (keyword_retriever, 0.3),
        ])
        results = retriever.search("query")
    """

    def __init__(self, retrievers_with_weights: List[tuple[Retriever, float]]):
        self.retrievers_with_weights = retrievers_with_weights
        
        # 归一化权重
        total_weight = sum(w for _, w in retrievers_with_weights)
        self.retrievers_with_weights = [
            (r, w / total_weight) for r, w in retrievers_with_weights
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """混合搜索"""
        all_results: Dict[str, tuple[SearchResult, float]] = {}
        
        for retriever, weight in self.retrievers_with_weights:
            results = retriever.search(query, top_k=top_k * 2)
            
            for result in results:
                key = result.chunk_id or result.content[:50]
                
                if key in all_results:
                    existing, existing_score = all_results[key]
                    all_results[key] = (existing, existing_score + result.score * weight)
                else:
                    all_results[key] = (result, result.score * weight)
        
        # 按加权分数排序
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return [
            SearchResult(
                content=r.content,
                score=score,
                metadata=r.metadata,
                chunk_id=r.chunk_id,
            )
            for r, score in sorted_results[:top_k]
        ]


