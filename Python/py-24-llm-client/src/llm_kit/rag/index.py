"""
向量索引

存储和检索向量
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from llm_kit.rag.chunker import Chunk
from llm_kit.rag.embedder import BaseEmbedder, Embedder

logger = structlog.get_logger()


@dataclass
class IndexEntry:
    """索引条目"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    vector: np.ndarray


class VectorIndex:
    """
    向量索引
    
    简单的内存向量存储，支持:
    - 添加文档
    - 余弦相似度搜索
    - 持久化和加载
    
    Usage:
        embedder = Embedder()
        index = VectorIndex(embedder)
        
        # 添加文档
        index.add_chunks(chunks)
        
        # 搜索
        results = index.search("query", top_k=5)
        
        # 持久化
        index.save("./index")
        index = VectorIndex.load("./index", embedder)
    """

    def __init__(self, embedder: Optional[BaseEmbedder] = None):
        self.embedder = embedder or Embedder()
        self._entries: List[IndexEntry] = []
        self._vectors: Optional[np.ndarray] = None
        self._dirty = False

    def add_chunk(self, chunk: Chunk):
        """添加单个块"""
        vector = self.embedder.embed(chunk.content)
        
        entry = IndexEntry(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            metadata=chunk.metadata,
            vector=vector,
        )
        
        self._entries.append(entry)
        self._dirty = True

    def add_chunks(self, chunks: List[Chunk]):
        """批量添加块"""
        if not chunks:
            return
        
        # 批量嵌入
        texts = [chunk.content for chunk in chunks]
        vectors = self.embedder.embed_batch(texts)
        
        for chunk, vector in zip(chunks, vectors):
            entry = IndexEntry(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                vector=vector,
            )
            self._entries.append(entry)
        
        self._dirty = True
        logger.info("chunks_added", count=len(chunks))

    def add_documents(self, chunks: List[Chunk]):
        """add_chunks 的别名"""
        return self.add_chunks(chunks)

    def _build_vectors_matrix(self):
        """构建向量矩阵（懒加载）"""
        if self._dirty or self._vectors is None:
            if self._entries:
                self._vectors = np.array([e.vector for e in self._entries])
            else:
                self._vectors = np.array([])
            self._dirty = False

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[IndexEntry, float]]:
        """
        搜索相似内容
        
        Args:
            query: 查询文本
            top_k: 返回数量
            threshold: 最低相似度阈值
        
        Returns:
            (条目, 相似度) 列表
        """
        if not self._entries:
            return []
        
        self._build_vectors_matrix()
        
        # 嵌入查询
        query_vector = self.embedder.embed(query)
        
        # 余弦相似度
        similarities = self._cosine_similarity(query_vector, self._vectors)
        
        # 排序
        indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in indices[:top_k]:
            score = float(similarities[idx])
            if score >= threshold:
                results.append((self._entries[idx], score))
        
        return results

    def search_by_vector(
        self,
        vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[IndexEntry, float]]:
        """使用向量搜索"""
        if not self._entries:
            return []
        
        self._build_vectors_matrix()
        similarities = self._cosine_similarity(vector, self._vectors)
        indices = np.argsort(similarities)[::-1]
        
        return [
            (self._entries[idx], float(similarities[idx]))
            for idx in indices[:top_k]
        ]

    def _cosine_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        # 归一化
        query_norm = query / np.linalg.norm(query)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # 点积
        return np.dot(vectors_norm, query_norm)

    def save(self, path: str | Path):
        """
        保存索引到目录
        
        Args:
            path: 目录路径
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存元数据
        metadata = {
            "count": len(self._entries),
            "dimension": self.embedder.dimension,
        }
        
        entries_data = []
        vectors_list = []
        
        for entry in self._entries:
            entries_data.append({
                "chunk_id": entry.chunk_id,
                "content": entry.content,
                "metadata": entry.metadata,
            })
            vectors_list.append(entry.vector)
        
        # 保存文件
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        with open(path / "entries.json", "w") as f:
            json.dump(entries_data, f)
        
        np.save(path / "vectors.npy", np.array(vectors_list))
        
        logger.info("index_saved", path=str(path), count=len(self._entries))

    @classmethod
    def load(cls, path: str | Path, embedder: Optional[BaseEmbedder] = None) -> "VectorIndex":
        """
        从目录加载索引
        
        Args:
            path: 目录路径
            embedder: 嵌入器（用于后续查询）
        
        Returns:
            VectorIndex
        """
        path = Path(path)
        
        with open(path / "entries.json") as f:
            entries_data = json.load(f)
        
        vectors = np.load(path / "vectors.npy")
        
        index = cls(embedder=embedder)
        
        for entry_data, vector in zip(entries_data, vectors):
            index._entries.append(IndexEntry(
                chunk_id=entry_data["chunk_id"],
                content=entry_data["content"],
                metadata=entry_data["metadata"],
                vector=vector,
            ))
        
        index._dirty = True
        
        logger.info("index_loaded", path=str(path), count=len(index._entries))
        return index

    def __len__(self) -> int:
        return len(self._entries)


