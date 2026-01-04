"""
向量索引

存储和检索向量
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from knowledge_assistant.rag.chunker import Chunk
from knowledge_assistant.rag.embedder import BaseEmbedder, Embedder

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
        self._id_to_index: Dict[str, int] = {}

    def add_chunk(self, chunk: Chunk):
        """添加单个块"""
        # 检查是否已存在
        if chunk.chunk_id in self._id_to_index:
            logger.debug("chunk_already_exists", chunk_id=chunk.chunk_id)
            return
        
        vector = self.embedder.embed(chunk.content)
        
        entry = IndexEntry(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            metadata=chunk.metadata,
            vector=vector,
        )
        
        self._id_to_index[chunk.chunk_id] = len(self._entries)
        self._entries.append(entry)
        self._dirty = True

    def add_chunks(self, chunks: List[Chunk]):
        """批量添加块"""
        if not chunks:
            return
        
        # 过滤已存在的
        new_chunks = [c for c in chunks if c.chunk_id not in self._id_to_index]
        
        if not new_chunks:
            return
        
        # 批量嵌入
        texts = [chunk.content for chunk in new_chunks]
        vectors = self.embedder.embed_batch(texts)
        
        for chunk, vector in zip(new_chunks, vectors):
            entry = IndexEntry(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                vector=vector,
            )
            self._id_to_index[chunk.chunk_id] = len(self._entries)
            self._entries.append(entry)
        
        self._dirty = True
        logger.info("chunks_added", count=len(new_chunks))

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
        if len(vectors) == 0:
            return np.array([])
        
        # 归一化
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        
        # 点积
        return np.dot(vectors_norm, query_norm)

    def get_by_id(self, chunk_id: str) -> Optional[IndexEntry]:
        """根据 ID 获取条目"""
        idx = self._id_to_index.get(chunk_id)
        if idx is not None:
            return self._entries[idx]
        return None

    def delete_by_doc_id(self, doc_id: str):
        """删除指定文档的所有块"""
        to_remove = [
            entry.chunk_id for entry in self._entries
            if entry.metadata.get("doc_id") == doc_id
        ]
        
        for chunk_id in to_remove:
            self.delete_by_id(chunk_id)

    def delete_by_id(self, chunk_id: str):
        """删除指定块"""
        if chunk_id not in self._id_to_index:
            return
        
        idx = self._id_to_index.pop(chunk_id)
        self._entries.pop(idx)
        
        # 重建索引映射
        self._id_to_index = {
            entry.chunk_id: i for i, entry in enumerate(self._entries)
        }
        self._dirty = True

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
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        with open(path / "entries.json", "w", encoding="utf-8") as f:
            json.dump(entries_data, f, ensure_ascii=False, indent=2)
        
        if vectors_list:
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
        
        if not (path / "entries.json").exists():
            logger.warning("index_not_found", path=str(path))
            return cls(embedder=embedder)
        
        with open(path / "entries.json", encoding="utf-8") as f:
            entries_data = json.load(f)
        
        vectors_path = path / "vectors.npy"
        if vectors_path.exists():
            vectors = np.load(vectors_path)
        else:
            vectors = []
        
        index = cls(embedder=embedder)
        
        for entry_data, vector in zip(entries_data, vectors):
            index._entries.append(IndexEntry(
                chunk_id=entry_data["chunk_id"],
                content=entry_data["content"],
                metadata=entry_data["metadata"],
                vector=vector,
            ))
            index._id_to_index[entry_data["chunk_id"]] = len(index._entries) - 1
        
        index._dirty = True
        
        logger.info("index_loaded", path=str(path), count=len(index._entries))
        return index

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self):
        """清空索引"""
        self._entries.clear()
        self._id_to_index.clear()
        self._vectors = None
        self._dirty = False


