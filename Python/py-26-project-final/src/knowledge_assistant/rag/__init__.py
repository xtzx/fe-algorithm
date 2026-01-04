"""
RAG 模块

提供:
- 文档加载 (loader)
- 智能分块 (chunker)
- 向量嵌入 (embedder)
- 向量索引 (index)
- 检索器 (retriever)
- 生成器 (generator)
"""

from knowledge_assistant.rag.loader import Document, DocumentLoader, PDFLoader
from knowledge_assistant.rag.chunker import Chunk, Chunker, ChunkingStrategy
from knowledge_assistant.rag.embedder import BaseEmbedder, Embedder, OpenAIEmbedder
from knowledge_assistant.rag.index import VectorIndex, IndexEntry
from knowledge_assistant.rag.retriever import Retriever, SearchResult, Citation
from knowledge_assistant.rag.generator import RAGGenerator, RAGResponse

__all__ = [
    # Loader
    "Document",
    "DocumentLoader",
    "PDFLoader",
    # Chunker
    "Chunk",
    "Chunker",
    "ChunkingStrategy",
    # Embedder
    "BaseEmbedder",
    "Embedder",
    "OpenAIEmbedder",
    # Index
    "VectorIndex",
    "IndexEntry",
    # Retriever
    "Retriever",
    "SearchResult",
    "Citation",
    # Generator
    "RAGGenerator",
    "RAGResponse",
]


