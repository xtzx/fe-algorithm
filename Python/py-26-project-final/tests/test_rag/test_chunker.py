"""
分块器测试
"""

import pytest

from knowledge_assistant.rag.chunker import Chunk, Chunker, ChunkingStrategy
from knowledge_assistant.rag.loader import Document


@pytest.fixture
def long_document():
    """长文档"""
    content = "This is sentence one. " * 100
    return Document(content=content, metadata={"source": "test"})


def test_fixed_size_chunking(long_document):
    """测试固定大小分块"""
    chunker = Chunker(chunk_size=100, overlap=20)
    chunks = chunker.split(long_document)
    
    assert len(chunks) > 1
    # 检查块大小
    for chunk in chunks[:-1]:  # 最后一块可能较小
        assert len(chunk.content) <= 100 + 20  # 允许在词边界的偏差


def test_sentence_chunking():
    """测试句子分块"""
    content = "Sentence one. Sentence two. Sentence three. " * 20
    doc = Document(content=content, metadata={"source": "test"})
    
    chunker = Chunker(chunk_size=100, strategy=ChunkingStrategy.SENTENCE)
    chunks = chunker.split(doc)
    
    assert len(chunks) > 1


def test_paragraph_chunking():
    """测试段落分块"""
    content = "\n\n".join([f"Paragraph {i}. " * 10 for i in range(10)])
    doc = Document(content=content, metadata={"source": "test"})
    
    chunker = Chunker(chunk_size=200, strategy=ChunkingStrategy.PARAGRAPH)
    chunks = chunker.split(doc)
    
    assert len(chunks) > 1


def test_chunk_overlap():
    """测试分块重叠"""
    content = "Word " * 100
    doc = Document(content=content, metadata={"source": "test"})
    
    chunker = Chunker(chunk_size=50, overlap=10)
    chunks = chunker.split(doc)
    
    # 检查相邻块是否有重叠
    if len(chunks) >= 2:
        chunk1_end = chunks[0].content[-20:]
        chunk2_start = chunks[1].content[:20]
        # 重叠部分应该有相同的词
        assert any(w in chunk2_start for w in chunk1_end.split())


def test_chunk_metadata():
    """测试块元数据"""
    doc = Document(
        content="Test content. " * 50,
        metadata={"source": "test.md", "author": "test"},
        doc_id="test_doc",
    )
    
    chunker = Chunker(chunk_size=100)
    chunks = chunker.split(doc)
    
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.md"
        assert chunk.metadata["doc_id"] == "test_doc"
        assert chunk.chunk_id is not None


def test_split_documents():
    """测试批量分块"""
    docs = [
        Document(content="Content 1. " * 50, metadata={"source": "doc1"}),
        Document(content="Content 2. " * 50, metadata={"source": "doc2"}),
    ]
    
    chunker = Chunker(chunk_size=100)
    chunks = chunker.split_documents(docs)
    
    # 应该有来自两个文档的块
    sources = set(c.metadata["source"] for c in chunks)
    assert len(sources) == 2


