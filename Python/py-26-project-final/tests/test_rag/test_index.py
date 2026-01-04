"""
向量索引测试
"""

import tempfile
from pathlib import Path

import pytest

from knowledge_assistant.rag.chunker import Chunk
from knowledge_assistant.rag.embedder import Embedder
from knowledge_assistant.rag.index import VectorIndex


@pytest.fixture
def embedder():
    """创建嵌入器"""
    return Embedder(dimension=64)


@pytest.fixture
def index(embedder):
    """创建索引"""
    return VectorIndex(embedder)


@pytest.fixture
def sample_chunks():
    """示例块"""
    return [
        Chunk(
            content="Python is a programming language.",
            metadata={"doc_id": "doc1", "source": "python.txt"},
            chunk_index=0,
        ),
        Chunk(
            content="JavaScript is used for web development.",
            metadata={"doc_id": "doc2", "source": "js.txt"},
            chunk_index=0,
        ),
        Chunk(
            content="Machine learning is a subset of AI.",
            metadata={"doc_id": "doc3", "source": "ml.txt"},
            chunk_index=0,
        ),
    ]


def test_add_chunk(index, sample_chunks):
    """测试添加块"""
    index.add_chunk(sample_chunks[0])
    
    assert len(index) == 1


def test_add_chunks(index, sample_chunks):
    """测试批量添加块"""
    index.add_chunks(sample_chunks)
    
    assert len(index) == 3


def test_search(index, sample_chunks):
    """测试搜索"""
    index.add_chunks(sample_chunks)
    
    results = index.search("Python programming", top_k=2)
    
    assert len(results) > 0
    entry, score = results[0]
    assert "Python" in entry.content or score > 0


def test_search_empty_index(index):
    """测试空索引搜索"""
    results = index.search("test query")
    
    assert len(results) == 0


def test_get_by_id(index, sample_chunks):
    """测试按 ID 获取"""
    index.add_chunks(sample_chunks)
    
    chunk_id = sample_chunks[0].chunk_id
    entry = index.get_by_id(chunk_id)
    
    assert entry is not None
    assert entry.content == sample_chunks[0].content


def test_delete_by_id(index, sample_chunks):
    """测试按 ID 删除"""
    index.add_chunks(sample_chunks)
    chunk_id = sample_chunks[0].chunk_id
    
    index.delete_by_id(chunk_id)
    
    assert len(index) == 2
    assert index.get_by_id(chunk_id) is None


def test_delete_by_doc_id(index, sample_chunks):
    """测试按文档 ID 删除"""
    index.add_chunks(sample_chunks)
    
    index.delete_by_doc_id("doc1")
    
    assert len(index) == 2


def test_save_and_load(index, sample_chunks):
    """测试保存和加载"""
    index.add_chunks(sample_chunks)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "index"
        
        # 保存
        index.save(save_path)
        
        # 加载
        loaded_index = VectorIndex.load(save_path, index.embedder)
        
        assert len(loaded_index) == len(index)
        
        # 搜索应该正常工作
        results = loaded_index.search("Python", top_k=1)
        assert len(results) > 0


def test_clear(index, sample_chunks):
    """测试清空"""
    index.add_chunks(sample_chunks)
    index.clear()
    
    assert len(index) == 0


