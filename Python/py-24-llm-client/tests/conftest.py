"""
测试配置
"""

import pytest

from llm_kit.rag import Chunker, DocumentLoader, Embedder, VectorIndex
from llm_kit.rag.loader import Document


@pytest.fixture
def sample_document():
    """示例文档"""
    return Document(
        content="This is a sample document. It contains multiple sentences. "
                "The document is used for testing purposes. "
                "It should be long enough to be split into chunks.",
        metadata={"source": "test.txt"},
    )


@pytest.fixture
def sample_documents():
    """示例文档列表"""
    return [
        Document(
            content="Python is a programming language. It is popular for data science.",
            metadata={"source": "python.txt"},
        ),
        Document(
            content="RAG stands for Retrieval-Augmented Generation. It combines retrieval and generation.",
            metadata={"source": "rag.txt"},
        ),
        Document(
            content="Machine learning is a subset of AI. It learns from data.",
            metadata={"source": "ml.txt"},
        ),
    ]


@pytest.fixture
def embedder():
    """嵌入器"""
    return Embedder(dimension=384)


@pytest.fixture
def chunker():
    """分块器"""
    return Chunker(chunk_size=100, overlap=20)


@pytest.fixture
def vector_index(embedder):
    """向量索引"""
    return VectorIndex(embedder)


