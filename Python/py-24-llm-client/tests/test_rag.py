"""
RAG 模块测试
"""

import pytest
import tempfile
from pathlib import Path

from llm_kit.rag import Chunker, DocumentLoader, Embedder, VectorIndex, Retriever
from llm_kit.rag.loader import Document
from llm_kit.rag.chunker import ChunkingStrategy


class TestDocumentLoader:
    """文档加载器测试"""

    def test_load_text(self):
        """测试从文本创建文档"""
        loader = DocumentLoader()
        doc = loader.load_text("Hello world", source="test")
        
        assert doc.content == "Hello world"
        assert doc.metadata["source"] == "test"

    def test_load_texts(self):
        """测试批量创建文档"""
        loader = DocumentLoader()
        docs = loader.load_texts(["Hello", "World"], source_prefix="doc")
        
        assert len(docs) == 2
        assert docs[0].content == "Hello"
        assert docs[1].content == "World"

    def test_load_file(self, tmp_path):
        """测试加载文件"""
        # 创建临时文件
        file_path = tmp_path / "test.txt"
        file_path.write_text("Test content")
        
        loader = DocumentLoader()
        doc = loader.load_file(file_path)
        
        assert doc.content == "Test content"
        assert "test.txt" in doc.metadata["source"]

    def test_load_directory(self, tmp_path):
        """测试加载目录"""
        # 创建临时文件
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.md").write_text("Content 2")
        
        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path)
        
        assert len(docs) == 2


class TestChunker:
    """分块器测试"""

    def test_fixed_size_chunking(self, sample_document):
        """测试固定大小分块"""
        chunker = Chunker(chunk_size=50, overlap=10)
        chunks = chunker.split(sample_document)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 60  # 允许一些余量

    def test_sentence_chunking(self):
        """测试句子分块"""
        doc = Document(
            content="First sentence. Second sentence. Third sentence. Fourth sentence.",
            metadata={"source": "test"},
        )
        
        chunker = Chunker(
            chunk_size=30,
            strategy=ChunkingStrategy.SENTENCE,
        )
        chunks = chunker.split(doc)
        
        assert len(chunks) >= 1

    def test_chunk_metadata(self, sample_document):
        """测试块元数据"""
        chunker = Chunker(chunk_size=50)
        chunks = chunker.split(sample_document)
        
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.chunk_id is not None

    def test_split_documents(self, sample_documents, chunker):
        """测试批量分块"""
        chunks = chunker.split_documents(sample_documents)
        
        assert len(chunks) >= len(sample_documents)


class TestEmbedder:
    """嵌入器测试"""

    def test_embed_single(self, embedder):
        """测试单个文本嵌入"""
        vector = embedder.embed("Hello world")
        
        assert vector.shape == (384,)
        assert abs(vector.sum()) > 0  # 非零向量

    def test_embed_batch(self, embedder):
        """测试批量嵌入"""
        vectors = embedder.embed_batch(["Hello", "World"])
        
        assert vectors.shape == (2, 384)

    def test_embed_consistency(self, embedder):
        """测试嵌入一致性"""
        v1 = embedder.embed("Hello")
        v2 = embedder.embed("Hello")
        
        assert (v1 == v2).all()


class TestVectorIndex:
    """向量索引测试"""

    def test_add_and_search(self, vector_index, sample_documents, chunker):
        """测试添加和搜索"""
        chunks = chunker.split_documents(sample_documents)
        vector_index.add_chunks(chunks)
        
        results = vector_index.search("What is Python?", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(score, float) for _, score in results)

    def test_empty_search(self, vector_index):
        """测试空索引搜索"""
        results = vector_index.search("test", top_k=5)
        assert len(results) == 0

    def test_save_and_load(self, vector_index, sample_documents, chunker, tmp_path, embedder):
        """测试保存和加载"""
        chunks = chunker.split_documents(sample_documents)
        vector_index.add_chunks(chunks)
        
        # 保存
        save_path = tmp_path / "index"
        vector_index.save(save_path)
        
        # 加载
        loaded_index = VectorIndex.load(save_path, embedder)
        
        assert len(loaded_index) == len(vector_index)


class TestRetriever:
    """检索器测试"""

    def test_search(self, vector_index, sample_documents, chunker):
        """测试检索"""
        chunks = chunker.split_documents(sample_documents)
        vector_index.add_chunks(chunks)
        
        retriever = Retriever(vector_index, top_k=2)
        results = retriever.search("Python programming")
        
        assert len(results) <= 2
        for result in results:
            assert hasattr(result, "content")
            assert hasattr(result, "score")

    def test_search_with_citations(self, vector_index, sample_documents, chunker):
        """测试带引用的检索"""
        chunks = chunker.split_documents(sample_documents)
        vector_index.add_chunks(chunks)
        
        retriever = Retriever(vector_index, top_k=2)
        results, citations = retriever.search_with_citations("RAG")
        
        assert len(citations) == len(results)
        for citation in citations:
            assert hasattr(citation, "source")

    def test_get_context(self, vector_index, sample_documents, chunker):
        """测试获取上下文"""
        chunks = chunker.split_documents(sample_documents)
        vector_index.add_chunks(chunks)
        
        retriever = Retriever(vector_index, top_k=2)
        context = retriever.get_context("machine learning")
        
        assert isinstance(context, str)
        assert len(context) > 0


