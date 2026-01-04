"""
文档加载器测试
"""

import tempfile
from pathlib import Path

import pytest

from knowledge_assistant.rag.loader import Document, DocumentLoader


def test_load_text():
    """测试从文本创建文档"""
    loader = DocumentLoader()
    doc = loader.load_text("Hello, world!", source="test")
    
    assert doc.content == "Hello, world!"
    assert doc.source == "test"
    assert doc.doc_id is not None


def test_load_file():
    """测试加载文件"""
    loader = DocumentLoader()
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test content")
        f.flush()
        
        doc = loader.load_file(f.name)
        
        assert doc.content == "Test content"
        assert doc.metadata["extension"] == ".txt"


def test_load_markdown_file():
    """测试加载 Markdown 文件"""
    loader = DocumentLoader()
    
    content = """# Title

This is a test.

## Section

More content here.
"""
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        f.flush()
        
        doc = loader.load_file(f.name)
        
        assert "# Title" in doc.content
        assert doc.metadata["extension"] == ".md"


def test_load_directory():
    """测试加载目录"""
    loader = DocumentLoader()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试文件
        (Path(tmpdir) / "test1.txt").write_text("Content 1")
        (Path(tmpdir) / "test2.md").write_text("Content 2")
        
        docs = loader.load_directory(tmpdir)
        
        assert len(docs) == 2


def test_load_unsupported_file():
    """测试加载不支持的文件类型"""
    loader = DocumentLoader()
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write("test")
        f.flush()
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_file(f.name)


def test_document_id_generation():
    """测试文档 ID 生成"""
    doc1 = Document(content="Hello", metadata={"source": "test"})
    doc2 = Document(content="Hello", metadata={"source": "test"})
    doc3 = Document(content="World", metadata={"source": "test"})
    
    # 相同内容应该生成相同的 ID
    assert doc1.doc_id == doc2.doc_id
    # 不同内容应该生成不同的 ID
    assert doc1.doc_id != doc3.doc_id


