"""
文档加载器

支持多种文档格式:
- 文本文件 (.txt)
- Markdown (.md)
- PDF (.pdf)
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class Document:
    """文档数据类"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = self._generate_id()

    def _generate_id(self) -> str:
        """生成文档 ID"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]
        source = self.metadata.get("source", "unknown")
        return f"{Path(source).stem}_{content_hash}"

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def filename(self) -> str:
        return self.metadata.get("filename", "unknown")

    def __len__(self) -> int:
        return len(self.content)


class DocumentLoader:
    """
    文档加载器
    
    支持:
    - 文本文件 (.txt)
    - Markdown (.md, .markdown)
    - JSON (.json)
    
    Usage:
        loader = DocumentLoader()
        
        # 加载单个文件
        doc = loader.load_file("doc.txt")
        
        # 加载目录
        docs = loader.load_directory("./docs", pattern="*.md")
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".json"}

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def load_file(self, path: str | Path) -> Document:
        """
        加载单个文件
        
        Args:
            path: 文件路径
        
        Returns:
            Document
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        content = path.read_text(encoding=self.encoding)
        
        metadata = {
            "source": str(path),
            "filename": path.name,
            "extension": path.suffix,
            "size": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "type": "text",
        }
        
        logger.debug("document_loaded", path=str(path), size=len(content))
        
        return Document(content=content, metadata=metadata)

    def load_directory(
        self,
        path: str | Path,
        pattern: str = "*",
        recursive: bool = True,
    ) -> List[Document]:
        """
        加载目录中的文件
        
        Args:
            path: 目录路径
            pattern: 文件模式 (如 "*.md")
            recursive: 是否递归
        
        Returns:
            文档列表
        """
        path = Path(path)
        
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        
        glob_method = path.rglob if recursive else path.glob
        
        docs = []
        for file_path in glob_method(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.load_file(file_path)
                    docs.append(doc)
                except Exception as e:
                    logger.warning("document_load_failed", path=str(file_path), error=str(e))
        
        logger.info("directory_loaded", path=str(path), count=len(docs))
        return docs

    def load_text(self, text: str, source: str = "text") -> Document:
        """
        从文本创建文档
        
        Args:
            text: 文本内容
            source: 来源标识
        
        Returns:
            Document
        """
        return Document(
            content=text,
            metadata={"source": source, "type": "text"},
        )


class PDFLoader:
    """
    PDF 文档加载器
    
    使用 pypdf 提取 PDF 文本内容
    
    Usage:
        loader = PDFLoader()
        doc = loader.load_file("document.pdf")
    """

    def __init__(self):
        try:
            from pypdf import PdfReader
            self._PdfReader = PdfReader
        except ImportError:
            raise ImportError("PDFLoader requires pypdf: pip install pypdf")

    def load_file(self, path: str | Path) -> Document:
        """
        加载 PDF 文件
        
        Args:
            path: 文件路径
        
        Returns:
            Document
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        
        reader = self._PdfReader(path)
        
        # 提取文本
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(f"[Page {i + 1}]\n{text}")
        
        content = "\n\n".join(pages_text)
        
        # 提取元数据
        pdf_metadata = reader.metadata or {}
        
        metadata = {
            "source": str(path),
            "filename": path.name,
            "extension": ".pdf",
            "size": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "type": "pdf",
            "pages": len(reader.pages),
            "title": pdf_metadata.get("/Title", ""),
            "author": pdf_metadata.get("/Author", ""),
        }
        
        logger.debug("pdf_loaded", path=str(path), pages=len(reader.pages))
        
        return Document(content=content, metadata=metadata)


class MultiFormatLoader:
    """
    多格式文档加载器
    
    自动根据文件类型选择合适的加载器
    
    Usage:
        loader = MultiFormatLoader()
        doc = loader.load("document.pdf")  # 自动使用 PDFLoader
        doc = loader.load("readme.md")     # 自动使用 DocumentLoader
    """

    def __init__(self):
        self._text_loader = DocumentLoader()
        self._pdf_loader: Optional[PDFLoader] = None
        
        try:
            self._pdf_loader = PDFLoader()
        except ImportError:
            logger.warning("pdf_loader_unavailable", reason="pypdf not installed")

    def load(self, path: str | Path) -> Document:
        """
        加载文档（自动选择加载器）
        
        Args:
            path: 文件路径
        
        Returns:
            Document
        """
        path = Path(path)
        
        if path.suffix.lower() == ".pdf":
            if self._pdf_loader is None:
                raise ImportError("PDF support requires pypdf: pip install pypdf")
            return self._pdf_loader.load_file(path)
        
        return self._text_loader.load_file(path)

    def load_directory(
        self,
        path: str | Path,
        recursive: bool = True,
    ) -> List[Document]:
        """
        加载目录中的所有支持的文件
        
        Args:
            path: 目录路径
            recursive: 是否递归
        
        Returns:
            文档列表
        """
        path = Path(path)
        
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        
        supported_extensions = {".txt", ".md", ".markdown", ".json"}
        if self._pdf_loader:
            supported_extensions.add(".pdf")
        
        glob_method = path.rglob if recursive else path.glob
        
        docs = []
        for file_path in glob_method("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    doc = self.load(file_path)
                    docs.append(doc)
                except Exception as e:
                    logger.warning("document_load_failed", path=str(file_path), error=str(e))
        
        logger.info("directory_loaded", path=str(path), count=len(docs))
        return docs


