"""
文档加载器

支持多种文档格式的加载
"""

import hashlib
import os
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
        return f"{source}_{content_hash}"

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    def __len__(self) -> int:
        return len(self.content)


class DocumentLoader:
    """
    文档加载器
    
    支持:
    - 文本文件 (.txt)
    - Markdown (.md)
    - JSON (.json)
    - 目录批量加载
    
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

    def load_texts(self, texts: List[str], source_prefix: str = "text") -> List[Document]:
        """批量从文本创建文档"""
        return [
            self.load_text(text, source=f"{source_prefix}_{i}")
            for i, text in enumerate(texts)
        ]


class WebLoader:
    """
    网页加载器（简化版）
    
    Usage:
        loader = WebLoader()
        doc = loader.load_url("https://example.com")
    """

    def __init__(self):
        try:
            import httpx
            from bs4 import BeautifulSoup
            self._httpx = httpx
            self._bs4 = BeautifulSoup
        except ImportError:
            raise ImportError("WebLoader requires httpx and beautifulsoup4")

    def load_url(self, url: str) -> Document:
        """
        加载网页
        
        Args:
            url: 网页 URL
        
        Returns:
            Document
        """
        response = self._httpx.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        
        soup = self._bs4(response.text, "html.parser")
        
        # 移除 script 和 style
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        
        # 提取文本
        text = soup.get_text(separator="\n", strip=True)
        
        # 提取标题
        title = soup.title.string if soup.title else url
        
        return Document(
            content=text,
            metadata={
                "source": url,
                "title": title,
                "type": "web",
            },
        )


