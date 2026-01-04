"""
文档分块器

将长文档分割为适合嵌入和检索的小块
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

from llm_kit.rag.loader import Document

logger = structlog.get_logger()


class ChunkingStrategy(str, Enum):
    """分块策略"""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


@dataclass
class Chunk:
    """文档块"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    
    # 位置信息
    start_idx: int = 0
    end_idx: int = 0

    def __post_init__(self):
        if not self.chunk_id:
            doc_id = self.metadata.get("doc_id", "unknown")
            self.chunk_id = f"{doc_id}_chunk_{self.start_idx}"

    def __len__(self) -> int:
        return len(self.content)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")


class Chunker:
    """
    文档分块器
    
    支持多种分块策略:
    - 固定大小（按字符数）
    - 句子分割
    - 段落分割
    - 语义分割（需要 tiktoken）
    
    Usage:
        chunker = Chunker(chunk_size=500, overlap=50)
        
        # 分块单个文档
        chunks = chunker.split(document)
        
        # 分块多个文档
        all_chunks = chunker.split_documents(documents)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        length_function: Optional[Callable[[str], int]] = None,
    ):
        """
        Args:
            chunk_size: 块大小
            overlap: 重叠大小
            strategy: 分块策略
            length_function: 长度计算函数（默认 len，可用 tiktoken）
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self.length_function = length_function or len
        
        # 加载 tiktoken（如果可用）
        self._tiktoken = None
        try:
            import tiktoken
            self._tiktoken = tiktoken
        except ImportError:
            pass

    def split(self, document: Document) -> List[Chunk]:
        """
        分割文档
        
        Args:
            document: 文档
        
        Returns:
            块列表
        """
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._split_fixed_size(document)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._split_by_sentence(document)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._split_by_paragraph(document)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._split_semantic(document)
        else:
            return self._split_fixed_size(document)

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """分割多个文档"""
        all_chunks = []
        for doc in documents:
            chunks = self.split(doc)
            all_chunks.extend(chunks)
        
        logger.info("documents_chunked", doc_count=len(documents), chunk_count=len(all_chunks))
        return all_chunks

    def _split_fixed_size(self, document: Document) -> List[Chunk]:
        """固定大小分块"""
        text = document.content
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 如果不是最后一块，尝试在词边界分割
            if end < len(text):
                # 向前找空格
                space_idx = text.rfind(" ", start, end)
                if space_idx > start:
                    end = space_idx
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "doc_id": document.doc_id,
                    },
                    start_idx=start,
                    end_idx=end,
                ))
            
            start = end - self.overlap
            if start <= chunks[-1].start_idx if chunks else 0:
                start = end
        
        return chunks

    def _split_by_sentence(self, document: Document) -> List[Chunk]:
        """按句子分块"""
        # 简单的句子分割正则
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, document.content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_idx = 0
        
        for sentence in sentences:
            sentence_length = self.length_function(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # 保存当前块
                chunk_text = " ".join(current_chunk)
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "doc_id": document.doc_id,
                    },
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                ))
                
                # 处理重叠
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + self.length_function(s) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += self.length_function(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
                start_idx = start_idx + len(chunk_text) - overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # 处理最后一块
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                content=chunk_text,
                metadata={
                    **document.metadata,
                    "doc_id": document.doc_id,
                },
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text),
            ))
        
        return chunks

    def _split_by_paragraph(self, document: Document) -> List[Chunk]:
        """按段落分块"""
        paragraphs = re.split(r'\n\s*\n', document.content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = self.length_function(para)
            
            if current_length + para_length > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "doc_id": document.doc_id,
                    },
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                ))
                
                current_chunk = []
                current_length = 0
                start_idx = start_idx + len(chunk_text)
            
            current_chunk.append(para)
            current_length += para_length
        
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(Chunk(
                content=chunk_text,
                metadata={
                    **document.metadata,
                    "doc_id": document.doc_id,
                },
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text),
            ))
        
        return chunks

    def _split_semantic(self, document: Document) -> List[Chunk]:
        """语义分块（使用 tiktoken）"""
        if self._tiktoken is None:
            logger.warning("tiktoken_not_available, falling back to fixed_size")
            return self._split_fixed_size(document)
        
        # 使用 cl100k_base 编码（GPT-4 使用）
        encoding = self._tiktoken.get_encoding("cl100k_base")
        
        def token_length(text: str) -> int:
            return len(encoding.encode(text))
        
        # 使用 token 计数的固定大小分块
        old_length_function = self.length_function
        self.length_function = token_length
        chunks = self._split_by_sentence(document)
        self.length_function = old_length_function
        
        return chunks


def create_length_function_tiktoken(model: str = "gpt-4"):
    """
    创建基于 tiktoken 的长度函数
    
    Args:
        model: 模型名称
    
    Returns:
        长度计算函数
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return lambda text: len(encoding.encode(text))
    except ImportError:
        return len


