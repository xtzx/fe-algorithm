"""
文档分块器

将长文档分割为适合嵌入和检索的小块
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

from knowledge_assistant.rag.loader import Document

logger = structlog.get_logger()


class ChunkingStrategy(str, Enum):
    """分块策略"""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    MARKDOWN = "markdown"


@dataclass
class Chunk:
    """文档块"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    
    # 位置信息
    start_idx: int = 0
    end_idx: int = 0
    chunk_index: int = 0

    def __post_init__(self):
        if not self.chunk_id:
            doc_id = self.metadata.get("doc_id", "unknown")
            self.chunk_id = f"{doc_id}_chunk_{self.chunk_index}"

    def __len__(self) -> int:
        return len(self.content)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def doc_id(self) -> str:
        return self.metadata.get("doc_id", "unknown")


class Chunker:
    """
    文档分块器
    
    支持多种分块策略:
    - 固定大小（按字符数）
    - 句子分割
    - 段落分割
    - Markdown 结构分割
    
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
        elif self.strategy == ChunkingStrategy.MARKDOWN:
            return self._split_by_markdown(document)
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
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 如果不是最后一块，尝试在词边界分割
            if end < len(text):
                # 向前找空格或换行
                boundary_idx = text.rfind(" ", start, end)
                newline_idx = text.rfind("\n", start, end)
                split_idx = max(boundary_idx, newline_idx)
                if split_idx > start:
                    end = split_idx
            
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
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
            
            # 计算下一个起始位置（考虑重叠）
            start = end - self.overlap
            if chunks and start <= chunks[-1].start_idx:
                start = end
        
        return chunks

    def _split_by_sentence(self, document: Document) -> List[Chunk]:
        """按句子分块"""
        # 句子分割正则（支持中英文）
        sentence_pattern = r'(?<=[.!?。！？])\s+'
        sentences = re.split(sentence_pattern, document.content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_idx = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
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
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
                
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
                chunk_index=chunk_index,
            ))
        
        return chunks

    def _split_by_paragraph(self, document: Document) -> List[Chunk]:
        """按段落分块"""
        paragraphs = re.split(r'\n\s*\n', document.content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_idx = 0
        chunk_index = 0
        
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
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
                
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
                chunk_index=chunk_index,
            ))
        
        return chunks

    def _split_by_markdown(self, document: Document) -> List[Chunk]:
        """按 Markdown 结构分块"""
        # 按标题分割
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = document.content.split('\n')
        
        sections = []
        current_section: List[str] = []
        current_header = ""
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            if header_match:
                # 保存之前的 section
                if current_section:
                    sections.append((current_header, '\n'.join(current_section)))
                
                current_header = line
                current_section = [line]
            else:
                current_section.append(line)
        
        # 最后一个 section
        if current_section:
            sections.append((current_header, '\n'.join(current_section)))
        
        # 将 sections 转换为 chunks，如果太长则进一步分割
        chunks = []
        chunk_index = 0
        
        for header, content in sections:
            if self.length_function(content) <= self.chunk_size:
                chunks.append(Chunk(
                    content=content,
                    metadata={
                        **document.metadata,
                        "doc_id": document.doc_id,
                        "section_header": header,
                    },
                    start_idx=0,
                    end_idx=len(content),
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
            else:
                # 使用段落分割进一步处理
                temp_doc = Document(content=content, metadata=document.metadata, doc_id=document.doc_id)
                sub_chunks = self._split_by_paragraph(temp_doc)
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["section_header"] = header
                    sub_chunk.chunk_index = chunk_index
                    sub_chunk.chunk_id = f"{document.doc_id}_chunk_{chunk_index}"
                    chunks.append(sub_chunk)
                    chunk_index += 1
        
        return chunks


