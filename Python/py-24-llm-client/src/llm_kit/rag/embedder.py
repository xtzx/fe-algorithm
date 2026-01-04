"""
向量嵌入

将文本转换为向量表示
"""

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import structlog

from llm_kit.rag.chunker import Chunk

logger = structlog.get_logger()


class BaseEmbedder(ABC):
    """嵌入器基类"""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """嵌入单个文本"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        pass


class Embedder(BaseEmbedder):
    """
    向量嵌入器（Stub 实现）
    
    实际使用时应替换为真实的嵌入模型:
    - OpenAI Embeddings
    - Sentence Transformers
    - Cohere Embeddings
    
    这个 Stub 实现使用简单的哈希方法生成伪向量，
    仅用于测试和演示。
    
    Usage:
        # Stub 模式
        embedder = Embedder(dimension=384)
        vector = embedder.embed("Hello world")
        
        # 使用 OpenAI
        embedder = OpenAIEmbedder(api_key="...")
        vector = embedder.embed("Hello world")
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._cache: Dict[str, np.ndarray] = {}
        logger.info("embedder_initialized", dimension=dimension, mode="stub")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """
        嵌入文本（Stub 实现）
        
        使用文本哈希生成伪向量
        """
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 使用哈希生成伪随机向量
        np.random.seed(int(cache_key[:8], 16))
        vector = np.random.randn(self._dimension).astype(np.float32)
        vector = vector / np.linalg.norm(vector)  # 归一化
        
        self._cache[cache_key] = vector
        return vector

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入"""
        vectors = [self.embed(text) for text in texts]
        return np.array(vectors)

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """嵌入文档块"""
        texts = [chunk.content for chunk in chunks]
        return self.embed_batch(texts)


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI 嵌入器
    
    使用 OpenAI 的 text-embedding 模型
    
    Usage:
        embedder = OpenAIEmbedder(api_key="sk-...")
        vector = embedder.embed("Hello world")
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
    ):
        import httpx
        
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1536)
        
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        
        logger.info("openai_embedder_initialized", model=model)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """嵌入单个文本"""
        result = self.embed_batch([text])
        return result[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入"""
        response = self._client.post(
            "/embeddings",
            json={
                "model": self.model,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        vectors = [item["embedding"] for item in data["data"]]
        return np.array(vectors, dtype=np.float32)

    def close(self):
        self._client.close()


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Sentence Transformers 嵌入器
    
    使用本地模型，无需 API
    
    Usage:
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        vector = embedder.embed("Hello world")
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info("sentence_transformer_initialized", model=model_name)
        except ImportError:
            raise ImportError("SentenceTransformerEmbedder requires sentence-transformers")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True)


