# 📚 RAG 基础

> 检索增强生成（Retrieval-Augmented Generation）

---

## 什么是 RAG

```
RAG 通过检索外部知识来增强大模型的生成能力：

用户问题 → 检索相关文档 → 构造上下文 → LLM 生成答案

优势：
1. 知识可更新：无需重新训练模型
2. 减少幻觉：基于真实文档回答
3. 可追溯：答案有来源出处
4. 成本低：比微调更经济
```

---

## RAG 核心流程

```
┌─────────────────────────────────────────────────────────────┐
│                        离线索引阶段                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  文档 → 切分(Chunking) → Embedding → 存入向量数据库            │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        在线查询阶段                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  问题 → Embedding → 向量检索 → Top-K 文档 → 构造 Prompt → LLM │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 从零实现 RAG

### 完整代码

```python
import os
from typing import List, Dict
import numpy as np

# ========== 1. 文档切分 ==========
def simple_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """简单的固定长度切分"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def recursive_chunk(text: str, chunk_size: int = 500, separators: List[str] = None) -> List[str]:
    """递归切分：优先按语义边界切分"""
    if separators is None:
        separators = ["\n\n", "\n", "。", ".", " ", ""]

    chunks = []

    def split_text(text: str, sep_index: int = 0) -> List[str]:
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        if sep_index >= len(separators):
            # 无法再分，强制切分
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        sep = separators[sep_index]
        if sep:
            parts = text.split(sep)
        else:
            parts = list(text)

        result = []
        current = ""

        for part in parts:
            test = current + sep + part if current else part
            if len(test) <= chunk_size:
                current = test
            else:
                if current:
                    result.append(current)
                # 递归处理过长的部分
                if len(part) > chunk_size:
                    result.extend(split_text(part, sep_index + 1))
                else:
                    current = part

        if current:
            result.append(current)

        return result

    return split_text(text)


# ========== 2. Embedding ==========
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量生成 embedding"""
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, query: str) -> np.ndarray:
        """单条查询的 embedding"""
        return self.model.encode([query])[0]


# ========== 3. 向量存储 ==========
import faiss

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # 内积（余弦相似度）
        self.documents: List[Dict] = []

    def add(self, embeddings: np.ndarray, documents: List[Dict]):
        """添加文档"""
        # 归一化以使用内积计算余弦相似度
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """检索"""
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)

        return results


# ========== 4. 生成 ==========
from openai import OpenAI

client = OpenAI()

def generate_answer(query: str, contexts: List[str], system_prompt: str = None) -> str:
    """基于检索结果生成答案"""

    if system_prompt is None:
        system_prompt = """你是一个知识问答助手。请基于提供的参考资料回答用户问题。
如果参考资料中没有相关信息，请明确说明。
回答要准确、简洁，并在适当时候引用来源。"""

    context_text = "\n\n".join([f"[文档{i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])

    user_prompt = f"""参考资料：
{context_text}

用户问题：{query}

请基于参考资料回答问题："""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


# ========== 5. RAG Pipeline ==========
class SimpleRAG:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = EmbeddingModel(embedding_model)
        self.vector_store = VectorStore(dimension=384)

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """添加文档到知识库"""
        if metadatas is None:
            metadatas = [{}] * len(documents)

        # 切分
        all_chunks = []
        all_metadata = []

        for doc, meta in zip(documents, metadatas):
            chunks = recursive_chunk(doc, chunk_size=500)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    **meta,
                    "chunk_index": i,
                    "content": chunk
                })

        print(f"切分为 {len(all_chunks)} 个块")

        # Embedding
        embeddings = self.embedder.embed(all_chunks)

        # 存储
        self.vector_store.add(embeddings, all_metadata)

        print(f"已添加 {len(all_chunks)} 个文档块")

    def query(self, question: str, top_k: int = 5) -> Dict:
        """查询"""
        # 检索
        query_embedding = self.embedder.embed_query(question)
        results = self.vector_store.search(query_embedding, top_k)

        # 构造上下文
        contexts = [r['content'] for r in results]

        # 生成答案
        answer = generate_answer(question, contexts)

        return {
            "question": question,
            "answer": answer,
            "sources": results
        }


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 创建 RAG 实例
    rag = SimpleRAG()

    # 添加文档
    documents = [
        """
        Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。
        Python 以其简洁的语法和强大的标准库而闻名，广泛应用于 Web 开发、
        数据科学、人工智能等领域。Python 支持多种编程范式，包括面向对象、
        函数式和过程式编程。
        """,
        """
        机器学习是人工智能的一个分支，专注于让计算机从数据中学习。
        常见的机器学习算法包括线性回归、决策树、随机森林、支持向量机等。
        深度学习是机器学习的子领域，使用多层神经网络处理复杂任务。
        """,
        """
        大语言模型（LLM）是基于 Transformer 架构的预训练模型。
        代表性的 LLM 包括 GPT、BERT、LLaMA 等。这些模型通过在大规模
        文本数据上训练，学习语言的统计规律，能够完成文本生成、问答、
        摘要等多种任务。
        """
    ]

    metadatas = [
        {"source": "python_intro.txt"},
        {"source": "ml_basics.txt"},
        {"source": "llm_overview.txt"}
    ]

    rag.add_documents(documents, metadatas)

    # 查询
    result = rag.query("什么是大语言模型？它和机器学习有什么关系？")

    print("问题:", result["question"])
    print("\n答案:", result["answer"])
    print("\n来源:")
    for source in result["sources"][:3]:
        print(f"  - {source.get('source', 'unknown')} (score: {source['score']:.4f})")
```

---

## Chunking 策略详解

### 1. 固定长度切分

```python
def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    优点：简单、可预测
    缺点：可能在句子中间截断
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
```

### 2. 句子级切分

```python
import re

def sentence_chunk(text: str, max_chunk_size: int = 500):
    """按句子切分，尽量不超过最大长度"""
    # 分句
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
```

### 3. 语义切分

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunk(text: str, model: SentenceTransformer,
                   threshold: float = 0.7, max_chunk_size: int = 1000):
    """基于语义相似度的切分"""
    # 分句
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return sentences

    # 计算句子 embedding
    embeddings = model.encode(sentences)

    # 计算相邻句子的相似度
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        # 计算当前句子与前一句的相似度
        sim = np.dot(embeddings[i], embeddings[i-1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1])
        )

        current_text = "".join(current_chunk + [sentences[i]])

        # 如果相似度低于阈值或超过最大长度，开始新块
        if sim < threshold or len(current_text) > max_chunk_size:
            chunks.append("".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks
```

### 4. Parent-Child（父子文档）

```python
class ParentChildChunker:
    """
    父文档用于上下文，子文档用于精确检索
    """

    def __init__(self, parent_size: int = 2000, child_size: int = 400):
        self.parent_size = parent_size
        self.child_size = child_size

    def chunk(self, text: str) -> List[Dict]:
        results = []

        # 先切分父文档
        parents = simple_chunk(text, self.parent_size, overlap=200)

        for parent_idx, parent in enumerate(parents):
            # 再切分子文档
            children = simple_chunk(parent, self.child_size, overlap=50)

            for child_idx, child in enumerate(children):
                results.append({
                    "parent_id": parent_idx,
                    "parent_content": parent,
                    "child_content": child,
                    "child_id": f"{parent_idx}_{child_idx}"
                })

        return results
```

---

## Embedding 模型选择

```python
# 不同 Embedding 模型对比
models = {
    # 开源模型
    "all-MiniLM-L6-v2": {
        "dim": 384,
        "speed": "fast",
        "quality": "good",
        "multilingual": False
    },
    "all-mpnet-base-v2": {
        "dim": 768,
        "speed": "medium",
        "quality": "better",
        "multilingual": False
    },
    "multilingual-e5-large": {
        "dim": 1024,
        "speed": "slow",
        "quality": "best",
        "multilingual": True
    },

    # API 模型
    "text-embedding-3-small": {
        "dim": 1536,
        "provider": "OpenAI",
        "cost": "$0.02/1M tokens"
    },
    "text-embedding-3-large": {
        "dim": 3072,
        "provider": "OpenAI",
        "cost": "$0.13/1M tokens"
    }
}

# 使用 OpenAI Embedding
def openai_embed(texts: List[str], model: str = "text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]
```

---

## 向量数据库选择

```python
# 1. Faiss (本地，适合中小规模)
import faiss

index = faiss.IndexFlatIP(384)  # 内积
index = faiss.IndexFlatL2(384)  # L2 距离
index = faiss.IndexIVFFlat(quantizer, 384, nlist)  # 加速

# 2. ChromaDB (本地，易用)
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_collection")
collection.add(
    documents=["doc1", "doc2"],
    embeddings=[[0.1, 0.2], [0.3, 0.4]],
    ids=["id1", "id2"]
)
results = collection.query(query_embeddings=[[0.1, 0.2]], n_results=5)

# 3. Milvus (分布式，生产级)
from pymilvus import connections, Collection

connections.connect("default", host="localhost", port="19530")
# ... 创建 collection 和索引

# 4. Pinecone (云服务)
import pinecone

pinecone.init(api_key="xxx")
index = pinecone.Index("my-index")
index.upsert(vectors=[("id1", [0.1, 0.2], {"meta": "data"})])
```

---

## 练习题

### 练习 1：实现智能切分

```python
# 任务：实现一个切分函数，能够：
# 1. 识别 Markdown 标题作为自然边界
# 2. 保持代码块完整
# 3. 尽量按句子边界切分
```

### 练习 2：对比 Embedding 模型

```python
# 任务：对比以下模型在中文问答场景的效果
# - all-MiniLM-L6-v2
# - multilingual-e5-small
# - text-embedding-3-small
#
# 测试数据：准备 10 个中文问题和对应的文档
```

### 练习 3：构建简单 RAG

```python
# 任务：基于提供的代码，构建一个能够：
# 1. 读取 PDF 文件
# 2. 切分并索引
# 3. 回答关于 PDF 内容的问题
```

---

## 小结

```
本节要点：
1. RAG 流程：切分 → Embedding → 检索 → 生成
2. Chunking 策略：固定长度、句子级、语义级、Parent-Child
3. Embedding 选择：开源 vs API，多语言支持
4. 向量数据库：Faiss、ChromaDB、Milvus、Pinecone
```

---

## ➡️ 下一步

继续 [04-RAG进阶.md](./04-RAG进阶.md)

