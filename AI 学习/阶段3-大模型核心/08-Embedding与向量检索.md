# 🔍 08 - Embedding 与向量检索

> Embedding 将文本转为向量，向量检索是 RAG 的核心技术

---

## 目录

1. [文本 Embedding](#1-文本-embedding)
2. [相似度计算](#2-相似度计算)
3. [向量数据库](#3-向量数据库)
4. [实战：语义搜索](#4-实战语义搜索)
5. [练习题](#5-练习题)

---

## 1. 文本 Embedding

### 1.1 什么是 Embedding

```
Embedding：将离散对象（文本、图像）映射到连续向量空间

文本 → Embedding 模型 → 向量（如 768 维）

特点：
- 语义相似的文本，向量也相近
- 可以用向量距离衡量语义相似度
- 是 RAG、语义搜索的基础
```

### 1.2 获取 Embedding

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# 方法 1：使用 BERT 获取 Embedding
def get_bert_embedding(text, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # 方法 1: [CLS] token
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # 方法 2: 平均池化（通常效果更好）
    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    mean_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return mean_embedding

# 测试
text = "Hello, how are you?"
embedding = get_bert_embedding(text)
print(f"Embedding shape: {embedding.shape}")  # [1, 768]
```

### 1.3 专用 Embedding 模型

```python
from sentence_transformers import SentenceTransformer

# 方法 2：使用 Sentence Transformers（推荐）
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "This is a sentence.",
    "This is another sentence.",
    "Completely different topic."
]

# 获取 embedding
embeddings = model.encode(sentences)
print(f"Embeddings shape: {embeddings.shape}")  # [3, 384]

# 常用的 Embedding 模型
"""
通用：
- sentence-transformers/all-MiniLM-L6-v2 (384维，快)
- sentence-transformers/all-mpnet-base-v2 (768维，效果好)

中文：
- shibing624/text2vec-base-chinese
- BAAI/bge-base-zh-v1.5
- moka-ai/m3e-base

多语言：
- BAAI/bge-m3
- intfloat/multilingual-e5-large
"""

# 使用 BGE 模型
from transformers import AutoTokenizer, AutoModel

def get_bge_embedding(texts, model_name="BAAI/bge-base-en-v1.5"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # BGE 推荐加前缀
    texts = ["passage: " + t for t in texts]

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings

# 测试
embeddings = get_bge_embedding(["Hello world", "你好世界"])
print(f"BGE embeddings: {embeddings.shape}")
```

---

## 2. 相似度计算

### 2.1 余弦相似度

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(a, b):
    """余弦相似度：[-1, 1]，越大越相似"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 使用 sklearn
embeddings = model.encode([
    "I love machine learning",
    "I enjoy deep learning",
    "The weather is nice today"
])

similarity_matrix = cosine_similarity(embeddings)
print("相似度矩阵:")
print(similarity_matrix)

# 结果：前两句相似度高，第三句与前两句相似度低
```

### 2.2 其他距离度量

```python
from scipy.spatial.distance import euclidean, cityblock

# 欧氏距离（L2）
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# 曼哈顿距离（L1）
def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

# 点积（如果向量已归一化，等价于余弦相似度）
def dot_product(a, b):
    return np.dot(a, b)

# 比较
a = embeddings[0]
b = embeddings[1]
c = embeddings[2]

print(f"a-b 余弦相似度: {cosine_sim(a, b):.4f}")
print(f"a-c 余弦相似度: {cosine_sim(a, c):.4f}")
print(f"a-b 欧氏距离: {euclidean_distance(a, b):.4f}")
print(f"a-c 欧氏距离: {euclidean_distance(a, c):.4f}")
```

---

## 3. 向量数据库

### 3.1 Faiss

```python
import faiss
import numpy as np

# 准备数据
d = 384  # 向量维度
nb = 10000  # 数据库大小
nq = 5  # 查询数量

np.random.seed(42)
xb = np.random.random((nb, d)).astype('float32')  # 数据库向量
xq = np.random.random((nq, d)).astype('float32')  # 查询向量

# 创建索引
# 1. 精确搜索（适合小数据集）
index_flat = faiss.IndexFlatL2(d)

# 2. IVF（适合大数据集）
nlist = 100  # 聚类数
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
index_ivf.train(xb)  # 需要训练

# 3. HNSW（高精度，内存换速度）
index_hnsw = faiss.IndexHNSWFlat(d, 32)  # 32 是 M 参数

# 添加向量
index_flat.add(xb)
print(f"索引大小: {index_flat.ntotal}")

# 搜索
k = 4  # 返回最近的 k 个
D, I = index_flat.search(xq, k)
print(f"距离:\n{D}")
print(f"索引:\n{I}")

# 带 ID 的索引
index_with_ids = faiss.IndexIDMap(faiss.IndexFlatL2(d))
ids = np.arange(nb).astype('int64')
index_with_ids.add_with_ids(xb, ids)
```

### 3.2 ChromaDB

```python
import chromadb
from chromadb.config import Settings

# 创建客户端
client = chromadb.Client()
# 或持久化存储
# client = chromadb.PersistentClient(path="./chroma_db")

# 创建集合
collection = client.create_collection(
    name="my_collection",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

# 添加文档
collection.add(
    documents=[
        "This is a document about machine learning",
        "Deep learning is a subset of machine learning",
        "The weather is sunny today"
    ],
    metadatas=[
        {"source": "doc1"},
        {"source": "doc2"},
        {"source": "doc3"}
    ],
    ids=["id1", "id2", "id3"]
)

# 查询
results = collection.query(
    query_texts=["What is deep learning?"],
    n_results=2
)

print("查询结果:")
print(f"Documents: {results['documents']}")
print(f"Distances: {results['distances']}")
print(f"IDs: {results['ids']}")

# 使用自定义 embedding
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

class CustomEmbedding:
    def __call__(self, texts):
        return model.encode(texts).tolist()

# collection = client.create_collection(
#     name="custom_embedding_collection",
#     embedding_function=CustomEmbedding()
# )
```

### 3.3 向量数据库对比

```
| 数据库 | 特点 | 适用场景 |
|--------|------|---------|
| Faiss | 高性能，Meta 开发 | 大规模相似度搜索 |
| ChromaDB | 简单易用，内置 embedding | 原型开发，小规模 |
| Pinecone | 云服务，托管 | 生产环境，不想运维 |
| Milvus | 分布式，高可用 | 大规模生产环境 |
| Weaviate | 支持混合搜索 | 需要关键词+向量搜索 |
| Qdrant | Rust 实现，高性能 | 高性能需求 |
```

---

## 4. 实战：语义搜索

### 4.1 完整语义搜索引擎

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SemanticSearchEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, documents):
        """构建索引"""
        self.documents = documents

        # 获取 embeddings
        embeddings = self.model.encode(documents, show_progress_bar=True)
        embeddings = embeddings.astype('float32')

        # 归一化（用于余弦相似度）
        faiss.normalize_L2(embeddings)

        # 创建索引
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)  # IP = Inner Product（归一化后等于余弦）
        self.index.add(embeddings)

        print(f"索引构建完成，共 {self.index.ntotal} 个文档")

    def search(self, query, k=5):
        """搜索"""
        # 查询 embedding
        query_embedding = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)

        # 搜索
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'rank': i + 1,
                    'score': float(score),
                    'document': self.documents[idx]
                })

        return results

    def save(self, path):
        """保存索引"""
        faiss.write_index(self.index, f"{path}/index.faiss")
        np.save(f"{path}/documents.npy", self.documents)

    def load(self, path):
        """加载索引"""
        self.index = faiss.read_index(f"{path}/index.faiss")
        self.documents = np.load(f"{path}/documents.npy", allow_pickle=True).tolist()

# 使用示例
documents = [
    "Python is a popular programming language for machine learning.",
    "TensorFlow is an open-source machine learning framework by Google.",
    "PyTorch is developed by Facebook and is widely used in research.",
    "Natural language processing deals with text and speech data.",
    "Computer vision is about teaching computers to understand images.",
    "Reinforcement learning is about learning through trial and error.",
    "Deep learning uses neural networks with many layers.",
    "Transfer learning allows reusing pre-trained models.",
]

# 创建搜索引擎
engine = SemanticSearchEngine()
engine.build_index(documents)

# 搜索
query = "What framework should I use for neural networks?"
results = engine.search(query, k=3)

print(f"\n查询: {query}")
print("-" * 50)
for r in results:
    print(f"[{r['rank']}] (score: {r['score']:.4f})")
    print(f"    {r['document']}")
```

### 4.2 混合搜索

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearchEngine:
    """混合搜索：BM25 关键词匹配 + 语义向量搜索"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.bm25 = None
        self.embeddings = None

    def build_index(self, documents):
        self.documents = documents

        # BM25 索引
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # 向量索引
        self.embeddings = self.model.encode(documents)
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

    def search(self, query, k=5, alpha=0.5):
        """
        混合搜索
        alpha: 语义搜索权重（0-1）
        """
        # BM25 分数
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)

        # 语义分数
        query_emb = self.model.encode([query])
        query_emb = query_emb / np.linalg.norm(query_emb)
        semantic_scores = np.dot(self.embeddings, query_emb.T).flatten()
        semantic_scores = (semantic_scores + 1) / 2  # 归一化到 [0, 1]

        # 混合分数
        hybrid_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores

        # 排序
        top_indices = np.argsort(hybrid_scores)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(hybrid_scores[idx]),
                'bm25_score': float(bm25_scores[idx]),
                'semantic_score': float(semantic_scores[idx])
            })

        return results

# 测试
hybrid_engine = HybridSearchEngine()
hybrid_engine.build_index(documents)

results = hybrid_engine.search("Python deep learning", k=3, alpha=0.7)
print("\n混合搜索结果:")
for r in results:
    print(f"Score: {r['score']:.4f} (BM25: {r['bm25_score']:.4f}, Semantic: {r['semantic_score']:.4f})")
    print(f"  {r['document']}")
```

---

## 5. 练习题

### 基础练习

1. 比较不同 Embedding 模型在语义相似度任务上的效果
2. 用 ChromaDB 构建一个简单的文档检索系统
3. 实现一个支持增量更新的向量索引

### 参考答案

<details>
<summary>点击查看答案</summary>

```python
# 1. Embedding 模型对比
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compare_models(sentences, similar_pairs, dissimilar_pairs):
    models = [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
    ]

    for model_name in models:
        print(f"\n{model_name}:")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(sentences)

        # 相似对的平均相似度
        similar_scores = []
        for i, j in similar_pairs:
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            similar_scores.append(sim)

        # 不相似对的平均相似度
        dissimilar_scores = []
        for i, j in dissimilar_pairs:
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            dissimilar_scores.append(sim)

        print(f"  相似对平均分: {np.mean(similar_scores):.4f}")
        print(f"  不相似对平均分: {np.mean(dissimilar_scores):.4f}")
        print(f"  区分度: {np.mean(similar_scores) - np.mean(dissimilar_scores):.4f}")

sentences = [
    "I love machine learning",      # 0
    "Machine learning is great",    # 1
    "The weather is nice today",    # 2
    "It's sunny outside",           # 3
]

similar_pairs = [(0, 1), (2, 3)]
dissimilar_pairs = [(0, 2), (0, 3), (1, 2), (1, 3)]

compare_models(sentences, similar_pairs, dissimilar_pairs)


# 2. ChromaDB 文档检索
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")

documents = [
    "Python tutorial for beginners",
    "Advanced machine learning techniques",
    "Introduction to deep learning",
    "Web development with Django",
]

collection.add(
    documents=documents,
    ids=[f"doc{i}" for i in range(len(documents))]
)

# 检索
results = collection.query(
    query_texts=["How to learn Python?"],
    n_results=2
)
print("检索结果:", results['documents'])


# 3. 增量更新索引
class IncrementalIndex:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def add(self, documents):
        """增量添加文档"""
        new_embeddings = self.model.encode(documents)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.documents.extend(documents)
        print(f"添加 {len(documents)} 个文档，总计 {len(self.documents)} 个")

    def search(self, query, k=3):
        query_emb = self.model.encode([query])
        scores = cosine_similarity(query_emb, self.embeddings)[0]
        top_k = np.argsort(scores)[::-1][:k]
        return [(self.documents[i], scores[i]) for i in top_k]

# 测试
idx = IncrementalIndex()
idx.add(["Document 1", "Document 2"])
idx.add(["Document 3"])
print(idx.search("doc"))
```

</details>

---

## ➡️ 下一步

学完本节后，继续学习 [09-多模态基础.md](./09-多模态基础.md)

