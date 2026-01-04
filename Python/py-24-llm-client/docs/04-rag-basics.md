# RAG 基础

## 什么是 RAG

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术：

```
问题 → 检索相关文档 → 将文档作为上下文 → LLM 生成答案
```

## 为什么需要 RAG

1. **知识时效性**：LLM 训练数据有截止日期
2. **领域知识**：LLM 可能不了解特定领域
3. **准确性**：减少幻觉，基于真实文档回答
4. **引用来源**：可以标注答案来源

## RAG 流程

```
1. 离线索引
   文档 → 加载 → 分块 → 嵌入 → 存储

2. 在线检索
   查询 → 嵌入 → 搜索 → 排序 → 返回
   
3. 生成回答
   查询 + 上下文 → LLM → 答案
```

## 1. 文档加载

### 1.1 加载文件

```python
from llm_kit.rag import DocumentLoader

loader = DocumentLoader()

# 单个文件
doc = loader.load_file("knowledge.md")
print(doc.content)
print(doc.metadata)  # {"source": "knowledge.md", ...}

# 目录
docs = loader.load_directory("./docs", pattern="*.md")
```

### 1.2 支持的格式

- `.txt` - 纯文本
- `.md` - Markdown
- `.json` - JSON

### 1.3 网页加载

```python
from llm_kit.rag.loader import WebLoader

loader = WebLoader()
doc = loader.load_url("https://example.com")
```

## 2. 文档分块

### 2.1 为什么要分块

- 嵌入模型有输入限制
- 小块更精确定位相关内容
- LLM 上下文窗口有限

### 2.2 分块策略

```python
from llm_kit.rag import Chunker, ChunkingStrategy

# 固定大小
chunker = Chunker(
    chunk_size=500,
    overlap=50,
    strategy=ChunkingStrategy.FIXED_SIZE,
)

# 按句子
chunker = Chunker(
    chunk_size=500,
    strategy=ChunkingStrategy.SENTENCE,
)

# 按段落
chunker = Chunker(
    chunk_size=1000,
    strategy=ChunkingStrategy.PARAGRAPH,
)

chunks = chunker.split(document)
```

### 2.3 重叠（Overlap）

重叠确保跨块的上下文不丢失：

```
Chunk 1: [--------------------]
Chunk 2:              [--------------------]
                      ^^^^^^^
                      overlap
```

## 3. 向量嵌入

### 3.1 概念

将文本转换为向量，相似的文本在向量空间中距离接近。

```python
from llm_kit.rag import Embedder

embedder = Embedder()  # Stub 实现

# 单个文本
vector = embedder.embed("Hello world")  # [0.1, 0.2, ...]

# 批量
vectors = embedder.embed_batch(["Hello", "World"])
```

### 3.2 真实嵌入模型

```python
# OpenAI
from llm_kit.rag.embedder import OpenAIEmbedder

embedder = OpenAIEmbedder(api_key="sk-...")
vector = embedder.embed("Hello world")

# Sentence Transformers（本地）
from llm_kit.rag.embedder import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
```

## 4. 向量索引

```python
from llm_kit.rag import VectorIndex

# 创建索引
index = VectorIndex(embedder)

# 添加文档
index.add_chunks(chunks)

# 搜索
results = index.search("What is RAG?", top_k=5)
for entry, score in results:
    print(f"Score: {score:.3f}")
    print(f"Content: {entry.content[:100]}...")

# 持久化
index.save("./my_index")

# 加载
index = VectorIndex.load("./my_index", embedder)
```

## 5. 完整示例

```python
from llm_kit.rag import DocumentLoader, Chunker, Embedder, VectorIndex

# 1. 加载
loader = DocumentLoader()
docs = loader.load_directory("./knowledge_base")

# 2. 分块
chunker = Chunker(chunk_size=500, overlap=50)
chunks = chunker.split_documents(docs)

# 3. 索引
embedder = Embedder()
index = VectorIndex(embedder)
index.add_chunks(chunks)

# 4. 检索
results = index.search("How does authentication work?")

# 5. 生成
context = "\n\n".join([r[0].content for r in results[:3]])
response = llm_client.chat([
    {"role": "system", "content": f"Context:\n{context}"},
    {"role": "user", "content": "How does authentication work?"},
])
```


