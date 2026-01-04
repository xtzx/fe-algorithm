# RAG 进阶

## 1. 检索器

### 1.1 基础检索

```python
from llm_kit.rag import Retriever, VectorIndex

retriever = Retriever(
    index=index,
    top_k=5,
    score_threshold=0.5,  # 最低相似度
)

results = retriever.search("What is RAG?")
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Source: {result.source}")
    print(f"Content: {result.content[:100]}...")
```

### 1.2 带引用的检索

```python
results, citations = retriever.search_with_citations("What is RAG?")

# 格式化引用
print(retriever.format_citations(citations))
# [1] knowledge/rag.md
# [2] docs/overview.md
```

### 1.3 获取上下文

```python
# 直接获取上下文字符串
context = retriever.get_context(
    query="What is RAG?",
    max_tokens=2000,
)

# 用于 LLM 提示
response = client.chat([
    {"role": "system", "content": f"Context:\n{context}"},
    {"role": "user", "content": query},
])
```

## 2. 混合检索

结合向量检索和关键词检索：

```python
from llm_kit.rag.retriever import HybridRetriever

# 两个检索器
vector_retriever = Retriever(vector_index)
keyword_retriever = Retriever(keyword_index)

# 混合（加权）
hybrid = HybridRetriever([
    (vector_retriever, 0.7),   # 70% 权重
    (keyword_retriever, 0.3),  # 30% 权重
])

results = hybrid.search("What is RAG?")
```

## 3. 元数据过滤

```python
# 只在特定来源中搜索
results = retriever.search(
    query="authentication",
    filters={"source": "docs/auth.md"},
)

# 多值过滤
results = retriever.search(
    query="API",
    filters={"type": ["md", "txt"]},
)
```

## 4. 分块策略选择

| 策略 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| 固定大小 | 通用 | 简单、可控 | 可能切断语义 |
| 句子 | 精确检索 | 保持语义完整 | 块大小不均匀 |
| 段落 | 长文档 | 自然边界 | 块可能太大 |
| 语义 | 高质量要求 | 最佳边界 | 需要额外处理 |

### 4.1 Token 感知分块

```python
from llm_kit.rag.chunker import create_length_function_tiktoken

# 使用 token 计数
length_func = create_length_function_tiktoken("gpt-4")

chunker = Chunker(
    chunk_size=500,  # 500 tokens
    length_function=length_func,
)
```

## 5. 上下文窗口管理

```python
def get_relevant_context(query: str, max_tokens: int = 4000):
    """获取不超过 max_tokens 的上下文"""
    results = retriever.search(query, top_k=20)
    
    context_parts = []
    total_tokens = 0
    
    for result in results:
        chunk_tokens = count_tokens(result.content)
        if total_tokens + chunk_tokens > max_tokens:
            break
        context_parts.append(result.content)
        total_tokens += chunk_tokens
    
    return "\n\n".join(context_parts)
```

## 6. 引用返回

```python
from llm_kit.prompts import RAGPromptTemplate

template = RAGPromptTemplate()

# 带引用的上下文
context = [
    {"content": chunk.content, "source": chunk.source}
    for chunk in results
]

prompt = template.render(
    question="What is RAG?",
    context=context,
)

# LLM 回答时会包含 [Source: xxx] 引用
```

## 7. 评估 RAG 系统

### 7.1 检索评估

- **召回率（Recall）**：相关文档被检索到的比例
- **精确率（Precision）**：检索结果中相关文档的比例
- **MRR**：第一个相关结果的排名倒数

### 7.2 端到端评估

- **答案相关性**：答案是否回答了问题
- **忠实度**：答案是否基于检索到的内容
- **引用准确性**：引用是否正确

## 8. 优化技巧

1. **调整 chunk_size**：根据内容类型调整
2. **增加 overlap**：避免丢失跨块信息
3. **元数据丰富**：添加更多可过滤字段
4. **查询改写**：使用 LLM 改写查询
5. **重排序**：用 LLM 对检索结果重排序


