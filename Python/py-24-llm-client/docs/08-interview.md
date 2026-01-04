# 面试高频题

## 1. 什么是 RAG？为什么需要 RAG？

### 答案

**RAG（Retrieval-Augmented Generation）** 是一种结合检索和生成的技术：

1. **检索**：从知识库中找到相关文档
2. **增强**：将文档作为上下文
3. **生成**：LLM 基于上下文生成答案

**为什么需要**：

| 问题 | RAG 如何解决 |
|------|-------------|
| 知识过时 | 检索最新文档 |
| 领域知识缺失 | 接入专业知识库 |
| 幻觉（编造） | 基于真实文档回答 |
| 无法引用来源 | 返回文档来源 |

```
用户问题 → 检索相关文档 → LLM + 文档上下文 → 有据可查的答案
```

---

## 2. 如何选择分块策略？

### 答案

**考虑因素**：

1. **内容类型**
   - 技术文档：段落分块
   - 对话记录：句子分块
   - 代码：函数/类分块

2. **检索精度要求**
   - 高精度：小块（200-500 字符）
   - 上下文丰富：大块（1000-2000 字符）

3. **嵌入模型限制**
   - 注意模型的最大输入长度

**推荐**：

```python
# 通用文档
Chunker(chunk_size=500, overlap=50, strategy=ChunkingStrategy.SENTENCE)

# 技术文档
Chunker(chunk_size=1000, overlap=100, strategy=ChunkingStrategy.PARAGRAPH)

# 高精度检索
Chunker(chunk_size=300, overlap=30, strategy=ChunkingStrategy.SENTENCE)
```

---

## 3. 如何处理 LLM 的流式响应？

### 答案

**SSE 格式解析**：

```python
for line in response.iter_lines():
    if line.startswith("data: "):
        data = json.loads(line[6:])
        if data == "[DONE]":
            break
        yield data["choices"][0]["delta"]["content"]
```

**关键点**：

1. **增量输出**：使用 `flush=True` 即时显示
2. **内容收集**：累积完整响应
3. **中断处理**：支持用户取消
4. **工具调用**：流式收集函数参数

---

## 4. 如何实现结构化输出？

### 答案

**方法**：

1. **JSON Schema 约束**
```python
system_prompt = f"""
Output JSON matching this schema:
{json.dumps(MyModel.model_json_schema())}
"""
```

2. **Pydantic 验证**
```python
data = json.loads(response.content)
result = MyModel.model_validate(data)
```

3. **重试机制**
```python
for attempt in range(max_retries):
    try:
        return parse_and_validate(response)
    except ValidationError as e:
        # 添加错误反馈，重试
        messages.append({"role": "user", "content": f"Invalid: {e}"})
```

4. **温度设为 0**：确保输出稳定

---

## 5. 如何评估 RAG 系统？

### 答案

**检索评估**：

- **召回率**：相关文档被检索到的比例
- **精确率**：检索结果中相关文档的比例
- **MRR**：第一个相关结果的排名倒数

**生成评估**：

- **答案相关性**：是否回答了问题
- **忠实度**：是否基于检索内容
- **引用准确性**：引用是否正确

**端到端评估**：

```python
# 构建测试集
test_cases = [
    {"question": "...", "expected_answer": "...", "relevant_docs": [...]},
]

# 评估
for case in test_cases:
    result = rag.ask(case["question"])
    # 比较 result 与 expected_answer
```

---

## 6. 如何处理上下文长度限制？

### 答案

**策略**：

1. **截断**：限制检索结果数量
```python
context = retriever.get_context(query, max_tokens=4000)
```

2. **压缩**：摘要长文档
```python
if len(doc) > max_length:
    doc = summarize(doc)
```

3. **分层检索**：先粗后细
```python
# 第一轮：获取相关文档
docs = retriever.search(query, top_k=10)
# 第二轮：在文档内检索段落
chunks = retriever.search_in_docs(query, docs)
```

4. **动态选择**：根据问题复杂度调整
```python
if is_complex_query(query):
    top_k = 5  # 更多上下文
else:
    top_k = 2  # 精简上下文
```

---

## 7. 什么是向量嵌入？

### 答案

**定义**：将文本转换为固定维度的向量表示。

```python
"Hello world" → [0.1, 0.2, 0.3, ..., 0.8]  # 768 维
```

**特点**：

- 相似文本 → 向量距离近
- 不同文本 → 向量距离远
- 支持语义搜索

**常用模型**：

| 模型 | 维度 | 特点 |
|------|------|------|
| OpenAI text-embedding-3-small | 1536 | API 调用 |
| all-MiniLM-L6-v2 | 384 | 本地、快速 |
| BGE-large | 1024 | 中文优化 |

---

## 8. 如何优化 RAG 的检索质量？

### 答案

**技术手段**：

1. **查询改写**
```python
# 使用 LLM 改写用户查询
rewritten = llm.chat("Rewrite for search: " + query)
```

2. **混合检索**
```python
# 向量 + 关键词
results = hybrid_search(vector_results, keyword_results)
```

3. **重排序**
```python
# 使用 LLM 对结果重排序
reranked = llm.rerank(query, results)
```

4. **元数据过滤**
```python
results = retriever.search(query, filters={"type": "recent"})
```

5. **分块优化**
- 调整 chunk_size
- 增加 overlap
- 使用语义分块

6. **嵌入模型选择**
- 选择适合领域的模型
- 考虑多语言支持

---

## 附加题

### 9. 如何实现 LLM 调用的成本控制？

1. **Token 计数**：使用 tiktoken
2. **成本估算**：根据定价计算
3. **预算限制**：设置每日/每月上限
4. **缓存**：相同查询复用结果
5. **模型选择**：简单任务用便宜模型

### 10. RAG vs Fine-tuning，如何选择？

| 场景 | 推荐方案 |
|------|----------|
| 知识经常更新 | RAG |
| 需要引用来源 | RAG |
| 改变模型行为/风格 | Fine-tuning |
| 私有数据不能上传 | Fine-tuning |
| 快速上线 | RAG |
| 成本敏感 | 视情况 |


