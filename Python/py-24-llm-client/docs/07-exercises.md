# 练习题

## 练习 1: LLM 客户端

实现一个支持超时和重试的 LLM 客户端：

**要求**：
1. 请求超时 30 秒
2. 速率限制错误自动重试（最多 3 次）
3. 指数退避策略

---

## 练习 2: 流式输出

实现流式输出的 CLI 程序：

```python
# 期望效果
# $ python chat.py "What is Python?"
# Python is a high-level programming language...
# (逐字输出)
```

**要求**：
1. 使用 `chat_stream`
2. 实时显示输出
3. 支持 Ctrl+C 中断

---

## 练习 3: 结构化输出

使用结构化输出提取新闻信息：

```python
class NewsArticle(BaseModel):
    title: str
    date: str
    summary: str
    keywords: List[str]
```

**要求**：
1. 从任意新闻文本提取信息
2. 实现验证失败重试
3. 处理提取失败的情况

---

## 练习 4: 文档分块

实现一个 Markdown 感知的分块器：

**要求**：
1. 保持代码块完整
2. 在标题处分割
3. 保留层级信息

---

## 练习 5: 向量索引

实现一个支持持久化的向量索引：

**要求**：
1. 添加文档
2. 余弦相似度搜索
3. 保存到文件
4. 从文件加载

---

## 练习 6: 检索器

实现带元数据过滤的检索器：

```python
results = retriever.search(
    "authentication",
    filters={
        "type": "markdown",
        "author": "admin",
    }
)
```

---

## 练习 7: RAG 系统

构建一个完整的 RAG 问答系统：

**要求**：
1. 加载文档目录
2. 分块和索引
3. 检索相关内容
4. 使用 LLM 生成答案
5. 返回引用来源

---

## 练习 8: 对话历史

实现带上下文窗口的对话管理：

**要求**：
1. 限制最大 token 数
2. 超过时移除旧消息
3. 保留系统提示

---

## 练习 9: 提示模板

设计一个代码解释器的提示模板：

```python
template.render(
    code="def fib(n): ...",
    language="python",
    detail_level="beginner",
)
```

---

## 练习 10: 函数调用

实现一个计算器工具：

```python
class CalculatorArgs(BaseModel):
    expression: str

# 用户: "What is 123 * 456?"
# LLM: [tool_call: calculator(expression="123 * 456")]
# 结果: 56088
```

---

## 练习 11: 混合检索

实现向量 + 关键词的混合检索：

**要求**：
1. 两种检索方式
2. 加权融合
3. 结果去重

---

## 练习 12: 成本统计

实现详细的成本统计：

```python
# 期望输出
{
    "total_requests": 100,
    "total_tokens": 50000,
    "input_tokens": 30000,
    "output_tokens": 20000,
    "total_cost": 0.05,
    "by_model": {
        "gpt-4o-mini": {"requests": 80, "cost": 0.03},
        "gpt-4": {"requests": 20, "cost": 0.02},
    }
}
```

---

## 参考答案

### 练习 1 答案

```python
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

class RetryClient:
    def __init__(self, api_key, timeout=30.0):
        self.client = OpenAIClient(api_key, timeout=timeout)
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
    )
    def chat(self, messages, **kwargs):
        return self.client.chat(messages, **kwargs)
```

### 练习 7 答案

```python
class RAGSystem:
    def __init__(self, docs_dir: str, llm_client):
        self.llm = llm_client
        
        # 加载和索引
        loader = DocumentLoader()
        docs = loader.load_directory(docs_dir)
        
        chunker = Chunker(chunk_size=500)
        chunks = chunker.split_documents(docs)
        
        embedder = Embedder()
        self.index = VectorIndex(embedder)
        self.index.add_chunks(chunks)
        
        self.retriever = Retriever(self.index, top_k=3)
        self.template = RAGPromptTemplate()
    
    def ask(self, question: str) -> dict:
        results, citations = self.retriever.search_with_citations(question)
        
        context = [
            {"content": r.content, "source": r.source}
            for r in results
        ]
        
        prompt = self.template.render(question=question, context=context)
        response = self.llm.chat([{"role": "user", "content": prompt}])
        
        return {
            "answer": response.content,
            "citations": [c.source for c in citations],
        }
```


