# P24: LLM 客户端与 RAG

> 构建生产级 LLM 应用

## 🎯 学完后能做

- 构建 LLM 客户端抽象
- 实现 RAG 系统
- 处理流式输出

## 📁 目录结构

```
py-24-llm-client/
├── README.md
├── pyproject.toml
├── docs/
│   ├── 01-llm-client.md        # LLM 客户端
│   ├── 02-structured-output.md # 结构化输出
│   ├── 03-streaming.md         # 流式处理
│   ├── 04-rag-basics.md        # RAG 基础
│   ├── 05-rag-advanced.md      # RAG 进阶
│   ├── 06-prompts.md           # 提示工程
│   ├── 07-exercises.md         # 练习题
│   └── 08-interview.md         # 面试题
├── src/llm_kit/
│   ├── __init__.py
│   ├── client/
│   │   ├── __init__.py
│   │   ├── base.py             # 基础客户端抽象
│   │   ├── openai.py           # OpenAI 实现
│   │   ├── streaming.py        # 流式处理
│   │   └── structured.py       # 结构化输出
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py           # 文档加载器
│   │   ├── chunker.py          # 分块策略
│   │   ├── embedder.py         # 向量嵌入
│   │   ├── index.py            # 向量存储
│   │   └── retriever.py        # 检索器
│   └── prompts/
│       ├── __init__.py
│       ├── template.py         # 模板引擎
│       └── chat.py             # 对话管理
├── tests/
├── examples/
└── scripts/
```

## 🚀 快速开始

### 安装

```bash
cd py-24-llm-client
pip install -e ".[dev]"
```

### LLM 客户端使用

```python
from llm_kit.client import OpenAIClient

# 创建客户端
client = OpenAIClient(api_key="your-key")

# 基础调用
response = client.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gpt-4o-mini",
)
print(response.content)

# 流式输出
for chunk in client.chat_stream(messages=[...]):
    print(chunk.delta, end="", flush=True)
```

### 结构化输出

```python
from pydantic import BaseModel
from llm_kit.client import StructuredClient

class Person(BaseModel):
    name: str
    age: int
    occupation: str

client = StructuredClient(api_key="your-key")
person = client.generate(
    prompt="Extract: John is a 30-year-old software engineer",
    schema=Person,
)
print(person.name, person.age)  # John 30
```

### RAG 系统

```python
from llm_kit.rag import DocumentLoader, Chunker, Embedder, VectorIndex, Retriever

# 1. 加载文档
loader = DocumentLoader()
docs = loader.load_directory("./docs")

# 2. 分块
chunker = Chunker(chunk_size=500, overlap=50)
chunks = chunker.split_documents(docs)

# 3. 嵌入并索引
embedder = Embedder()
index = VectorIndex(embedder)
index.add_documents(chunks)

# 4. 检索
retriever = Retriever(index, top_k=3)
results = retriever.search("What is RAG?")

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.content[:100]}...")
    print(f"Source: {result.metadata['source']}")
```

## 🔧 核心概念

### 1. LLM 客户端抽象

```python
# 不绑定厂商的抽象接口
class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, messages, **kwargs) -> ChatResponse: ...
    
    @abstractmethod
    def chat_stream(self, messages, **kwargs) -> Iterator[StreamChunk]: ...
```

### 2. 结构化输出

```python
# JSON Schema 约束 + Pydantic 验证
response = client.generate_structured(
    prompt="...",
    schema=MyModel,
    max_retries=3,  # 验证失败自动重试
)
```

### 3. RAG 流程

```
文档 → 加载 → 分块 → 嵌入 → 索引
                              ↓
查询 → 嵌入 → 检索 → 上下文 → LLM → 回答
```

## 📚 学习路径

1. **LLM 客户端** - 抽象设计、重试、超时
2. **结构化输出** - JSON Schema、Pydantic
3. **流式处理** - SSE、增量解析
4. **RAG 基础** - 加载、分块、嵌入
5. **RAG 进阶** - 检索优化、引用
6. **提示工程** - 模板、上下文管理

## ✅ 功能清单

- [x] LLM 客户端抽象
- [x] 不绑定厂商 SDK
- [x] timeout/retry/streaming
- [x] 幂等 request_id
- [x] 成本/耗时统计
- [x] JSON Schema 结构化输出
- [x] pydantic 验证
- [x] SSE 流式响应
- [x] 增量解析
- [x] 中断处理
- [x] 文档加载器
- [x] 分块策略
- [x] 向量嵌入
- [x] 向量存储
- [x] 检索器
- [x] 引用返回
- [x] 模板设计
- [x] 上下文管理
- [x] 多轮对话


