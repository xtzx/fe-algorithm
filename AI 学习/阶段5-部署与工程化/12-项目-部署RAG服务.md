# 🎯 项目：部署 RAG 服务

> 将 RAG 系统完整部署到生产环境

---

## 项目目标

```
将阶段4开发的 RAG 知识库系统：
1. 使用推理引擎（Ollama/vLLM）部署本地模型
2. 用 FastAPI 暴露 OpenAI 兼容接口
3. 用 Docker 打包整个系统
4. 添加日志记录和基础监控
5. 可选：使用 Ragas 评估质量
```

---

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      生产架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐                                              │
│   │  Nginx   │ ← 反向代理 + SSL + 限流                       │
│   └────┬─────┘                                              │
│        │                                                    │
│        ↓                                                    │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│   │ FastAPI  │────→│  Ollama  │     │ ChromaDB │          │
│   │  (API)   │     │  (LLM)   │     │ (向量DB) │          │
│   └──────────┘     └──────────┘     └──────────┘          │
│        │                                                    │
│        ↓                                                    │
│   ┌──────────┐     ┌──────────┐                            │
│   │   日志   │     │  监控    │                            │
│   └──────────┘     └──────────┘                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
rag-production/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用
│   ├── config.py            # 配置
│   ├── rag_engine.py        # RAG 引擎
│   ├── middleware.py        # 中间件
│   └── security.py          # 安全
├── docker/
│   ├── Dockerfile.api       # API 镜像
│   ├── Dockerfile.ollama    # Ollama 镜像
│   └── nginx.conf           # Nginx 配置
├── scripts/
│   ├── init_db.py           # 初始化数据库
│   └── evaluate.py          # 评估脚本
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 核心代码

### config.py

```python
"""配置管理"""
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # 应用
    app_name: str = "RAG Production API"
    debug: bool = False

    # LLM
    llm_backend: str = "http://ollama:11434"
    llm_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"

    # 向量数据库
    chroma_host: str = "chromadb"
    chroma_port: int = 8000

    # 安全
    api_key: str = ""

    # 日志
    log_level: str = "INFO"
    log_file: str = "/var/log/rag/app.log"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
```

### main.py

```python
"""FastAPI 主应用"""
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import time
import uuid
import structlog

from api.config import get_settings, Settings
from api.rag_engine import RAGEngine
from api.middleware import LoggingMiddleware, RateLimitMiddleware
from api.security import verify_api_key

# 配置日志
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# 创建应用
app = FastAPI(title="RAG Production API", version="1.0.0")

# 中间件
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

# RAG 引擎（延迟初始化）
rag_engine: Optional[RAGEngine] = None

@app.on_event("startup")
async def startup():
    global rag_engine
    settings = get_settings()
    rag_engine = RAGEngine(settings)
    logger.info("RAG engine initialized")

# ========== 数据模型 ==========
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "rag"
    stream: bool = False
    temperature: float = 0.7

class DocumentUpload(BaseModel):
    content: str
    source: str

# ========== API 端点 ==========
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """RAG 问答接口"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # 获取最后一条用户消息
    user_message = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        ""
    )

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    logger.info("chat_request", request_id=request_id, user_message=user_message[:100])

    if request.stream:
        return StreamingResponse(
            rag_engine.stream_query(user_message, request_id),
            media_type="text/event-stream"
        )
    else:
        result = await rag_engine.query(user_message)

        latency = time.time() - start_time
        logger.info("chat_response", request_id=request_id, latency_ms=latency*1000)

        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["answer"]},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0)
            }
        }

@app.post("/documents")
async def add_document(
    doc: DocumentUpload,
    api_key: str = Depends(verify_api_key)
):
    """添加文档到知识库"""
    num_chunks = await rag_engine.add_document(doc.content, doc.source)
    return {"message": "Document added", "chunks": num_chunks}

@app.get("/stats")
async def get_stats(api_key: str = Depends(verify_api_key)):
    """获取统计信息"""
    return await rag_engine.get_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### rag_engine.py

```python
"""RAG 引擎"""
import httpx
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, AsyncGenerator
import json

class RAGEngine:
    def __init__(self, settings):
        self.settings = settings

        # 初始化 Embedding 模型
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # 初始化 ChromaDB
        self.chroma = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port
        )
        self.collection = self.chroma.get_or_create_collection("knowledge_base")

    async def add_document(self, content: str, source: str) -> int:
        """添加文档"""
        # 切分
        chunks = self._chunk_text(content)

        # Embedding
        embeddings = self.embedder.encode(chunks).tolist()

        # 存储
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"source": source}] * len(chunks)
        )

        return len(chunks)

    async def query(self, question: str) -> Dict:
        """RAG 查询"""
        # 检索
        query_embedding = self.embedder.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=5
        )

        contexts = results["documents"][0] if results["documents"] else []

        # 生成
        answer = await self._generate(question, contexts)

        return {
            "answer": answer,
            "sources": results.get("metadatas", [[]])[0]
        }

    async def stream_query(self, question: str, request_id: str) -> AsyncGenerator:
        """流式 RAG 查询"""
        # 检索
        query_embedding = self.embedder.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=5
        )

        contexts = results["documents"][0] if results["documents"] else []

        # 流式生成
        async for chunk in self._stream_generate(question, contexts):
            yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"

        yield "data: [DONE]\n\n"

    async def _generate(self, question: str, contexts: List[str]) -> str:
        """调用 LLM 生成"""
        context_text = "\n\n".join(contexts)

        prompt = f"""基于以下参考资料回答问题。

参考资料：
{context_text}

问题：{question}

回答："""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.settings.llm_backend}/api/generate",
                json={
                    "model": self.settings.llm_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60.0
            )

            return response.json()["response"]

    async def _stream_generate(self, question: str, contexts: List[str]) -> AsyncGenerator:
        """流式生成"""
        context_text = "\n\n".join(contexts)

        prompt = f"""基于以下参考资料回答问题。

参考资料：
{context_text}

问题：{question}

回答："""

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.settings.llm_backend}/api/generate",
                json={
                    "model": self.settings.llm_model,
                    "prompt": prompt,
                    "stream": True
                },
                timeout=60.0
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if not data.get("done"):
                            yield data["response"]

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """切分文本"""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks

    async def get_stats(self) -> Dict:
        """获取统计"""
        return {
            "total_documents": self.collection.count(),
            "model": self.settings.llm_model
        }
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api
    restart: unless-stopped

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    environment:
      - LLM_BACKEND=http://ollama:11434
      - CHROMA_HOST=chromadb
      - API_KEY=${API_KEY:-secret123}
    depends_on:
      - ollama
      - chromadb
    restart: unless-stopped
    volumes:
      - ./logs:/var/log/rag

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    restart: unless-stopped

volumes:
  ollama_data:
  chroma_data:
```

### Dockerfile.api

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/

RUN mkdir -p /var/log/rag

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 部署步骤

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 设置 API_KEY 等

# 2. 启动服务
docker compose up -d

# 3. 下载模型（首次）
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull nomic-embed-text

# 4. 验证服务
curl http://localhost/health

# 5. 测试 API
curl http://localhost/v1/chat/completions \
    -H "Authorization: Bearer secret123" \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "你好"}]}'

# 6. 查看日志
docker compose logs -f api
```

---

## 实现 Checklist

```
□ Ollama/vLLM 部署并可访问
□ FastAPI 服务正常运行
□ OpenAI 兼容接口可用
□ 流式输出正常
□ Docker 打包完成
□ Docker Compose 编排完成
□ 日志记录正常
□ API 认证已实现
□ 健康检查端点可用
□ 可选：Ragas 评估脚本
```

---

## 运维命令

```bash
# 查看状态
docker compose ps

# 查看日志
docker compose logs -f

# 重启服务
docker compose restart api

# 更新部署
docker compose pull
docker compose up -d --build

# 备份数据
docker compose exec chromadb tar -czvf /backup/chroma.tar.gz /chroma/chroma
```

---

## ➡️ 下一步

继续 [13-自测清单.md](./13-自测清单.md)

