# 📚 项目：企业知识库 RAG 助手

> 构建一个支持 PDF/网页的知识库问答系统

---

## 项目概述

### 功能需求

```
1. 文档管理
   - 支持上传 PDF、TXT、Markdown 文件
   - 支持网页链接抓取
   - 文档列表和删除

2. 智能问答
   - 基于知识库回答问题
   - 显示引用来源
   - 支持多轮对话

3. 管理界面
   - 文档管理
   - 对话历史
   - 系统配置
```

### 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (Streamlit)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│   │  文档上传    │  │   对话界面   │  │   管理界面   │    │
│   └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                        后端 (FastAPI)                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│   │  文档处理    │  │  RAG Pipeline │  │  API 服务   │    │
│   └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                        存储层                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│   │   ChromaDB   │  │   SQLite    │  │   文件存储   │    │
│   │  (向量存储)  │  │  (元数据)   │  │   (原始文档) │    │
│   └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 完整代码

### 项目结构

```
rag_knowledge_base/
├── app/
│   ├── __init__.py
│   ├── config.py          # 配置
│   ├── document_loader.py # 文档加载
│   ├── rag_engine.py      # RAG 引擎
│   ├── api.py             # FastAPI 服务
│   └── database.py        # 数据库操作
├── frontend/
│   └── app.py             # Streamlit 前端
├── data/
│   ├── uploads/           # 上传的文件
│   └── chroma_db/         # 向量数据库
├── requirements.txt
└── run.py
```

### requirements.txt

```
openai>=1.0.0
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.10
chromadb>=0.4.0
sentence-transformers>=2.2.0
pypdf>=3.0.0
python-docx>=0.8.11
beautifulsoup4>=4.12.0
requests>=2.31.0
fastapi>=0.100.0
uvicorn>=0.23.0
streamlit>=1.28.0
python-multipart>=0.0.6
python-dotenv>=1.0.0
```

### config.py

```python
"""配置文件"""
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # 路径
    base_dir: Path = Path(__file__).parent.parent
    upload_dir: Path = base_dir / "data" / "uploads"
    chroma_dir: Path = base_dir / "data" / "chroma_db"
    db_path: Path = base_dir / "data" / "metadata.db"

    # RAG 配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5

    class Config:
        env_file = ".env"

settings = Settings()

# 确保目录存在
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
```

### document_loader.py

```python
"""文档加载器"""
from pathlib import Path
from typing import List, Dict
import requests
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.schema import Document

from app.config import settings

class DocumentLoader:
    """文档加载和处理"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )

    def load_file(self, file_path: str) -> List[Document]:
        """加载文件"""
        path = Path(file_path)

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        elif path.suffix.lower() in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(str(path))
        else:
            raise ValueError(f"不支持的文件类型: {path.suffix}")

        documents = loader.load()

        # 添加元数据
        for doc in documents:
            doc.metadata["source"] = path.name
            doc.metadata["file_path"] = str(path)

        # 切分
        chunks = self.text_splitter.split_documents(documents)

        return chunks

    def load_url(self, url: str) -> List[Document]:
        """加载网页"""
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # 移除脚本和样式
        for script in soup(["script", "style"]):
            script.decompose()

        # 获取文本
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)

        # 创建文档
        doc = Document(
            page_content=text,
            metadata={"source": url, "type": "webpage"}
        )

        # 切分
        chunks = self.text_splitter.split_documents([doc])

        return chunks

    def process_documents(self, documents: List[Document]) -> List[Dict]:
        """处理文档为字典格式"""
        processed = []
        for i, doc in enumerate(documents):
            processed.append({
                "id": f"{doc.metadata.get('source', 'unknown')}_{i}",
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        return processed
```

### rag_engine.py

```python
"""RAG 引擎"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from app.config import settings
from app.document_loader import DocumentLoader

class RAGEngine:
    """RAG 引擎"""

    def __init__(self):
        # 初始化 Embedding
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )

        # 初始化 ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.3,
            openai_api_key=settings.openai_api_key
        )

        # 文档加载器
        self.loader = DocumentLoader()

    def add_document(self, file_path: str) -> int:
        """添加文档到知识库"""
        # 加载文档
        chunks = self.loader.load_file(file_path)

        if not chunks:
            return 0

        # 生成 embedding
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)

        # 存储到 ChromaDB
        ids = [f"{file_path}_{i}" for i in range(len(chunks))]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        return len(chunks)

    def add_url(self, url: str) -> int:
        """添加网页到知识库"""
        chunks = self.loader.load_url(url)

        if not chunks:
            return 0

        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)

        ids = [f"url_{hash(url)}_{i}" for i in range(len(chunks))]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        return len(chunks)

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """检索相关文档"""
        if top_k is None:
            top_k = settings.top_k

        # 生成查询 embedding
        query_embedding = self.embeddings.embed_query(query)

        # 检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # 格式化结果
        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # 转换为相似度
            })

        return formatted

    def query(self, question: str, chat_history: List[Dict] = None) -> Dict:
        """RAG 问答"""
        # 检索
        search_results = self.search(question)

        if not search_results:
            return {
                "answer": "抱歉，知识库中没有相关信息。",
                "sources": []
            }

        # 构造上下文
        context = "\n\n".join([
            f"[来源: {r['metadata'].get('source', '未知')}]\n{r['content']}"
            for r in search_results
        ])

        # 构造 prompt
        system_message = """你是一个知识库问答助手。请基于提供的参考资料回答用户问题。

要求：
1. 只使用参考资料中的信息回答
2. 如果参考资料中没有相关信息，明确告知用户
3. 回答要准确、简洁
4. 在适当时候引用来源"""

        user_message = f"""参考资料：
{context}

用户问题：{question}

请回答："""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]

        # 如果有历史对话，加入上下文
        if chat_history:
            history_text = "\n".join([
                f"用户: {h['user']}\n助手: {h['assistant']}"
                for h in chat_history[-3:]  # 只保留最近 3 轮
            ])
            messages.insert(1, HumanMessage(content=f"历史对话：\n{history_text}"))

        # 生成回答
        response = self.llm.invoke(messages)

        return {
            "answer": response.content,
            "sources": [
                {"source": r["metadata"].get("source", "未知"), "score": r["score"]}
                for r in search_results
            ]
        }

    def delete_document(self, source: str):
        """删除文档"""
        # 获取所有相关的 ID
        results = self.collection.get(
            where={"source": source},
            include=["metadatas"]
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    def get_stats(self) -> Dict:
        """获取知识库统计"""
        return {
            "total_chunks": self.collection.count(),
            "sources": self._get_unique_sources()
        }

    def _get_unique_sources(self) -> List[str]:
        """获取所有来源"""
        results = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            if "source" in meta:
                sources.add(meta["source"])
        return list(sources)


# 单例
_engine = None

def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
```

### api.py

```python
"""FastAPI 服务"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
from pathlib import Path

from app.config import settings
from app.rag_engine import get_engine

app = FastAPI(title="RAG 知识库 API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = None

class URLRequest(BaseModel):
    url: str

# API 端点
@app.get("/")
def root():
    return {"message": "RAG 知识库 API", "version": "1.0"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件"""
    # 保存文件
    file_path = settings.upload_dir / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 添加到知识库
    engine = get_engine()
    num_chunks = engine.add_document(str(file_path))

    return {
        "message": f"文件上传成功",
        "filename": file.filename,
        "chunks": num_chunks
    }

@app.post("/add_url")
async def add_url(request: URLRequest):
    """添加网页"""
    engine = get_engine()
    try:
        num_chunks = engine.add_url(request.url)
        return {"message": "网页添加成功", "url": request.url, "chunks": num_chunks}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    """问答"""
    engine = get_engine()
    result = engine.query(request.question, request.chat_history)
    return result

@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    engine = get_engine()
    return engine.get_stats()

@app.delete("/document/{source}")
async def delete_document(source: str):
    """删除文档"""
    engine = get_engine()
    deleted = engine.delete_document(source)
    return {"message": f"删除了 {deleted} 个文档块"}

@app.get("/sources")
async def get_sources():
    """获取所有来源"""
    engine = get_engine()
    stats = engine.get_stats()
    return {"sources": stats["sources"]}
```

### frontend/app.py

```python
"""Streamlit 前端"""
import streamlit as st
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(page_title="知识库助手", page_icon="📚", layout="wide")

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 侧边栏
with st.sidebar:
    st.header("📁 文档管理")

    # 文件上传
    uploaded_file = st.file_uploader(
        "上传文档",
        type=["pdf", "txt", "md"],
        help="支持 PDF、TXT、Markdown 格式"
    )

    if uploaded_file:
        if st.button("上传"):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(f"{API_URL}/upload", files=files)
            if response.status_code == 200:
                result = response.json()
                st.success(f"上传成功！处理了 {result['chunks']} 个文档块")
            else:
                st.error("上传失败")

    st.divider()

    # 网页链接
    url = st.text_input("添加网页链接")
    if st.button("添加网页"):
        if url:
            response = requests.post(f"{API_URL}/add_url", json={"url": url})
            if response.status_code == 200:
                result = response.json()
                st.success(f"添加成功！处理了 {result['chunks']} 个文档块")
            else:
                st.error("添加失败")

    st.divider()

    # 知识库统计
    st.subheader("📊 知识库统计")
    try:
        stats = requests.get(f"{API_URL}/stats").json()
        st.write(f"文档块数量: {stats['total_chunks']}")
        st.write("来源列表:")
        for source in stats.get("sources", []):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {source[:30]}...")
            with col2:
                if st.button("删除", key=f"del_{source}"):
                    requests.delete(f"{API_URL}/document/{source}")
                    st.rerun()
    except:
        st.write("无法连接到后端服务")

    st.divider()

    if st.button("清空对话历史"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# 主界面
st.title("📚 知识库问答助手")
st.caption("基于您的文档进行智能问答")

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("查看来源"):
                for source in message["sources"]:
                    st.write(f"• {source['source']} (相关度: {source['score']:.2f})")

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 获取回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": prompt,
                    "chat_history": st.session_state.chat_history
                }
            )

            if response.status_code == 200:
                result = response.json()
                st.markdown(result["answer"])

                # 显示来源
                if result.get("sources"):
                    with st.expander("查看来源"):
                        for source in result["sources"]:
                            st.write(f"• {source['source']} (相关度: {source['score']:.2f})")

                # 保存到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", [])
                })

                st.session_state.chat_history.append({
                    "user": prompt,
                    "assistant": result["answer"]
                })
            else:
                st.error("获取回答失败，请重试")
```

### run.py

```python
"""启动脚本"""
import subprocess
import sys
import time
import threading

def run_backend():
    subprocess.run([sys.executable, "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"])

def run_frontend():
    time.sleep(2)  # 等待后端启动
    subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"])

if __name__ == "__main__":
    # 启动后端
    backend_thread = threading.Thread(target=run_backend)
    backend_thread.start()

    # 启动前端
    run_frontend()
```

---

## 运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
echo "OPENAI_API_KEY=sk-xxx" > .env

# 3. 启动应用
python run.py

# 或分别启动
# 后端: uvicorn app.api:app --reload
# 前端: streamlit run frontend/app.py
```

---

## 扩展方向

```
1. 添加 Rerank 提升检索质量
2. 支持更多文档格式（Word、Excel）
3. 添加用户认证
4. 支持多知识库
5. 添加文档预览
6. 实现异步处理大文件
```

---

## ➡️ 下一步

继续 [11-项目-SQL-Agent.md](./11-项目-SQL-Agent.md)

