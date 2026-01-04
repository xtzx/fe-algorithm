# 🎯 AI 知识库助手

> P26 终极项目 - 综合运用所有 Python 知识构建的生产级 AI 应用

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 项目简介

企业知识库问答助手是一个完整的生产级 RAG（检索增强生成）应用，支持：

- 📄 **多格式文档处理**：PDF、Markdown、TXT
- 🔍 **RAG 检索增强生成**：智能分块、向量检索、混合搜索
- 💬 **多轮对话**：支持上下文连续对话
- 📝 **引用来源标注**：每个回答都有来源追溯
- 🔐 **JWT 认证**：安全的用户认证系统
- 🌊 **流式响应（SSE）**：实时流式输出
- 🛡️ **安全防护**：注入检测、内容审核
- 📊 **评测系统**：完整的评测脚本和指标

## 🏗️ 项目结构

```
py-26-project-final/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
│
├── src/
│   └── knowledge_assistant/
│       ├── main.py              # 应用入口
│       ├── config.py            # 配置管理
│       │
│       ├── api/                 # API 层
│       │   ├── app.py           # FastAPI 应用
│       │   ├── routers/         # 路由
│       │   │   ├── auth.py      # 认证
│       │   │   ├── ingest.py    # 文档摄取
│       │   │   ├── query.py     # 问答查询
│       │   │   └── health.py    # 健康检查
│       │   ├── schemas/         # 数据模型
│       │   └── dependencies/    # 依赖注入
│       │
│       ├── rag/                 # RAG 核心
│       │   ├── loader.py        # 文档加载
│       │   ├── chunker.py       # 智能分块
│       │   ├── embedder.py      # 向量嵌入
│       │   ├── index.py         # 向量索引
│       │   ├── retriever.py     # 检索器
│       │   └── generator.py     # 生成器
│       │
│       ├── llm/                 # LLM 客户端
│       │   ├── client.py        # API 客户端
│       │   └── prompts.py       # 提示词模板
│       │
│       ├── safety/              # 安全模块
│       │   ├── input_guard.py   # 输入过滤
│       │   └── output_guard.py  # 输出审核
│       │
│       └── evaluation/          # 评测模块
│           ├── dataset.py       # 数据集
│           ├── metrics.py       # 评测指标
│           └── runner.py        # 评测运行器
│
├── tests/                       # 测试
│   ├── test_api/
│   ├── test_rag/
│   └── test_safety/
│
├── data/
│   ├── sample_docs/             # 示例文档
│   └── eval_dataset/            # 评测数据
│
└── scripts/
    ├── run_dev.sh               # 开发启动
    ├── run_tests.sh             # 运行测试
    ├── run_eval.sh              # 运行评测
    └── docker_build.sh          # Docker 构建
```

## 🚀 快速开始

### 环境要求

- Python 3.11+
- pip 或 uv

### 1. 安装依赖

```bash
# 克隆项目
cd py-26-project-final

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -e ".[dev]"
```

### 2. 配置环境

```bash
# 复制环境变量示例（手动创建 .env 文件）
# 主要配置项：
# - LLM_PROVIDER: stub（测试）或 openai（生产）
# - OPENAI_API_KEY: 你的 OpenAI API 密钥
# - JWT_SECRET_KEY: JWT 密钥（生产环境必须更改）
```

### 3. 启动服务

```bash
# 开发模式
bash scripts/run_dev.sh

# 或直接使用 uvicorn
uvicorn knowledge_assistant.main:app --reload
```

### 4. 访问服务

- API 文档: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/healthz

## 📖 API 使用

### 认证

```bash
# 获取 Token（默认账号: admin/admin123）
curl -X POST "http://localhost:8000/api/v1/auth/token" \
  -d "username=admin&password=admin123"

# 返回
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### 上传文档

```bash
# 上传文件
curl -X POST "http://localhost:8000/api/v1/ingest/upload" \
  -H "Authorization: Bearer <token>" \
  -F "files=@document.pdf"

# 直接上传文本
curl -X POST "http://localhost:8000/api/v1/ingest/text?text=你的文本&source=来源" \
  -H "Authorization: Bearer <token>"
```

### 问答查询

```bash
# 普通查询
curl -X POST "http://localhost:8000/api/v1/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 RAG？"}'

# 流式查询（SSE）
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 RAG？"}'
```

## 🐳 Docker 部署

```bash
# 构建镜像
bash scripts/docker_build.sh

# 使用 docker-compose 启动
docker-compose up -d

# 查看日志
docker-compose logs -f app
```

## 🧪 测试

```bash
# 运行所有测试
bash scripts/run_tests.sh

# 运行特定测试
pytest tests/test_rag/ -v

# 生成覆盖率报告
pytest --cov=knowledge_assistant --cov-report=html
```

## 📊 评测

```bash
# 创建示例评测数据集
python -m knowledge_assistant.evaluation.cli create-dataset

# 运行评测（需要先启动服务）
bash scripts/run_eval.sh

# 查看评测结果
cat data/eval_results.json
```

## 🔧 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `APP_ENV` | 运行环境 | development |
| `LLM_PROVIDER` | LLM 提供商 | stub |
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `OPENAI_MODEL` | 模型名称 | gpt-4o-mini |
| `EMBEDDING_PROVIDER` | 嵌入模型提供商 | stub |
| `CHUNK_SIZE` | 分块大小 | 500 |
| `TOP_K` | 检索数量 | 5 |
| `JWT_SECRET_KEY` | JWT 密钥 | dev-secret-key |

## 📚 技术栈

- **Web 框架**: FastAPI
- **数据验证**: Pydantic
- **向量计算**: NumPy
- **HTTP 客户端**: httpx
- **认证**: python-jose, passlib
- **日志**: structlog
- **测试**: pytest, pytest-cov
- **容器化**: Docker

## 🎯 验收标准

- [x] 所有 API 正常工作
- [x] 测试覆盖率 > 80%
- [x] 评测脚本可运行
- [x] Docker 部署成功
- [x] README 完整

## 📝 License

MIT License

---

> 这是 Python 完整学习的终极项目（P26），综合运用了所有前置阶段的知识。


