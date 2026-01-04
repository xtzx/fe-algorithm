"""
测试配置

提供通用的测试 fixtures
"""

import pytest
from fastapi.testclient import TestClient

from knowledge_assistant.api.app import create_app
from knowledge_assistant.api.dependencies.services import reset_services
from knowledge_assistant.config import Settings


@pytest.fixture(scope="session")
def test_settings():
    """测试配置"""
    return Settings(
        app_env="testing",
        debug=True,
        llm_provider="stub",
        embedding_provider="stub",
        jwt_secret_key="test-secret-key",
    )


@pytest.fixture
def app():
    """创建测试应用"""
    reset_services()
    return create_app()


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    """获取认证头"""
    # 使用默认管理员账号登录
    response = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_document():
    """示例文档"""
    return """# RAG 简介

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合检索和生成的 AI 技术。

## 主要特点

1. **减少幻觉**: 通过检索真实数据减少模型编造内容
2. **知识可更新**: 可以随时更新知识库，无需重新训练模型
3. **来源可追溯**: 可以提供回答的来源引用

## 工作流程

1. 用户提问
2. 检索相关文档
3. 将文档作为上下文
4. LLM 生成回答

这是一个强大的技术，广泛应用于知识库问答、客服系统等场景。
"""


@pytest.fixture
def sample_chunks():
    """示例分块"""
    from knowledge_assistant.rag.chunker import Chunk
    
    return [
        Chunk(
            content="RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合检索和生成的 AI 技术。",
            metadata={"doc_id": "test_doc", "source": "test.md"},
            chunk_index=0,
        ),
        Chunk(
            content="减少幻觉: 通过检索真实数据减少模型编造内容。知识可更新: 可以随时更新知识库。",
            metadata={"doc_id": "test_doc", "source": "test.md"},
            chunk_index=1,
        ),
        Chunk(
            content="工作流程: 用户提问 -> 检索相关文档 -> 将文档作为上下文 -> LLM 生成回答。",
            metadata={"doc_id": "test_doc", "source": "test.md"},
            chunk_index=2,
        ),
    ]


