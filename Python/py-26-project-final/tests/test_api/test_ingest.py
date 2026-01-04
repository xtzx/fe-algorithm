"""
文档摄取测试
"""

import io
import pytest


def test_ingest_text(client, auth_headers):
    """测试文本摄取"""
    response = client.post(
        "/api/v1/ingest/text",
        params={"text": "这是一段测试文本，用于测试知识库摄取功能。", "source": "test"},
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert len(data["documents"]) == 1


def test_ingest_requires_auth(client):
    """测试摄取需要认证"""
    response = client.post(
        "/api/v1/ingest/text",
        params={"text": "test", "source": "test"},
    )
    
    assert response.status_code == 401


def test_list_documents(client, auth_headers):
    """测试列出文档"""
    response = client.get(
        "/api/v1/ingest/documents",
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data


def test_upload_file(client, auth_headers, sample_document):
    """测试上传文件"""
    # 创建模拟文件
    file_content = sample_document.encode("utf-8")
    files = {"files": ("test.md", io.BytesIO(file_content), "text/markdown")}
    
    response = client.post(
        "/api/v1/ingest/upload",
        files=files,
        headers=auth_headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"


