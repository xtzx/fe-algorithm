"""
健康检查测试
"""

import pytest


def test_health_check(client):
    """测试健康检查端点"""
    response = client.get("/healthz")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_readiness_check(client):
    """测试就绪检查端点"""
    response = client.get("/readyz")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data


def test_app_info(client):
    """测试应用信息端点"""
    response = client.get("/info")
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "知识库助手 API"
    assert "version" in data


