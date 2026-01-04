"""
知识库助手 - AI 知识库问答系统

提供:
- 文档上传与处理（PDF、Markdown、TXT）
- RAG 检索增强生成
- 多轮对话
- 引用来源标注
- 安全防护
"""

__version__ = "1.0.0"

from knowledge_assistant.config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]


