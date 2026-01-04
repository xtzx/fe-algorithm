"""
对话管理

特性:
- 消息历史
- 上下文窗口管理
- Token 计数
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()


class Role(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """消息"""
    role: Role | str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        """转换为 API 格式"""
        role = self.role.value if isinstance(self.role, Role) else self.role
        return {"role": role, "content": self.content}

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role=Role.ASSISTANT, content=content)


class ChatHistory:
    """
    对话历史管理
    
    特性:
    - 消息存储
    - 上下文窗口（限制 token 数）
    - 系统提示管理
    
    Usage:
        history = ChatHistory(system_prompt="You are a helpful assistant")
        
        history.add_user("Hello!")
        history.add_assistant("Hi! How can I help?")
        
        messages = history.get_messages()
        # [{"role": "system", "content": "..."}, 
        #  {"role": "user", "content": "Hello!"},
        #  {"role": "assistant", "content": "Hi! ..."}]
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self._messages: List[Message] = []
        self._token_counter = None
        
        # 尝试加载 tiktoken
        try:
            import tiktoken
            self._encoding = tiktoken.get_encoding("cl100k_base")
            self._token_counter = self._count_tokens_tiktoken
        except ImportError:
            self._token_counter = self._count_tokens_simple

    def add(self, role: Role | str, content: str):
        """添加消息"""
        message = Message(role=role, content=content)
        self._messages.append(message)
        self._trim_history()

    def add_user(self, content: str):
        """添加用户消息"""
        self.add(Role.USER, content)

    def add_assistant(self, content: str):
        """添加助手消息"""
        self.add(Role.ASSISTANT, content)

    def get_messages(self) -> List[Dict[str, str]]:
        """获取消息列表（API 格式）"""
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        for msg in self._messages:
            messages.append(msg.to_dict())
        
        return messages

    def get_last_n(self, n: int) -> List[Message]:
        """获取最近 N 条消息"""
        return self._messages[-n:] if n > 0 else []

    def clear(self):
        """清除历史（保留系统提示）"""
        self._messages = []

    def _trim_history(self):
        """修剪历史以符合限制"""
        # 按消息数限制
        if self.max_messages and len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
        
        # 按 token 数限制
        if self.max_tokens:
            while self._count_total_tokens() > self.max_tokens and len(self._messages) > 1:
                self._messages.pop(0)

    def _count_total_tokens(self) -> int:
        """计算总 token 数"""
        total = 0
        
        if self.system_prompt:
            total += self._token_counter(self.system_prompt)
        
        for msg in self._messages:
            total += self._token_counter(msg.content)
        
        return total

    def _count_tokens_tiktoken(self, text: str) -> int:
        """使用 tiktoken 计数"""
        return len(self._encoding.encode(text))

    def _count_tokens_simple(self, text: str) -> int:
        """简单计数（近似）"""
        return len(text) // 4

    def format_history(self, separator: str = "\n") -> str:
        """格式化历史为字符串"""
        lines = []
        for msg in self._messages:
            role = msg.role.value if isinstance(msg.role, Role) else msg.role
            lines.append(f"{role.capitalize()}: {msg.content}")
        return separator.join(lines)

    @property
    def token_count(self) -> int:
        """当前 token 数"""
        return self._count_total_tokens()

    def __len__(self) -> int:
        return len(self._messages)


class ConversationManager:
    """
    对话管理器
    
    管理多个对话会话
    
    Usage:
        manager = ConversationManager()
        
        # 创建会话
        session_id = manager.create_session(system_prompt="...")
        
        # 添加消息
        manager.add_message(session_id, "user", "Hello!")
        
        # 获取消息
        messages = manager.get_messages(session_id)
    """

    def __init__(self, default_system_prompt: Optional[str] = None):
        self.default_system_prompt = default_system_prompt
        self._sessions: Dict[str, ChatHistory] = {}

    def create_session(
        self,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """创建会话"""
        import uuid
        
        session_id = session_id or str(uuid.uuid4())
        prompt = system_prompt or self.default_system_prompt
        
        self._sessions[session_id] = ChatHistory(system_prompt=prompt, **kwargs)
        
        logger.info("session_created", session_id=session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[ChatHistory]:
        """获取会话"""
        return self._sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str):
        """添加消息"""
        session = self._sessions.get(session_id)
        if session:
            session.add(role, content)

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        """获取消息"""
        session = self._sessions.get(session_id)
        if session:
            return session.get_messages()
        return []

    def delete_session(self, session_id: str):
        """删除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("session_deleted", session_id=session_id)

    def list_sessions(self) -> List[str]:
        """列出所有会话"""
        return list(self._sessions.keys())


