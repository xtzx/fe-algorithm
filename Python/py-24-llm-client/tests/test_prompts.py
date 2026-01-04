"""
提示模块测试
"""

import pytest

from llm_kit.prompts import ChatHistory, Message, PromptTemplate, RAGPromptTemplate


class TestPromptTemplate:
    """提示模板测试"""

    def test_basic_render(self):
        """测试基础渲染"""
        template = PromptTemplate("Hello, {name}!")
        result = template.render(name="World")
        
        assert result == "Hello, World!"

    def test_multiple_variables(self):
        """测试多变量"""
        template = PromptTemplate("{greeting}, {name}! How is {thing}?")
        result = template.render(greeting="Hi", name="Alice", thing="the weather")
        
        assert result == "Hi, Alice! How is the weather?"

    def test_conditional_true(self):
        """测试条件渲染（真）"""
        template = PromptTemplate("{#if context}Context: {context}{/if}")
        result = template.render(context="Some context")
        
        assert "Some context" in result

    def test_conditional_false(self):
        """测试条件渲染（假）"""
        template = PromptTemplate("{#if context}Context: {context}{/if}Question: {q}")
        result = template.render(q="What?", context=None)
        
        assert "Context" not in result
        assert "Question: What?" in result

    def test_get_variables(self):
        """测试获取变量名"""
        template = PromptTemplate("Hello {name}, welcome to {place}!")
        variables = template.get_variables()
        
        assert "name" in variables
        assert "place" in variables


class TestRAGPromptTemplate:
    """RAG 提示模板测试"""

    def test_with_list_context(self):
        """测试列表上下文"""
        template = RAGPromptTemplate()
        
        result = template.render(
            question="What is RAG?",
            context=[
                {"content": "RAG is...", "source": "doc1.md"},
                {"content": "RAG stands for...", "source": "doc2.md"},
            ],
        )
        
        assert "What is RAG?" in result
        assert "RAG is..." in result
        assert "doc1.md" in result

    def test_with_string_context(self):
        """测试字符串上下文"""
        template = RAGPromptTemplate()
        
        result = template.render(
            question="What is RAG?",
            context="RAG is a technique...",
        )
        
        assert "What is RAG?" in result
        assert "RAG is a technique..." in result

    def test_with_history(self):
        """测试带历史"""
        template = RAGPromptTemplate()
        
        result = template.render(
            question="Tell me more",
            context=[{"content": "Details...", "source": "doc.md"}],
            history="User: What is RAG?\nAssistant: RAG is...",
        )
        
        assert "Tell me more" in result
        assert "What is RAG?" in result


class TestChatHistory:
    """对话历史测试"""

    def test_add_messages(self):
        """测试添加消息"""
        history = ChatHistory()
        
        history.add_user("Hello!")
        history.add_assistant("Hi there!")
        
        assert len(history) == 2

    def test_get_messages(self):
        """测试获取消息"""
        history = ChatHistory(system_prompt="You are helpful.")
        
        history.add_user("Hello!")
        history.add_assistant("Hi!")
        
        messages = history.get_messages()
        
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_max_messages_limit(self):
        """测试消息数量限制"""
        history = ChatHistory(max_messages=2)
        
        history.add_user("Message 1")
        history.add_assistant("Response 1")
        history.add_user("Message 2")
        history.add_assistant("Response 2")
        
        assert len(history) == 2

    def test_clear(self):
        """测试清除历史"""
        history = ChatHistory(system_prompt="System")
        
        history.add_user("Hello!")
        history.clear()
        
        assert len(history) == 0
        # 系统提示保留
        messages = history.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_format_history(self):
        """测试格式化历史"""
        history = ChatHistory()
        
        history.add_user("Hello!")
        history.add_assistant("Hi!")
        
        formatted = history.format_history()
        
        assert "User:" in formatted
        assert "Assistant:" in formatted


class TestMessage:
    """消息测试"""

    def test_to_dict(self):
        """测试转换为字典"""
        msg = Message.user("Hello!")
        d = msg.to_dict()
        
        assert d["role"] == "user"
        assert d["content"] == "Hello!"

    def test_factory_methods(self):
        """测试工厂方法"""
        system = Message.system("You are helpful")
        user = Message.user("Hello")
        assistant = Message.assistant("Hi")
        
        assert system.role.value == "system"
        assert user.role.value == "user"
        assert assistant.role.value == "assistant"


