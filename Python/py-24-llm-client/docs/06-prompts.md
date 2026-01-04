# 提示工程

## 1. 提示模板

### 1.1 基础模板

```python
from llm_kit.prompts import PromptTemplate

template = PromptTemplate("""
You are a {role}.

Task: {task}

Input: {input}

Output:
""")

prompt = template.render(
    role="translator",
    task="Translate to Chinese",
    input="Hello world",
)
```

### 1.2 条件渲染

```python
template = PromptTemplate("""
{#if context}
Use this context:
{context}
{/if}

Question: {question}
""")

# 有上下文
prompt = template.render(question="...", context="...")

# 无上下文（条件块不渲染）
prompt = template.render(question="...")
```

## 2. RAG 提示模板

```python
from llm_kit.prompts import RAGPromptTemplate

template = RAGPromptTemplate()

prompt = template.render(
    question="What is RAG?",
    context=[
        {"content": "RAG stands for...", "source": "doc1.md"},
        {"content": "RAG is used...", "source": "doc2.md"},
    ],
)
```

### 2.1 带历史的 RAG

```python
prompt = template.render(
    question="Tell me more about it",
    context=[...],
    history="User: What is RAG?\nAssistant: RAG stands for...",
)
```

## 3. 对话历史管理

### 3.1 ChatHistory

```python
from llm_kit.prompts import ChatHistory

history = ChatHistory(
    system_prompt="You are a helpful assistant.",
    max_messages=20,      # 最多保留 20 条
    max_tokens=4000,      # 最多 4000 tokens
)

# 添加消息
history.add_user("Hello!")
history.add_assistant("Hi! How can I help?")
history.add_user("What is Python?")

# 获取消息
messages = history.get_messages()
# [
#   {"role": "system", "content": "..."},
#   {"role": "user", "content": "Hello!"},
#   {"role": "assistant", "content": "Hi! ..."},
#   {"role": "user", "content": "What is Python?"},
# ]
```

### 3.2 Token 计数

```python
# 当前 token 数
print(history.token_count)

# 自动修剪（超过 max_tokens 时移除旧消息）
history.add_user("Very long message...")
# 旧消息被自动移除以保持 token 限制
```

## 4. 会话管理

```python
from llm_kit.prompts.chat import ConversationManager

manager = ConversationManager(
    default_system_prompt="You are helpful."
)

# 创建会话
session_id = manager.create_session()

# 添加消息
manager.add_message(session_id, "user", "Hello!")
manager.add_message(session_id, "assistant", "Hi!")

# 获取消息
messages = manager.get_messages(session_id)

# 列出会话
sessions = manager.list_sessions()

# 删除会话
manager.delete_session(session_id)
```

## 5. 预定义模板

```python
from llm_kit.prompts.template import (
    SUMMARY_TEMPLATE,
    TRANSLATION_TEMPLATE,
    QA_TEMPLATE,
    CODE_REVIEW_TEMPLATE,
)

# 摘要
prompt = SUMMARY_TEMPLATE.render(text="Long article...")

# 翻译
prompt = TRANSLATION_TEMPLATE.render(
    text="Hello",
    target_language="Chinese",
)

# 代码审查
prompt = CODE_REVIEW_TEMPLATE.render(
    language="python",
    code="def add(a, b): return a + b",
)
```

## 6. 提示工程技巧

### 6.1 角色设定

```
You are an expert Python developer with 10 years of experience.
You write clean, efficient, and well-documented code.
```

### 6.2 输出格式指定

```
Output your answer in the following JSON format:
{
    "answer": "your answer here",
    "confidence": 0.9,
    "sources": ["source1", "source2"]
}
```

### 6.3 思维链（CoT）

```
Think step by step:
1. First, analyze the problem
2. Then, identify key factors
3. Finally, provide your conclusion

Let's solve this step by step:
```

### 6.4 Few-shot 示例

```
Example 1:
Input: "2+2"
Output: 4

Example 2:
Input: "3*4"
Output: 12

Now solve:
Input: "5+6"
Output:
```

## 7. 最佳实践

1. **明确指令**：清楚说明期望的输出
2. **提供上下文**：给出足够的背景信息
3. **使用分隔符**：区分不同部分
4. **限制输出**：指定长度、格式
5. **迭代优化**：测试并改进提示


