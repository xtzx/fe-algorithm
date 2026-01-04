"""
提示词模板

管理 RAG 系统的提示词
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# 系统提示词集合
SYSTEM_PROMPTS = {
    "default": """你是一个知识库助手。请根据提供的上下文信息回答用户的问题。

规则:
1. 仅基于提供的上下文回答问题
2. 如果上下文中没有相关信息，请明确告知用户
3. 在回答中引用来源，使用 [1], [2] 等标记
4. 保持回答简洁、准确
5. 使用与用户相同的语言回答""",

    "professional": """你是一名专业的企业知识库助理。你的任务是基于提供的上下文信息，准确、专业地回答用户问题。

工作原则:
1. 严格基于上下文信息回答，不要编造或假设
2. 使用清晰的结构化格式组织回答
3. 在回答中引用来源 [1], [2] 等
4. 如果信息不足，明确说明并建议用户提供更多细节
5. 保持专业、客观的语气
6. 适当使用要点或编号列表使回答更易读""",

    "concise": """你是一个简洁的问答助手。

规则:
- 只回答问题本身，不添加额外信息
- 用最少的文字传达核心内容
- 引用格式: [1], [2]
- 信息不足时直接说"未找到相关信息\"""",

    "educational": """你是一个教育型知识助手，擅长用通俗易懂的方式解释复杂概念。

你的特点:
1. 先回答核心问题，再提供补充解释
2. 使用类比和例子帮助理解
3. 适当引用来源 [1], [2]
4. 可以提出相关的延伸问题供用户思考
5. 保持友好、鼓励的语气""",
}


@dataclass
class PromptTemplate:
    """
    提示词模板
    
    用于生成结构化的 LLM 提示词
    
    Usage:
        template = PromptTemplate(
            system="你是一个助手",
            user="{context}\\n\\n问题: {question}"
        )
        
        messages = template.format(
            context="一些背景信息",
            question="用户的问题"
        )
    """
    
    system: str
    user: str
    assistant_prefix: Optional[str] = None
    
    def format(self, **kwargs: Any) -> List[Dict[str, str]]:
        """
        格式化模板
        
        Args:
            **kwargs: 模板变量
        
        Returns:
            消息列表
        """
        messages = [{"role": "system", "content": self.system}]
        
        user_content = self.user.format(**kwargs)
        messages.append({"role": "user", "content": user_content})
        
        if self.assistant_prefix:
            messages.append({"role": "assistant", "content": self.assistant_prefix})
        
        return messages
    
    def with_history(
        self,
        history: List[Dict[str, str]],
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        """
        带历史记录格式化
        
        Args:
            history: 对话历史
            **kwargs: 模板变量
        
        Returns:
            消息列表
        """
        messages = [{"role": "system", "content": self.system}]
        
        # 添加历史记录（最多 6 条）
        for msg in history[-6:]:
            messages.append(msg)
        
        user_content = self.user.format(**kwargs)
        messages.append({"role": "user", "content": user_content})
        
        if self.assistant_prefix:
            messages.append({"role": "assistant", "content": self.assistant_prefix})
        
        return messages


# 预定义模板
RAG_TEMPLATE = PromptTemplate(
    system=SYSTEM_PROMPTS["default"],
    user="""上下文信息:
{context}

---
来源列表:
{sources}

---
用户问题: {question}

请根据上下文回答问题，并在适当位置标注引用来源。""",
)

RAG_PROFESSIONAL_TEMPLATE = PromptTemplate(
    system=SYSTEM_PROMPTS["professional"],
    user="""# 参考资料

{context}

# 来源索引

{sources}

# 用户问题

{question}

# 请回答""",
)

SUMMARY_TEMPLATE = PromptTemplate(
    system="你是一个文档摘要助手。请对提供的内容生成简洁的摘要。",
    user="""请为以下内容生成摘要:

{content}

要求:
- 摘要长度: {max_length} 字以内
- 保留关键信息
- 使用简洁的语言""",
)


