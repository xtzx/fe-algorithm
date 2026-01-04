"""
提示模板

特性:
- 变量替换
- 条件渲染
- RAG 上下文注入
"""

import re
from dataclasses import dataclass
from string import Template
from typing import Any, Dict, List, Optional


class PromptTemplate:
    """
    提示模板
    
    支持变量替换和简单条件渲染
    
    Usage:
        template = PromptTemplate('''
        You are a helpful assistant.
        
        {#if context}
        Use the following context:
        {context}
        {/if}
        
        User question: {question}
        ''')
        
        prompt = template.render(
            question="What is RAG?",
            context="RAG stands for...",
        )
    """

    def __init__(self, template: str):
        self.template = template.strip()
        self._compiled = None

    def render(self, **kwargs) -> str:
        """
        渲染模板
        
        Args:
            **kwargs: 模板变量
        
        Returns:
            渲染后的字符串
        """
        result = self.template
        
        # 处理条件块 {#if var}...{/if}
        result = self._process_conditionals(result, kwargs)
        
        # 变量替换 {var}
        result = self._process_variables(result, kwargs)
        
        # 清理多余空行
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
        
        return result.strip()

    def _process_conditionals(self, template: str, variables: Dict[str, Any]) -> str:
        """处理条件块"""
        pattern = r'\{#if\s+(\w+)\}(.*?)\{/if\}'
        
        def replace_conditional(match):
            var_name = match.group(1)
            content = match.group(2)
            
            # 检查变量是否存在且非空
            if var_name in variables and variables[var_name]:
                return content
            return ""
        
        return re.sub(pattern, replace_conditional, template, flags=re.DOTALL)

    def _process_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """处理变量替换"""
        pattern = r'\{(\w+)\}'
        
        def replace_variable(match):
            var_name = match.group(1)
            if var_name in variables:
                return str(variables[var_name])
            return match.group(0)  # 保持原样
        
        return re.sub(pattern, replace_variable, template)

    def get_variables(self) -> List[str]:
        """获取模板中的变量名"""
        pattern = r'\{(\w+)\}'
        matches = re.findall(pattern, self.template)
        return list(set(matches))

    def __repr__(self) -> str:
        return f"PromptTemplate({self.template[:50]}...)"


class RAGPromptTemplate(PromptTemplate):
    """
    RAG 提示模板
    
    专门用于 RAG 场景的模板，包含上下文和引用处理
    
    Usage:
        template = RAGPromptTemplate()
        
        prompt = template.render(
            question="What is RAG?",
            context=[
                {"content": "RAG stands for...", "source": "doc1.md"},
                {"content": "RAG is used for...", "source": "doc2.md"},
            ],
        )
    """

    DEFAULT_TEMPLATE = """You are a helpful assistant. Answer the user's question based on the provided context.

## Context

{context}

## Instructions

- Answer based ONLY on the provided context
- If the context doesn't contain relevant information, say "I don't have enough information to answer this question"
- Cite your sources using [Source: filename] format
- Be concise and accurate

## Question

{question}

## Answer"""

    DEFAULT_TEMPLATE_WITH_HISTORY = """You are a helpful assistant. Answer the user's question based on the provided context and conversation history.

## Context

{context}

## Conversation History

{history}

## Instructions

- Answer based on the provided context and conversation history
- If the context doesn't contain relevant information, say "I don't have enough information to answer this question"
- Cite your sources using [Source: filename] format
- Be concise and accurate

## Current Question

{question}

## Answer"""

    def __init__(
        self,
        template: Optional[str] = None,
        include_sources: bool = True,
        max_context_items: int = 5,
    ):
        self.include_sources = include_sources
        self.max_context_items = max_context_items
        
        if template:
            super().__init__(template)
        else:
            super().__init__(self.DEFAULT_TEMPLATE)

    def render(
        self,
        question: str,
        context: List[Dict[str, Any]] | str,
        history: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        渲染 RAG 提示
        
        Args:
            question: 用户问题
            context: 上下文（列表或字符串）
            history: 对话历史
            **kwargs: 其他变量
        
        Returns:
            渲染后的提示
        """
        # 处理上下文
        if isinstance(context, list):
            context_str = self._format_context(context[:self.max_context_items])
        else:
            context_str = context
        
        # 使用带历史的模板
        if history:
            template = PromptTemplate(self.DEFAULT_TEMPLATE_WITH_HISTORY)
            return template.render(
                question=question,
                context=context_str,
                history=history,
                **kwargs,
            )
        
        return super().render(
            question=question,
            context=context_str,
            **kwargs,
        )

    def _format_context(self, context_items: List[Dict[str, Any]]) -> str:
        """格式化上下文"""
        parts = []
        
        for i, item in enumerate(context_items, 1):
            content = item.get("content", "")
            source = item.get("source", "unknown")
            
            if self.include_sources:
                parts.append(f"[{i}] Source: {source}\n{content}")
            else:
                parts.append(content)
        
        return "\n\n---\n\n".join(parts)


# 预定义模板
SUMMARY_TEMPLATE = PromptTemplate("""
Summarize the following text in a concise manner:

Text:
{text}

Summary:
""")

TRANSLATION_TEMPLATE = PromptTemplate("""
Translate the following text to {target_language}:

Text:
{text}

Translation:
""")

QA_TEMPLATE = PromptTemplate("""
Answer the following question based on your knowledge:

Question: {question}

Answer:
""")

CODE_REVIEW_TEMPLATE = PromptTemplate("""
Review the following code and provide feedback:

```{language}
{code}
```

Please analyze:
1. Code quality
2. Potential bugs
3. Performance issues
4. Suggestions for improvement

Review:
""")


