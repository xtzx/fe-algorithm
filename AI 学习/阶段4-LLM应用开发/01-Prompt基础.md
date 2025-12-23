# 📝 01 - Prompt Engineering 基础

> 掌握与大模型有效沟通的核心技巧

---

## 什么是 Prompt Engineering

```
Prompt Engineering 是设计和优化输入提示词的技术，
目标是让大模型产生更准确、更有用的输出。

核心要素：
1. 清晰的指令
2. 适当的上下文
3. 输出格式要求
4. 示例引导
```

---

## API 基础调用

### OpenAI API

```python
from openai import OpenAI

client = OpenAI()

def chat(prompt: str, system: str = "You are a helpful assistant."):
    """基础对话函数"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content

# 使用
result = chat("什么是机器学习？")
print(result)
```

### Anthropic API

```python
import anthropic

client = anthropic.Anthropic()

def chat_claude(prompt: str, system: str = "You are a helpful assistant."):
    """Claude 对话函数"""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text

result = chat_claude("什么是机器学习？")
print(result)
```

### 本地模型（Ollama）

```python
import requests

def chat_ollama(prompt: str, model: str = "llama3.1:8b"):
    """Ollama 本地模型对话"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

result = chat_ollama("什么是机器学习？")
print(result)
```

---

## 指令式 Prompt

### 基本原则

```
1. 明确任务：告诉模型要做什么
2. 提供上下文：给出必要的背景信息
3. 指定格式：说明期望的输出形式
4. 设定约束：限制输出的范围或风格
```

### 示例：文本摘要

```python
# ❌ 不好的 Prompt
prompt = "总结一下这段文字"

# ✅ 好的 Prompt
prompt = """
请将以下文章总结为 3 个要点，每个要点不超过 20 字。

文章内容：
{article_text}

输出格式：
1. [要点1]
2. [要点2]
3. [要点3]
"""
```

### 示例：代码生成

```python
prompt = """
任务：编写一个 Python 函数

要求：
1. 函数名：calculate_statistics
2. 输入：一个数字列表
3. 输出：包含 mean, median, std 的字典
4. 使用 numpy 库

请只输出代码，不需要解释。
"""

result = chat(prompt)
print(result)
```

### 示例：文本分类

```python
prompt = """
你是一个情感分析专家。请分析以下评论的情感倾向。

评论："{review_text}"

请只回复以下三个选项之一：
- 正面
- 负面
- 中性
"""
```

---

## Few-shot Learning（示例学习）

### 原理

```
通过在 prompt 中提供几个示例，
让模型理解任务模式和期望输出格式。

适用场景：
- 特定格式要求
- 专业领域任务
- 复杂分类任务
```

### 示例：命名实体识别

```python
few_shot_prompt = """
任务：从句子中提取人名、地点和组织。

示例：
输入：马云在杭州创立了阿里巴巴。
输出：{"人名": ["马云"], "地点": ["杭州"], "组织": ["阿里巴巴"]}

输入：乔布斯在加州成立了苹果公司。
输出：{"人名": ["乔布斯"], "地点": ["加州"], "组织": ["苹果公司"]}

输入：张三去北京参加了华为的面试。
输出：{"人名": ["张三"], "地点": ["北京"], "组织": ["华为"]}

现在请处理：
输入：{user_input}
输出：
"""

user_input = "李彦宏在北京创办了百度，后来张一鸣在北京创立了字节跳动。"
result = chat(few_shot_prompt.format(user_input=user_input))
print(result)
# 输出：{"人名": ["李彦宏", "张一鸣"], "地点": ["北京"], "组织": ["百度", "字节跳动"]}
```

### 示例：风格转换

```python
style_prompt = """
将正式文本转换为口语化表达。

示例：
正式：本次会议的主要议题是讨论下季度的销售策略。
口语：这次开会主要聊聊下个季度怎么卖货。

正式：请您务必在截止日期前提交相关材料。
口语：记得在截止日前把材料交了哈。

正式：{formal_text}
口语：
"""
```

### 示例数量选择

```python
"""
Few-shot 示例数量建议：
- 简单任务：1-2 个示例
- 中等复杂度：3-5 个示例
- 复杂任务：5-10 个示例

注意：
- 示例要有代表性
- 覆盖边界情况
- 保持格式一致
"""
```

---

## Chain of Thought（思维链）

### 原理

```
让模型展示推理过程，逐步解决问题。
通过"让我们一步一步思考"引导模型进行推理。

优势：
- 提高复杂推理准确性
- 便于调试和理解
- 减少幻觉
```

### Zero-shot CoT

```python
# 简单加一句"让我们一步一步思考"
cot_prompt = """
问题：一个商店有 23 个苹果，卖掉了 17 个，又进货了 12 个，
然后又卖掉了 6 个。现在商店有多少个苹果？

让我们一步一步思考：
"""

result = chat(cot_prompt)
print(result)
"""
输出：
1. 初始苹果数：23 个
2. 卖掉 17 个后：23 - 17 = 6 个
3. 进货 12 个后：6 + 12 = 18 个
4. 再卖掉 6 个后：18 - 6 = 12 个

答案：商店现在有 12 个苹果。
"""
```

### Few-shot CoT

```python
cot_few_shot = """
问题：小明有 5 本书，小红给了他 3 本，他又买了 2 本。小明现在有多少本书？

推理过程：
1. 小明初始有 5 本书
2. 小红给了 3 本：5 + 3 = 8 本
3. 又买了 2 本：8 + 2 = 10 本
答案：10 本

问题：一个班有 30 个学生，有 12 个男生，后来转来了 5 个女生，又走了 3 个男生。
现在班里有多少个学生？男女各多少？

推理过程：
"""
```

### 数学问题示例

```python
math_cot_prompt = """
你是一个数学老师，请详细展示解题步骤。

问题：
一列火车从 A 站出发，速度为 80 km/h，2 小时后另一列火车从同一站出发，
速度为 120 km/h，追赶前一列火车。问第二列火车需要多少时间才能追上第一列？

请按以下步骤解答：
1. 理解问题
2. 设定变量
3. 建立方程
4. 求解
5. 验证答案
"""

result = chat(math_cot_prompt)
print(result)
```

---

## 系统提示词设计

### 角色设定

```python
system_prompts = {
    "translator": """
你是一位专业的中英文翻译。
- 保持原文的语气和风格
- 专业术语使用行业标准译法
- 如遇歧义，提供多个翻译选项
""",

    "code_reviewer": """
你是一位资深软件工程师，负责代码审查。
- 关注代码质量、性能和安全性
- 指出潜在问题并给出改进建议
- 使用友好但专业的语气
""",

    "writing_assistant": """
你是一位写作助手。
- 帮助用户改进文章结构和表达
- 保持用户的写作风格
- 提供具体的修改建议而不是重写
"""
}
```

### 约束条件

```python
constrained_system = """
你是一个客服机器人，请遵循以下规则：

【必须】
- 始终保持礼貌和专业
- 只回答与公司产品相关的问题
- 提供准确的信息

【禁止】
- 不讨论政治、宗教等敏感话题
- 不提供医疗、法律建议
- 不泄露内部信息

【格式】
- 回答控制在 200 字以内
- 如需更多信息，提供联系方式
"""
```

### 输出格式控制

```python
format_system = """
你是一个数据分析助手。

对于每个分析请求，请按以下格式输出：

## 分析结果

### 关键发现
- [发现1]
- [发现2]

### 数据支持
| 指标 | 数值 | 变化 |
|------|------|------|
| ... | ... | ... |

### 建议
1. [建议1]
2. [建议2]

### 置信度
[高/中/低] - [原因]
"""
```

---

## 练习题

### 练习 1：优化 Prompt

```python
# 原始 Prompt
original = "帮我写一封邮件"

# 任务：改写为一个清晰、有效的 Prompt
# 提示：包含目的、收件人、语气、长度等要求
```

### 练习 2：Few-shot 设计

```python
# 任务：设计一个 Few-shot prompt，让模型将用户输入转换为 SQL 查询
#
# 表结构：
# - users(id, name, age, city)
# - orders(id, user_id, product, amount, created_at)
#
# 示例：
# 输入："找出北京的所有用户"
# 输出：SELECT * FROM users WHERE city = '北京'
```

### 练习 3：CoT 应用

```python
# 任务：设计一个 CoT prompt，让模型解决以下逻辑推理问题
#
# 问题：
# A、B、C 三人中有一人是医生，一人是教师，一人是律师。
# 已知：
# 1. A 比医生年龄大
# 2. 教师比 C 年龄小
# 3. B 和教师年龄不同
# 问：A、B、C 分别是什么职业？
```

---

## 小结

```
本节要点：
1. API 调用：熟悉 OpenAI / Anthropic / Ollama 的基本用法
2. 指令式 Prompt：明确任务、上下文、格式、约束
3. Few-shot：通过示例引导模型理解任务
4. CoT：展示推理过程，提高复杂任务准确性
5. 系统提示词：角色设定、约束条件、输出格式
```

---

## ➡️ 下一步

继续 [02-Prompt进阶.md](./02-Prompt进阶.md)

