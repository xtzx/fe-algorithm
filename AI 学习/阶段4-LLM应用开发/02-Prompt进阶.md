# 🎯 02 - Prompt Engineering 进阶

> 高级 Prompt 技术与结构化输出

---

## ReAct（Reasoning + Acting）

### 原理

```
ReAct 将推理（Reasoning）和行动（Acting）交替进行：
1. Thought：模型思考当前情况
2. Action：决定执行什么操作
3. Observation：获取操作结果
4. 重复直到完成任务

适用场景：
- 需要外部工具的任务
- 多步骤复杂问题
- Agent 设计
```

### 实现示例

```python
from openai import OpenAI

client = OpenAI()

# 模拟工具
def search(query: str) -> str:
    """模拟搜索工具"""
    fake_data = {
        "python release date": "Python was first released in 1991",
        "python creator": "Python was created by Guido van Rossum",
        "largest planet": "Jupiter is the largest planet in our solar system"
    }
    for key, value in fake_data.items():
        if key in query.lower():
            return value
    return "No relevant information found."

def calculator(expression: str) -> str:
    """计算器工具"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

# ReAct Prompt
react_prompt = """
你是一个能够使用工具的助手。

可用工具：
- search(query): 搜索信息
- calculator(expression): 数学计算

请按以下格式回答问题：

Thought: [你的思考过程]
Action: [工具名称](参数)
Observation: [工具返回结果，由系统填充]
... (可以重复多次)
Thought: [最终思考]
Answer: [最终答案]

问题：{question}
"""

def react_agent(question: str, max_iterations: int = 5):
    """ReAct Agent 实现"""
    tools = {"search": search, "calculator": calculator}

    prompt = react_prompt.format(question=question)

    for i in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stop=["Observation:"]
        )

        output = response.choices[0].message.content
        prompt += output

        # 检查是否有最终答案
        if "Answer:" in output:
            return output.split("Answer:")[-1].strip()

        # 解析 Action
        if "Action:" in output:
            action_line = [l for l in output.split('\n') if "Action:" in l][-1]
            # 解析工具调用
            import re
            match = re.search(r'Action:\s*(\w+)\((.*?)\)', action_line)
            if match:
                tool_name, tool_arg = match.groups()
                tool_arg = tool_arg.strip('"\'')

                if tool_name in tools:
                    observation = tools[tool_name](tool_arg)
                    prompt += f"\nObservation: {observation}\n"

    return "Unable to find answer"

# 测试
result = react_agent("Python 是什么时候发布的？距今多少年了？（假设现在是 2024 年）")
print(result)
```

---

## Tree of Thoughts（思维树）

### 原理

```
ToT 将问题分解为多个思考路径，评估每条路径的前景，
选择最有希望的路径继续探索。

步骤：
1. 生成多个候选思路
2. 评估每个思路的质量
3. 选择最佳思路继续
4. 回溯或扩展
```

### 实现示例

```python
def tree_of_thoughts(problem: str, num_thoughts: int = 3, depth: int = 3):
    """简化版 Tree of Thoughts"""

    # Step 1: 生成初始思路
    generate_prompt = f"""
问题：{problem}

请生成 {num_thoughts} 个不同的解决思路（只给出思路方向，不要完整解答）：
"""

    thoughts_response = chat(generate_prompt)

    # Step 2: 评估每个思路
    evaluate_prompt = f"""
问题：{problem}

候选思路：
{thoughts_response}

请评估每个思路的可行性（1-10 分），并选择最佳思路。
输出格式：
思路 X：评分 Y/10，理由：...
最佳思路：X
"""

    evaluation = chat(evaluate_prompt)

    # Step 3: 展开最佳思路
    expand_prompt = f"""
问题：{problem}

选定思路：
{evaluation}

请沿着最佳思路，继续深入分析并给出完整解答。
"""

    final_answer = chat(expand_prompt)
    return final_answer

# 使用示例
problem = """
一家创业公司想要开发一款 AI 产品，但预算有限。
请分析应该优先考虑哪些功能，如何分配资源。
"""
result = tree_of_thoughts(problem)
print(result)
```

---

## Self-Consistency（自洽性）

### 原理

```
多次采样不同的推理路径，然后对最终答案进行投票。
通过集成多个推理过程来提高准确性。

步骤：
1. 同一问题多次采样（高温度）
2. 提取每次的最终答案
3. 投票选择最频繁的答案
```

### 实现示例

```python
from collections import Counter

def self_consistency(question: str, num_samples: int = 5):
    """Self-Consistency 实现"""

    prompt = f"""
{question}

请一步一步思考，然后给出答案。
在最后用 "答案：X" 的格式给出最终答案。
"""

    answers = []

    for _ in range(num_samples):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # 较高温度以获得多样性
            max_tokens=500
        )

        output = response.choices[0].message.content

        # 提取答案
        if "答案：" in output:
            answer = output.split("答案：")[-1].strip().split()[0]
            answers.append(answer)

    # 投票
    if answers:
        most_common = Counter(answers).most_common(1)[0]
        return {
            "answer": most_common[0],
            "confidence": most_common[1] / len(answers),
            "all_answers": answers
        }

    return None

# 测试
result = self_consistency(
    "一个房间里有 3 个开关，分别控制另一个房间的 3 盏灯。"
    "你只能进入有灯的房间一次。如何确定每个开关控制哪盏灯？"
)
print(result)
```

---

## 结构化输出

### JSON 输出

```python
json_prompt = """
分析以下产品评论，以 JSON 格式输出分析结果。

评论："{review}"

请输出以下 JSON 格式：
{{
    "sentiment": "positive" | "negative" | "neutral",
    "score": 1-5,
    "keywords": ["关键词1", "关键词2"],
    "summary": "一句话总结",
    "aspects": {{
        "quality": "正面/负面/未提及",
        "price": "正面/负面/未提及",
        "service": "正面/负面/未提及"
    }}
}}

只输出 JSON，不要其他内容。
"""

review = "这个产品质量很好，但是价格有点贵。客服态度也不错。"
result = chat(json_prompt.format(review=review))
print(result)

# 解析 JSON
import json
data = json.loads(result)
print(f"情感: {data['sentiment']}, 评分: {data['score']}")
```

### 使用 Pydantic 验证

```python
from pydantic import BaseModel, Field
from typing import List, Literal
import json

class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    score: int = Field(ge=1, le=5)
    keywords: List[str]
    summary: str

def analyze_review_structured(review: str) -> ReviewAnalysis:
    prompt = f"""
分析以下产品评论，以 JSON 格式输出。

评论："{review}"

输出格式：
{{
    "sentiment": "positive" | "negative" | "neutral",
    "score": 1-5 的整数,
    "keywords": ["关键词列表"],
    "summary": "一句话总结"
}}
"""

    result = chat(prompt)

    # 清理可能的 markdown 标记
    result = result.strip()
    if result.startswith("```"):
        result = result.split("```")[1]
        if result.startswith("json"):
            result = result[4:]

    data = json.loads(result)
    return ReviewAnalysis(**data)

# 使用
analysis = analyze_review_structured("产品超棒！物美价廉，下次还买！")
print(analysis.model_dump())
```

### OpenAI JSON Mode

```python
# OpenAI 原生 JSON 模式
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "输出 JSON 格式的数据"},
        {"role": "user", "content": "列出 3 个编程语言及其特点"}
    ],
    response_format={"type": "json_object"}
)

result = json.loads(response.choices[0].message.content)
print(result)
```

### OpenAI Structured Outputs

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class ProgrammingLanguage(BaseModel):
    name: str
    year_created: int
    paradigm: str
    use_cases: list[str]

class LanguageList(BaseModel):
    languages: list[ProgrammingLanguage]

# 使用 parse 方法获取结构化输出
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "列出 3 个流行的编程语言"}
    ],
    response_format=LanguageList
)

result = response.choices[0].message.parsed
for lang in result.languages:
    print(f"{lang.name} ({lang.year_created}): {lang.paradigm}")
```

---

## Function Calling / Tool Use

### 原理

```
Function Calling 让模型能够：
1. 识别用户请求需要什么工具
2. 生成正确的函数参数
3. （由开发者执行函数）
4. 基于函数结果生成回复

这是构建 Agent 的基础能力。
```

### OpenAI Function Calling

```python
import json
from openai import OpenAI

client = OpenAI()

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "搜索产品信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "category": {
                        "type": "string",
                        "description": "产品类别"
                    },
                    "max_price": {
                        "type": "number",
                        "description": "最高价格"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# 模拟工具实现
def get_weather(city: str, unit: str = "celsius") -> dict:
    # 实际应调用天气 API
    return {"city": city, "temperature": 22, "condition": "晴", "unit": unit}

def search_products(query: str, category: str = None, max_price: float = None) -> list:
    # 实际应查询数据库
    return [{"name": f"{query} 产品1", "price": 99}, {"name": f"{query} 产品2", "price": 199}]

# Function Calling 流程
def chat_with_tools(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    # Step 1: 发送消息给模型
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # 让模型决定是否调用工具
    )

    assistant_message = response.choices[0].message

    # Step 2: 检查是否有工具调用
    if assistant_message.tool_calls:
        messages.append(assistant_message)

        # Step 3: 执行每个工具调用
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # 执行函数
            if function_name == "get_weather":
                result = get_weather(**arguments)
            elif function_name == "search_products":
                result = search_products(**arguments)
            else:
                result = {"error": "Unknown function"}

            # 添加工具结果
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

        # Step 4: 让模型基于工具结果生成回复
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        return final_response.choices[0].message.content

    # 如果没有工具调用，直接返回
    return assistant_message.content

# 测试
print(chat_with_tools("北京今天天气怎么样？"))
print(chat_with_tools("帮我搜索一下价格在 200 以内的耳机"))
```

### Anthropic Tool Use

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }
    }
]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "上海天气如何？"}]
)

# 检查是否需要使用工具
for block in response.content:
    if block.type == "tool_use":
        print(f"调用工具: {block.name}")
        print(f"参数: {block.input}")
```

---

## 练习题

### 练习 1：ReAct Agent

```python
# 任务：扩展 ReAct agent，添加以下工具：
# - weather(city): 获取天气
# - translate(text, target_lang): 翻译文本
# - news(topic): 获取新闻
#
# 测试问题："北京今天天气如何？帮我翻译成英文。"
```

### 练习 2：结构化输出

```python
# 任务：设计一个 prompt，将用户的自然语言描述转换为结构化的日程安排
#
# 输入："明天下午 3 点和张总开会讨论项目进度，大概一小时"
# 输出格式：
# {
#     "title": "...",
#     "date": "...",
#     "start_time": "...",
#     "duration_minutes": ...,
#     "participants": [...],
#     "description": "..."
# }
```

### 练习 3：Function Calling

```python
# 任务：实现一个带工具的助手，支持：
# - 查询用户余额
# - 转账
# - 查询交易记录
#
# 注意安全性：转账需要确认
```

---

## 小结

```
本节要点：
1. ReAct：推理与行动交替，适合需要工具的任务
2. ToT：多路径探索，适合复杂决策问题
3. Self-Consistency：多次采样投票，提高准确性
4. 结构化输出：JSON 格式、Pydantic 验证
5. Function Calling：让模型调用外部工具
```

---

## ➡️ 下一步

继续 [03-RAG基础.md](./03-RAG基础.md)

