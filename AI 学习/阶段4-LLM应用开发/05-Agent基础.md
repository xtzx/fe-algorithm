# 🤖 AI Agent 基础

> 理解 Agent 的核心组件与设计模式

---

## 什么是 AI Agent

```
AI Agent 是一个能够自主决策、执行任务的智能系统。

与普通 LLM 应用的区别：
- LLM 应用：输入 → 输出（单次）
- Agent：输入 → 规划 → 执行 → 观察 → 调整 → ... → 输出（循环）

核心特征：
1. 自主性：能够独立做出决策
2. 工具使用：可以调用外部工具
3. 规划能力：能分解复杂任务
4. 记忆能力：保持上下文和历史
5. 反思能力：能评估和改进自己
```

---

## Agent 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                        AI Agent                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│   │   Planning  │    │   Memory    │    │    Tools    │    │
│   │   (规划)    │    │   (记忆)    │    │   (工具)    │    │
│   └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                             │
│                    ┌─────────────┐                          │
│                    │  Reflection │                          │
│                    │   (反思)    │                          │
│                    └─────────────┘                          │
│                                                             │
│                    ┌─────────────┐                          │
│                    │     LLM     │                          │
│                    │   (大脑)    │                          │
│                    └─────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Planning（规划）

### 任务分解

```python
def task_decomposition(task: str) -> List[str]:
    """将复杂任务分解为子任务"""
    prompt = f"""
你是一个任务规划专家。请将以下任务分解为可执行的子步骤。

任务：{task}

要求：
1. 每个步骤应该是具体、可执行的
2. 步骤之间要有逻辑顺序
3. 考虑可能的依赖关系

请以 JSON 格式输出：
{{
    "steps": [
        {{"id": 1, "action": "步骤描述", "depends_on": []}},
        {{"id": 2, "action": "步骤描述", "depends_on": [1]}},
        ...
    ]
}}
"""
    response = chat(prompt)
    return json.loads(response)["steps"]

# 示例
task = "帮我分析某公司的股票，并给出投资建议"
steps = task_decomposition(task)
for step in steps:
    print(f"{step['id']}. {step['action']} (依赖: {step['depends_on']})")
```

### 计划类型

```python
# 1. 顺序规划
sequential_plan = [
    "搜索公司基本信息",
    "获取财务数据",
    "分析财务指标",
    "查看行业趋势",
    "生成投资建议"
]

# 2. 条件规划
conditional_plan = """
IF 股价低于历史均值:
    分析是否被低估
ELSE:
    检查是否有泡沫风险
"""

# 3. 循环规划（迭代改进）
iterative_plan = """
WHILE 答案质量不满意:
    收集更多信息
    重新分析
    生成新答案
"""
```

---

## Memory（记忆）

### 记忆类型

```python
class AgentMemory:
    """Agent 记忆系统"""

    def __init__(self):
        # 短期记忆：当前对话
        self.short_term: List[Dict] = []

        # 长期记忆：持久化存储
        self.long_term: Dict = {}

        # 工作记忆：当前任务状态
        self.working: Dict = {
            "current_task": None,
            "completed_steps": [],
            "observations": []
        }

    def add_message(self, role: str, content: str):
        """添加对话消息"""
        self.short_term.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

    def get_context(self, max_tokens: int = 4000) -> str:
        """获取上下文（带 token 限制）"""
        context = []
        total = 0

        for msg in reversed(self.short_term):
            msg_tokens = len(msg["content"]) // 4  # 粗略估计
            if total + msg_tokens > max_tokens:
                break
            context.insert(0, msg)
            total += msg_tokens

        return "\n".join([f"{m['role']}: {m['content']}" for m in context])

    def save_to_long_term(self, key: str, value: any):
        """保存到长期记忆"""
        self.long_term[key] = {
            "value": value,
            "timestamp": time.time()
        }

    def retrieve_from_long_term(self, query: str) -> List[Dict]:
        """从长期记忆检索（可以用向量检索增强）"""
        # 简单实现：关键词匹配
        results = []
        for key, data in self.long_term.items():
            if query.lower() in key.lower():
                results.append({"key": key, **data})
        return results


# 向量化长期记忆
class VectorMemory:
    """基于向量的记忆系统"""

    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_store = VectorStore()
        self.memories = []

    def add(self, content: str, metadata: Dict = None):
        """添加记忆"""
        embedding = self.embedder.embed([content])[0]
        self.vector_store.add(
            np.array([embedding]),
            [{"content": content, **(metadata or {})}]
        )
        self.memories.append(content)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关记忆"""
        query_emb = self.embedder.embed_query(query)
        return self.vector_store.search(query_emb, top_k)
```

---

## Tools（工具）

### 工具定义

```python
from typing import Callable
from dataclasses import dataclass

@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: Dict

class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: Dict):
        """装饰器方式注册工具"""
        def decorator(func):
            self.tools[name] = Tool(
                name=name,
                description=description,
                func=func,
                parameters=parameters
            )
            return func
        return decorator

    def get_tools_prompt(self) -> str:
        """生成工具描述 prompt"""
        lines = ["可用工具："]
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool.description}")
            lines.append(f"  参数: {json.dumps(tool.parameters, ensure_ascii=False)}")
        return "\n".join(lines)

    def execute(self, name: str, **kwargs):
        """执行工具"""
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        return self.tools[name].func(**kwargs)


# 注册工具
registry = ToolRegistry()

@registry.register(
    name="search_web",
    description="搜索互联网获取信息",
    parameters={"query": "搜索关键词"}
)
def search_web(query: str) -> str:
    # 实际实现会调用搜索 API
    return f"搜索结果: {query} 的相关信息..."

@registry.register(
    name="calculator",
    description="执行数学计算",
    parameters={"expression": "数学表达式"}
)
def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

@registry.register(
    name="read_file",
    description="读取文件内容",
    parameters={"path": "文件路径"}
)
def read_file(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取错误: {e}"
```

### 常用工具类型

```python
# 1. 搜索工具
def web_search(query: str) -> str:
    """使用 SerpAPI / Tavily / Brave 等搜索"""
    pass

# 2. 代码执行
def python_repl(code: str) -> str:
    """执行 Python 代码"""
    import subprocess
    result = subprocess.run(
        ["python", "-c", code],
        capture_output=True,
        text=True
    )
    return result.stdout or result.stderr

# 3. 数据库查询
def sql_query(query: str, db_path: str) -> str:
    """执行 SQL 查询"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(query)
    results = cursor.fetchall()
    conn.close()
    return str(results)

# 4. API 调用
def call_api(url: str, method: str = "GET", data: Dict = None) -> str:
    """调用外部 API"""
    import requests
    if method == "GET":
        response = requests.get(url, params=data)
    else:
        response = requests.post(url, json=data)
    return response.text

# 5. 文件操作
def write_file(path: str, content: str) -> str:
    """写入文件"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"已写入 {path}"
```

---

## Reflection（反思）

### 自我评估

```python
def self_evaluate(task: str, result: str) -> Dict:
    """评估任务完成质量"""
    prompt = f"""
请评估以下任务的完成情况：

任务：{task}
结果：{result}

请从以下维度评分（1-10）并说明理由：
1. 完整性：是否完整回答了问题
2. 准确性：信息是否准确
3. 相关性：是否切题
4. 可用性：结果是否可以直接使用

输出 JSON 格式：
{{
    "scores": {{
        "completeness": 分数,
        "accuracy": 分数,
        "relevance": 分数,
        "usability": 分数
    }},
    "overall": 总分,
    "feedback": "改进建议",
    "needs_improvement": true/false
}}
"""
    response = chat(prompt)
    return json.loads(response)


def reflect_and_improve(task: str, result: str, feedback: str) -> str:
    """根据反馈改进结果"""
    prompt = f"""
你之前完成了一个任务，但收到了改进反馈。请改进你的答案。

原任务：{task}
原结果：{result}
改进反馈：{feedback}

请给出改进后的结果：
"""
    return chat(prompt)
```

### Reflexion 模式

```python
class ReflexionAgent:
    """带反思的 Agent"""

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.memory = AgentMemory()

    def run(self, task: str) -> str:
        """执行任务（带反思循环）"""
        result = None

        for i in range(self.max_iterations):
            # 执行任务
            if result is None:
                result = self._execute(task)
            else:
                result = self._execute(task, previous_result=result, feedback=feedback)

            # 评估
            evaluation = self_evaluate(task, result)

            if not evaluation['needs_improvement']:
                return result

            # 记录反思
            feedback = evaluation['feedback']
            self.memory.add_message("reflection", f"迭代 {i+1}: {feedback}")

        return result

    def _execute(self, task: str, previous_result: str = None, feedback: str = None) -> str:
        """执行任务"""
        if previous_result and feedback:
            return reflect_and_improve(task, previous_result, feedback)
        else:
            return chat(f"请完成任务：{task}")
```

---

## 简单 Agent 实现

```python
class SimpleAgent:
    """简单的 ReAct Agent"""

    def __init__(self):
        self.tools = ToolRegistry()
        self.memory = AgentMemory()
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认工具"""
        @self.tools.register("search", "搜索信息", {"query": "搜索词"})
        def search(query):
            return f"搜索结果: 关于 {query} 的信息"

        @self.tools.register("calculate", "数学计算", {"expr": "表达式"})
        def calculate(expr):
            return str(eval(expr))

    def run(self, task: str, max_steps: int = 10) -> str:
        """执行任务"""
        self.memory.working["current_task"] = task

        system_prompt = f"""
你是一个智能助手，可以使用工具完成任务。

{self.tools.get_tools_prompt()}

请按以下格式思考和行动：
Thought: 分析当前情况
Action: tool_name(参数)
... 等待工具结果 ...
Thought: 分析结果
Action: ...
... 或者 ...
Thought: 我已经有足够信息了
Answer: 最终答案
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"任务：{task}"}
        ]

        for step in range(max_steps):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                stop=["Observation:"]
            )

            output = response.choices[0].message.content
            messages.append({"role": "assistant", "content": output})

            # 检查是否完成
            if "Answer:" in output:
                return output.split("Answer:")[-1].strip()

            # 解析并执行工具
            if "Action:" in output:
                action_line = output.split("Action:")[-1].strip().split("\n")[0]

                # 解析工具调用
                import re
                match = re.match(r'(\w+)\((.*)\)', action_line)
                if match:
                    tool_name, args_str = match.groups()

                    # 解析参数
                    try:
                        # 简单参数解析
                        args = {}
                        if args_str:
                            for arg in args_str.split(","):
                                if "=" in arg:
                                    k, v = arg.split("=", 1)
                                    args[k.strip()] = v.strip().strip('"\'')
                                else:
                                    # 假设是第一个参数
                                    param_name = list(self.tools.tools[tool_name].parameters.keys())[0]
                                    args[param_name] = arg.strip().strip('"\'')

                        result = self.tools.execute(tool_name, **args)
                    except Exception as e:
                        result = f"错误: {e}"

                    observation = f"\nObservation: {result}\n"
                    messages.append({"role": "user", "content": observation})

        return "达到最大步数，任务未完成"


# 使用
agent = SimpleAgent()
result = agent.run("计算 (123 + 456) * 2 的结果")
print(result)
```

---

## 单 Agent vs 多 Agent

```
单 Agent：
- 一个 LLM 负责所有决策
- 简单场景足够
- 上下文长度限制

多 Agent：
- 多个专业化 Agent 协作
- 各 Agent 有不同角色/能力
- 适合复杂任务
- 可以并行处理

协作模式：
1. 顺序执行：A → B → C
2. 层级结构：Manager → Workers
3. 辩论模式：多个 Agent 讨论得出结论
4. 自组织：Agent 自行协调
```

---

## 练习题

### 练习 1：添加工具

```python
# 任务：为 SimpleAgent 添加以下工具：
# - 天气查询
# - 网页抓取
# - 文件读写
```

### 练习 2：实现记忆

```python
# 任务：实现一个带长期记忆的 Agent
# - 能记住用户的偏好
# - 能从历史对话中学习
```

### 练习 3：反思机制

```python
# 任务：实现一个带反思的 Agent
# - 完成任务后自我评估
# - 如果评估不佳，自动改进
```

---

## 小结

```
本节要点：
1. Agent 四大组件：Planning、Memory、Tools、Reflection
2. Planning：任务分解、执行策略
3. Memory：短期/长期/工作记忆
4. Tools：工具注册、执行
5. Reflection：自我评估、改进
```

---

## ➡️ 下一步

继续 [06-Agent框架.md](./06-Agent框架.md)

