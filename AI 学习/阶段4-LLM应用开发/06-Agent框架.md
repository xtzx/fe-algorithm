# 🛠️ Agent 框架与 MCP

> LangGraph、AutoGen、CrewAI 与 MCP 协议

---

## LangGraph

### 概述

```
LangGraph 是 LangChain 团队开发的 Agent 编排框架：
- 基于图的状态机模型
- 支持循环和条件分支
- 内置持久化和检查点
- 支持人工干预

核心概念：
- State：状态
- Node：节点（处理逻辑）
- Edge：边（流转规则）
```

### 基础示例

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add]  # 消息列表（累加）
    next_step: str

# 创建图
workflow = StateGraph(AgentState)

# 定义节点
def agent_node(state: AgentState) -> AgentState:
    """Agent 决策节点"""
    messages = state["messages"]

    # 调用 LLM 决策
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": messages[-1]}]
    )

    result = response.choices[0].message.content

    # 判断是否需要工具
    if "需要搜索" in result:
        return {"messages": [result], "next_step": "tool"}
    else:
        return {"messages": [result], "next_step": "end"}

def tool_node(state: AgentState) -> AgentState:
    """工具执行节点"""
    # 执行工具
    tool_result = "工具执行结果..."
    return {"messages": [tool_result], "next_step": "agent"}

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)

# 设置入口
workflow.set_entry_point("agent")

# 定义边
def should_continue(state: AgentState) -> str:
    """决定下一步"""
    if state["next_step"] == "end":
        return END
    return state["next_step"]

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tool": "tool", END: END}
)

workflow.add_edge("tool", "agent")  # 工具执行后回到 agent

# 编译
app = workflow.compile()

# 运行
result = app.invoke({"messages": ["帮我搜索今天的新闻"], "next_step": ""})
print(result)
```

### ReAct Agent with LangGraph

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 定义工具
@tool
def search(query: str) -> str:
    """搜索互联网"""
    return f"搜索结果: {query}"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

# 创建 LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# 创建 ReAct Agent
agent = create_react_agent(llm, [search, calculator])

# 运行
result = agent.invoke({
    "messages": [{"role": "user", "content": "北京今天天气怎么样？"}]
})

for message in result["messages"]:
    print(f"{message.type}: {message.content}")
```

### 带检查点的 Agent

```python
from langgraph.checkpoint.memory import MemorySaver

# 创建检查点存储
checkpointer = MemorySaver()

# 创建带持久化的 Agent
agent = create_react_agent(llm, [search, calculator], checkpointer=checkpointer)

# 运行（带 thread_id 实现多轮对话）
config = {"configurable": {"thread_id": "user-123"}}

result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫小明"}]},
    config
)

result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config
)

print(result2["messages"][-1].content)  # 应该记得叫小明
```

---

## AutoGen

### 概述

```
AutoGen 是微软开发的多 Agent 框架：
- 支持多 Agent 对话
- 可配置的对话模式
- 支持人工参与
- 代码执行能力强

适用场景：
- 代码生成和调试
- 多角色协作
- 复杂任务分解
```

### 双 Agent 对话

```python
from autogen import ConversableAgent

# 创建两个 Agent
assistant = ConversableAgent(
    name="Assistant",
    system_message="你是一个有帮助的助手。",
    llm_config={"model": "gpt-4o-mini"}
)

user_proxy = ConversableAgent(
    name="User",
    human_input_mode="NEVER",  # 自动模式
    code_execution_config=False
)

# 开始对话
user_proxy.initiate_chat(
    assistant,
    message="帮我写一个 Python 快速排序函数"
)
```

### 代码执行 Agent

```python
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

# 代码执行器
code_executor = LocalCommandLineCodeExecutor(work_dir="./coding")

# 用户代理（可以执行代码）
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config={"executor": code_executor},
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "")
)

# 助手
assistant = AssistantAgent(
    name="Coder",
    system_message="""你是一个 Python 专家。
编写代码时：
1. 只输出代码块
2. 代码要完整可运行
3. 完成后说 TERMINATE""",
    llm_config={"model": "gpt-4o-mini"}
)

# 执行任务
user_proxy.initiate_chat(
    assistant,
    message="创建一个简单的 Flask API，有一个返回 'Hello World' 的端点"
)
```

### 多 Agent 群聊

```python
from autogen import GroupChat, GroupChatManager

# 创建多个专业 Agent
planner = AssistantAgent(
    name="Planner",
    system_message="你是项目规划师，负责分解任务。",
    llm_config={"model": "gpt-4o-mini"}
)

coder = AssistantAgent(
    name="Coder",
    system_message="你是程序员，负责编写代码。",
    llm_config={"model": "gpt-4o-mini"}
)

reviewer = AssistantAgent(
    name="Reviewer",
    system_message="你是代码审查员，负责检查代码质量。",
    llm_config={"model": "gpt-4o-mini"}
)

user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config=False
)

# 创建群聊
groupchat = GroupChat(
    agents=[user, planner, coder, reviewer],
    messages=[],
    max_round=10
)

manager = GroupChatManager(groupchat=groupchat, llm_config={"model": "gpt-4o-mini"})

# 开始群聊
user.initiate_chat(
    manager,
    message="开发一个简单的待办事项 API"
)
```

---

## CrewAI

### 概述

```
CrewAI 专注于多 Agent 角色协作：
- 基于角色（Agent）和任务（Task）
- 内置工作流编排
- 支持顺序和并行执行
- 简洁的 API
```

### 基础示例

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# 定义 Agent
researcher = Agent(
    role="研究员",
    goal="收集和分析信息",
    backstory="你是一个经验丰富的研究员，擅长收集和整理信息。",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="作家",
    goal="撰写高质量的内容",
    backstory="你是一个专业作家，擅长将复杂信息转化为易读的文章。",
    llm=llm,
    verbose=True
)

# 定义任务
research_task = Task(
    description="研究人工智能的最新发展趋势",
    expected_output="一份详细的研究报告",
    agent=researcher
)

writing_task = Task(
    description="基于研究报告撰写一篇科普文章",
    expected_output="一篇 500 字的科普文章",
    agent=writer
)

# 创建 Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # 顺序执行
    verbose=True
)

# 执行
result = crew.kickoff()
print(result)
```

### 带工具的 Crew

```python
from crewai_tools import SerperDevTool, WebsiteSearchTool

# 创建工具
search_tool = SerperDevTool()
web_tool = WebsiteSearchTool()

# 配置 Agent 使用工具
researcher = Agent(
    role="研究员",
    goal="使用搜索工具收集最新信息",
    tools=[search_tool, web_tool],
    llm=llm
)
```

---

## MCP（Model Context Protocol）

### 概述

```
MCP 是 Anthropic 提出的 Agent 工具标准协议：
- 统一的工具定义格式
- 标准化的工具调用流程
- 支持多种工具类型
- 可扩展架构

目标：让 Agent 能够连接到任何数据源和工具
```

### MCP 架构

```
┌─────────────────────────────────────────────────────────────┐
│                       MCP 架构                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐                     ┌─────────────┐      │
│   │   Client    │ ←── MCP Protocol ──→ │   Server    │      │
│   │  (Claude)   │                     │  (工具)     │      │
│   └─────────────┘                     └─────────────┘      │
│                                                             │
│   协议内容：                                                 │
│   - Resources: 数据/文件访问                                 │
│   - Tools: 可执行操作                                        │
│   - Prompts: 预定义提示词                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### MCP Server 示例

```python
from mcp import Server
from mcp.types import Resource, Tool

# 创建 MCP Server
server = Server("my-server")

# 注册资源
@server.resource("file://{path}")
async def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path, 'r') as f:
        return f.read()

# 注册工具
@server.tool("search")
async def search(query: str) -> str:
    """搜索工具"""
    # 实现搜索逻辑
    return f"搜索 {query} 的结果"

@server.tool("execute_sql")
async def execute_sql(query: str, database: str) -> str:
    """执行 SQL 查询"""
    import sqlite3
    conn = sqlite3.connect(database)
    result = conn.execute(query).fetchall()
    conn.close()
    return str(result)

# 运行服务器
if __name__ == "__main__":
    server.run()
```

### 在 Claude Desktop 中使用 MCP

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["path/to/my_mcp_server.py"]
    },
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "path/to/db.sqlite"]
    }
  }
}
```

### 常用 MCP Servers

```bash
# 文件系统
npx @modelcontextprotocol/server-filesystem /path/to/dir

# SQLite 数据库
npx @modelcontextprotocol/server-sqlite database.db

# GitHub
npx @modelcontextprotocol/server-github

# Slack
npx @modelcontextprotocol/server-slack
```

---

## 框架对比

| 特性 | LangGraph | AutoGen | CrewAI |
|------|-----------|---------|--------|
| 架构 | 图状态机 | 对话驱动 | 角色任务 |
| 复杂度 | 中 | 低 | 低 |
| 灵活性 | 高 | 中 | 中 |
| 代码执行 | 需扩展 | 内置 | 需扩展 |
| 多 Agent | 支持 | 专长 | 专长 |
| 持久化 | 内置 | 需扩展 | 需扩展 |
| 适用场景 | 复杂工作流 | 编程任务 | 角色协作 |

---

## 练习题

### 练习 1：LangGraph 工作流

```python
# 任务：使用 LangGraph 实现一个文档处理工作流
# 1. 文档上传
# 2. 内容提取
# 3. 摘要生成
# 4. 关键词提取
# 5. 存储
```

### 练习 2：AutoGen 代码助手

```python
# 任务：使用 AutoGen 创建一个代码助手
# - 能够编写代码
# - 能够执行代码
# - 能够调试错误
```

### 练习 3：CrewAI 研究团队

```python
# 任务：使用 CrewAI 创建一个研究团队
# - 研究员：收集信息
# - 分析师：分析数据
# - 报告员：撰写报告
```

---

## 小结

```
本节要点：
1. LangGraph：基于图的状态机，适合复杂工作流
2. AutoGen：对话驱动，擅长代码任务
3. CrewAI：角色协作，简洁易用
4. MCP：Anthropic 的工具标准协议
```

---

## ➡️ 下一步

继续 [07-微调基础.md](./07-微调基础.md)

