# 🗄️ 项目：SQL Agent

> 自然语言转 SQL，自动执行并返回结果

---

## 项目概述

### 功能需求

```
1. 自然语言理解
   - 理解用户的数据查询意图
   - 识别涉及的表和字段

2. SQL 生成
   - 根据意图生成正确的 SQL
   - 支持复杂查询（JOIN、子查询）

3. 执行与反馈
   - 安全执行 SQL
   - 格式化展示结果
   - 错误处理和修正
```

### 架构设计

```
用户问题
    ↓
┌─────────────────┐
│  Schema 理解    │ ← 表结构
└─────────────────┘
    ↓
┌─────────────────┐
│  SQL 生成       │ ← LLM
└─────────────────┘
    ↓
┌─────────────────┐
│  SQL 验证       │ ← 安全检查
└─────────────────┘
    ↓
┌─────────────────┐
│  执行查询       │ ← 数据库
└─────────────────┘
    ↓
┌─────────────────┐
│  结果解读       │ ← LLM
└─────────────────┘
    ↓
自然语言回答
```

---

## 完整代码

### 基础 SQL Agent

```python
"""SQL Agent 实现"""
import sqlite3
from typing import Dict, List, Optional
from openai import OpenAI
import json

client = OpenAI()

class SQLAgent:
    """SQL Agent：自然语言转 SQL"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.schema = self._get_schema()

    def _get_schema(self) -> str:
        """获取数据库 schema"""
        cursor = self.conn.cursor()

        # 获取所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema_parts = []
        for (table_name,) in tables:
            # 获取表结构
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            col_defs = [f"  {col[1]} {col[2]}" for col in columns]
            schema_parts.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);")

            # 获取示例数据
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            samples = cursor.fetchall()
            if samples:
                col_names = [col[1] for col in columns]
                schema_parts.append(f"-- 示例数据: {col_names}")
                for row in samples:
                    schema_parts.append(f"-- {row}")

        return "\n\n".join(schema_parts)

    def _generate_sql(self, question: str) -> str:
        """根据问题生成 SQL"""
        prompt = f"""你是一个 SQL 专家。根据用户问题生成 SQLite SQL 查询。

数据库结构：
{self.schema}

用户问题：{question}

要求：
1. 只输出 SQL 语句，不要其他内容
2. 使用 SQLite 语法
3. 确保查询安全（只允许 SELECT）
4. 如果需要多个查询，只输出最主要的一个

SQL："""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        sql = response.choices[0].message.content.strip()

        # 清理可能的 markdown 标记
        if sql.startswith("```"):
            sql = sql.split("```")[1]
            if sql.startswith("sql"):
                sql = sql[3:]
        sql = sql.strip()

        return sql

    def _validate_sql(self, sql: str) -> bool:
        """验证 SQL 安全性"""
        # 只允许 SELECT 语句
        sql_upper = sql.upper().strip()

        dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
                            "ALTER", "TRUNCATE", "EXEC", "EXECUTE"]

        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False

        if not sql_upper.startswith("SELECT"):
            return False

        return True

    def _execute_sql(self, sql: str) -> Dict:
        """执行 SQL 并返回结果"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)

            # 获取列名
            columns = [description[0] for description in cursor.description]

            # 获取数据
            rows = cursor.fetchall()

            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _interpret_result(self, question: str, sql: str, result: Dict) -> str:
        """解读查询结果"""
        if not result["success"]:
            return f"查询执行失败：{result['error']}"

        if result["row_count"] == 0:
            return "查询没有返回任何结果。"

        # 格式化结果
        result_text = f"查询返回了 {result['row_count']} 条记录。\n"
        result_text += f"列: {result['columns']}\n"
        result_text += f"数据（前 10 条）:\n"
        for row in result["rows"][:10]:
            result_text += f"  {row}\n"

        # 使用 LLM 生成自然语言解读
        prompt = f"""根据以下查询结果，用自然语言回答用户问题。

用户问题：{question}
执行的 SQL：{sql}
查询结果：
{result_text}

请用简洁、专业的语言回答用户的问题："""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    def query(self, question: str) -> Dict:
        """主查询方法"""
        # 1. 生成 SQL
        sql = self._generate_sql(question)

        # 2. 验证 SQL
        if not self._validate_sql(sql):
            return {
                "success": False,
                "error": "生成的 SQL 不安全或不是查询语句",
                "sql": sql
            }

        # 3. 执行 SQL
        result = self._execute_sql(sql)

        # 4. 解读结果
        if result["success"]:
            answer = self._interpret_result(question, sql, result)
        else:
            # 尝试修复 SQL
            answer = self._fix_and_retry(question, sql, result["error"])

        return {
            "success": result["success"],
            "sql": sql,
            "result": result,
            "answer": answer
        }

    def _fix_and_retry(self, question: str, original_sql: str, error: str) -> str:
        """修复 SQL 并重试"""
        prompt = f"""SQL 查询执行失败，请修复。

数据库结构：
{self.schema}

原始问题：{question}
原始 SQL：{original_sql}
错误信息：{error}

请输出修正后的 SQL（只输出 SQL，不要其他内容）："""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        fixed_sql = response.choices[0].message.content.strip()
        if fixed_sql.startswith("```"):
            fixed_sql = fixed_sql.split("```")[1].strip()
            if fixed_sql.startswith("sql"):
                fixed_sql = fixed_sql[3:].strip()

        # 重试
        if self._validate_sql(fixed_sql):
            result = self._execute_sql(fixed_sql)
            if result["success"]:
                return self._interpret_result(question, fixed_sql, result)

        return f"无法执行查询。原始错误：{error}"

    def close(self):
        """关闭连接"""
        self.conn.close()


# 测试用例
def create_test_db():
    """创建测试数据库"""
    conn = sqlite3.connect("test.db")
    cursor = conn.cursor()

    # 创建表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER,
        city TEXT,
        created_at TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        product TEXT,
        amount REAL,
        created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    # 插入测试数据
    users = [
        (1, "张三", 28, "北京", "2024-01-01"),
        (2, "李四", 35, "上海", "2024-01-02"),
        (3, "王五", 42, "广州", "2024-01-03"),
        (4, "赵六", 31, "北京", "2024-01-04"),
        (5, "钱七", 26, "深圳", "2024-01-05"),
    ]

    orders = [
        (1, 1, "iPhone", 8999, "2024-02-01"),
        (2, 1, "AirPods", 1999, "2024-02-02"),
        (3, 2, "MacBook", 12999, "2024-02-03"),
        (4, 3, "iPad", 5999, "2024-02-04"),
        (5, 3, "Apple Watch", 3999, "2024-02-05"),
        (6, 4, "iPhone", 8999, "2024-02-06"),
        (7, 5, "MacBook", 12999, "2024-02-07"),
    ]

    cursor.executemany("INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?, ?)", users)
    cursor.executemany("INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders)

    conn.commit()
    conn.close()

    print("测试数据库创建完成")


if __name__ == "__main__":
    # 创建测试数据
    create_test_db()

    # 创建 Agent
    agent = SQLAgent("test.db")

    # 测试查询
    questions = [
        "有多少用户？",
        "北京有哪些用户？",
        "每个城市的用户数量是多少？",
        "消费最高的用户是谁？",
        "哪个产品卖得最多？",
        "张三买了什么？",
        "所有用户的平均年龄是多少？"
    ]

    for q in questions:
        print(f"\n问题: {q}")
        print("-" * 50)
        result = agent.query(q)
        print(f"SQL: {result['sql']}")
        print(f"回答: {result['answer']}")

    agent.close()
```

---

## LangGraph 版本

```python
"""使用 LangGraph 实现 SQL Agent"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from operator import add
import sqlite3

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: Annotated[List, add]
    question: str
    sql: str
    result: dict
    answer: str
    retry_count: int

def create_sql_agent_graph(db_path: str):
    """创建 SQL Agent 图"""

    conn = sqlite3.connect(db_path)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 获取 schema
    def get_schema():
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        schema_parts = []
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            col_defs = [f"{col[1]} {col[2]}" for col in columns]
            schema_parts.append(f"{table_name}({', '.join(col_defs)})")

        return "\n".join(schema_parts)

    schema = get_schema()

    # 节点：生成 SQL
    def generate_sql(state: AgentState) -> AgentState:
        question = state["question"]

        prompt = f"""生成 SQLite SQL 查询。只输出 SQL，不要其他内容。

Schema:
{schema}

问题: {question}

SQL:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        sql = response.content.strip()

        if sql.startswith("```"):
            sql = sql.split("```")[1].replace("sql", "").strip()

        return {"sql": sql, "messages": [f"生成 SQL: {sql}"]}

    # 节点：执行 SQL
    def execute_sql(state: AgentState) -> AgentState:
        sql = state["sql"]

        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()

            result = {
                "success": True,
                "columns": columns,
                "rows": rows[:20]  # 限制返回行数
            }
        except Exception as e:
            result = {"success": False, "error": str(e)}

        return {"result": result, "messages": [f"执行结果: {result['success']}"]}

    # 节点：解读结果
    def interpret_result(state: AgentState) -> AgentState:
        question = state["question"]
        sql = state["sql"]
        result = state["result"]

        if not result["success"]:
            return {"answer": f"查询失败: {result['error']}", "messages": ["查询失败"]}

        result_text = f"列: {result['columns']}\n数据: {result['rows'][:5]}"

        prompt = f"""根据查询结果回答问题。

问题: {question}
SQL: {sql}
结果: {result_text}

简洁回答:"""

        response = llm.invoke([HumanMessage(content=prompt)])

        return {"answer": response.content, "messages": [response.content]}

    # 节点：修复 SQL
    def fix_sql(state: AgentState) -> AgentState:
        question = state["question"]
        sql = state["sql"]
        error = state["result"]["error"]
        retry_count = state.get("retry_count", 0)

        prompt = f"""修复 SQL 错误。只输出修正后的 SQL。

Schema: {schema}
问题: {question}
原 SQL: {sql}
错误: {error}

修正 SQL:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        new_sql = response.content.strip()

        if new_sql.startswith("```"):
            new_sql = new_sql.split("```")[1].replace("sql", "").strip()

        return {
            "sql": new_sql,
            "retry_count": retry_count + 1,
            "messages": [f"修复 SQL: {new_sql}"]
        }

    # 条件：是否成功
    def should_fix(state: AgentState) -> str:
        result = state["result"]
        retry_count = state.get("retry_count", 0)

        if result["success"]:
            return "interpret"
        elif retry_count < 2:
            return "fix"
        else:
            return "give_up"

    # 给出失败答案
    def give_up(state: AgentState) -> AgentState:
        return {
            "answer": f"无法完成查询，最后错误: {state['result'].get('error', '未知')}",
            "messages": ["放弃重试"]
        }

    # 构建图
    workflow = StateGraph(AgentState)

    workflow.add_node("generate", generate_sql)
    workflow.add_node("execute", execute_sql)
    workflow.add_node("interpret", interpret_result)
    workflow.add_node("fix", fix_sql)
    workflow.add_node("give_up", give_up)

    workflow.set_entry_point("generate")

    workflow.add_edge("generate", "execute")
    workflow.add_conditional_edges(
        "execute",
        should_fix,
        {"interpret": "interpret", "fix": "fix", "give_up": "give_up"}
    )
    workflow.add_edge("fix", "execute")
    workflow.add_edge("interpret", END)
    workflow.add_edge("give_up", END)

    return workflow.compile()


# 使用
if __name__ == "__main__":
    create_test_db()

    agent = create_sql_agent_graph("test.db")

    result = agent.invoke({
        "question": "每个城市有多少用户？",
        "messages": [],
        "sql": "",
        "result": {},
        "answer": "",
        "retry_count": 0
    })

    print(f"回答: {result['answer']}")
```

---

## Web 界面

```python
"""Streamlit 界面"""
import streamlit as st
from sql_agent import SQLAgent, create_test_db

st.set_page_config(page_title="SQL Agent", page_icon="🗄️")

st.title("🗄️ SQL Agent")
st.caption("用自然语言查询数据库")

# 初始化
if "agent" not in st.session_state:
    create_test_db()
    st.session_state.agent = SQLAgent("test.db")

if "history" not in st.session_state:
    st.session_state.history = []

# 显示 Schema
with st.expander("查看数据库结构"):
    st.code(st.session_state.agent.schema)

# 输入
question = st.text_input("请输入您的问题", placeholder="例如：每个城市有多少用户？")

if st.button("查询") and question:
    with st.spinner("分析中..."):
        result = st.session_state.agent.query(question)

        st.session_state.history.append({
            "question": question,
            "result": result
        })

# 显示历史
for item in reversed(st.session_state.history):
    with st.container():
        st.markdown(f"**问题:** {item['question']}")

        with st.expander("查看 SQL"):
            st.code(item['result']['sql'], language="sql")

        st.markdown(f"**回答:** {item['result']['answer']}")

        if item['result']['result'].get('success') and item['result']['result'].get('rows'):
            with st.expander("查看原始数据"):
                import pandas as pd
                df = pd.DataFrame(
                    item['result']['result']['rows'],
                    columns=item['result']['result']['columns']
                )
                st.dataframe(df)

        st.divider()
```

---

## 扩展方向

```
1. 支持更多数据库（MySQL、PostgreSQL）
2. 添加查询缓存
3. 支持复杂分析（图表生成）
4. 添加查询权限控制
5. 支持自然语言修改数据（带确认）
6. 查询优化建议
```

---

## ➡️ 下一步

继续 [12-项目-LoRA微调.md](./12-项目-LoRA微调.md)

