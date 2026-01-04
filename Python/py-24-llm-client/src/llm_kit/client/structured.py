"""
结构化输出

特性:
- JSON Schema 约束
- Pydantic 验证
- 自动重试
- 失败处理
"""

import json
from typing import Any, Dict, List, Optional, Type, TypeVar

import structlog
from pydantic import BaseModel, ValidationError

from llm_kit.client.base import BaseLLMClient, ChatMessage, InvalidRequestError

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """结构化输出错误"""
    pass


class StructuredClient:
    """
    结构化输出客户端
    
    使用 JSON Schema 约束 LLM 输出，并用 Pydantic 验证
    
    Usage:
        from pydantic import BaseModel
        
        class Person(BaseModel):
            name: str
            age: int
            occupation: str
        
        client = StructuredClient(llm_client)
        person = client.generate(
            prompt="Extract: John is a 30-year-old software engineer",
            schema=Person,
        )
        print(person.name, person.age)  # John 30
    """

    def __init__(
        self,
        client: BaseLLMClient,
        max_retries: int = 3,
        model: str = "gpt-4o-mini",
    ):
        self.client = client
        self.max_retries = max_retries
        self.model = model

    def generate(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> T:
        """
        生成结构化输出
        
        Args:
            prompt: 用户提示
            schema: Pydantic 模型类
            system_prompt: 系统提示
            temperature: 温度（默认 0 以获得确定性输出）
            **kwargs: 其他 LLM 参数
        
        Returns:
            Pydantic 模型实例
        
        Raises:
            StructuredOutputError: 验证失败
        """
        json_schema = self._get_json_schema(schema)
        
        # 构建系统提示
        base_system = system_prompt or "You are a helpful assistant."
        schema_instruction = self._build_schema_instruction(schema, json_schema)
        full_system = f"{base_system}\n\n{schema_instruction}"
        
        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]
        
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat(
                    messages=messages,
                    model=self.model,
                    temperature=temperature,
                    **kwargs,
                )
                
                # 提取 JSON
                json_str = self._extract_json(response.content)
                
                # 解析并验证
                data = json.loads(json_str)
                result = schema.model_validate(data)
                
                logger.info(
                    "structured_output_success",
                    schema=schema.__name__,
                    attempt=attempt + 1,
                )
                
                return result

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(
                    "structured_output_retry",
                    schema=schema.__name__,
                    attempt=attempt + 1,
                    error=str(e),
                )
                
                # 添加错误反馈用于重试
                messages.append({
                    "role": "assistant",
                    "content": response.content if 'response' in dir() else "",
                })
                messages.append({
                    "role": "user",
                    "content": f"Your response was invalid. Error: {e}\nPlease provide a valid JSON response matching the schema.",
                })

        raise StructuredOutputError(
            f"Failed to generate valid output after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def generate_list(
        self,
        prompt: str,
        item_schema: Type[T],
        max_items: Optional[int] = None,
        **kwargs,
    ) -> List[T]:
        """
        生成结构化列表
        
        Args:
            prompt: 用户提示
            item_schema: 列表项的 Pydantic 模型
            max_items: 最大项数
            **kwargs: 其他参数
        
        Returns:
            Pydantic 模型实例列表
        """
        # 创建包装 schema
        class ListWrapper(BaseModel):
            items: List[item_schema]  # type: ignore
        
        extra_instruction = ""
        if max_items:
            extra_instruction = f"Return at most {max_items} items."
        
        wrapper = self.generate(
            prompt=f"{prompt}\n{extra_instruction}".strip(),
            schema=ListWrapper,
            **kwargs,
        )
        
        return wrapper.items

    def _get_json_schema(self, schema: Type[BaseModel]) -> Dict[str, Any]:
        """获取 JSON Schema"""
        return schema.model_json_schema()

    def _build_schema_instruction(
        self,
        schema: Type[BaseModel],
        json_schema: Dict[str, Any],
    ) -> str:
        """构建 schema 指令"""
        schema_str = json.dumps(json_schema, indent=2)
        
        return f"""You must respond with a valid JSON object that matches this schema:

```json
{schema_str}
```

Important:
- Respond ONLY with the JSON object, no other text
- Ensure all required fields are present
- Use the correct data types
"""

    def _extract_json(self, content: str) -> str:
        """从响应中提取 JSON"""
        content = content.strip()
        
        # 移除 markdown 代码块
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.endswith("```"):
            content = content[:-3]
        
        return content.strip()


class FunctionCallParser:
    """
    函数调用解析器
    
    解析 LLM 的 tool_calls 响应
    
    Usage:
        parser = FunctionCallParser()
        
        # 注册函数 schema
        parser.register(
            name="get_weather",
            schema=GetWeatherArgs,
        )
        
        # 解析 tool_calls
        for call in response.tool_calls:
            func_name, args = parser.parse(call)
            # args 是经过验证的 Pydantic 实例
    """

    def __init__(self):
        self._schemas: Dict[str, Type[BaseModel]] = {}

    def register(self, name: str, schema: Type[BaseModel]):
        """注册函数 schema"""
        self._schemas[name] = schema

    def parse(self, tool_call: Dict[str, Any]) -> tuple[str, BaseModel]:
        """
        解析工具调用
        
        Args:
            tool_call: 工具调用字典
        
        Returns:
            (函数名, 参数实例)
        """
        func = tool_call.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        
        if name not in self._schemas:
            raise InvalidRequestError(f"Unknown function: {name}")
        
        schema = self._schemas[name]
        args_data = json.loads(args_str)
        args = schema.model_validate(args_data)
        
        return name, args

    def get_tools_definition(self) -> List[Dict[str, Any]]:
        """获取 tools 定义（用于 API 请求）"""
        tools = []
        for name, schema in self._schemas.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema.__doc__ or "",
                    "parameters": schema.model_json_schema(),
                },
            })
        return tools


