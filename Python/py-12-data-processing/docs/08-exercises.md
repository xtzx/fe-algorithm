# 07. 练习题

## 练习 1：定义 pydantic 模型

为以下 JSON 数据定义 pydantic 模型：

```json
{
  "id": 1,
  "name": "iPhone 15",
  "price": 999.99,
  "category": "electronics",
  "in_stock": true,
  "tags": ["phone", "apple"],
  "specs": {
    "color": "black",
    "storage": "256GB"
  }
}
```

---

## 练习 2：字段验证

创建一个 User 模型：
- name: 1-50 字符
- email: 必须包含 @
- age: 0-150
- phone: 11 位数字（可选）

---

## 练习 3：自定义验证器

为密码字段创建验证器：
- 至少 8 个字符
- 包含大写字母
- 包含小写字母
- 包含数字

---

## 练习 4：嵌套模型

定义一个订单系统的模型：
- Order 包含多个 OrderItem
- OrderItem 包含 Product 信息
- 自动计算订单总价

---

## 练习 5：CSV 转 JSON

编写函数将 CSV 文件转换为 JSON：
- 支持指定编码
- 处理空值
- 返回 list[dict]

---

## 练习 6：JSONL 处理

编写函数：
- 读取 JSONL 文件
- 过滤符合条件的记录
- 写入新的 JSONL 文件

---

## 练习 7：数据清洗

清洗以下脏数据：

```python
dirty = [
    {"name": "  Alice  ", "age": "30", "email": "ALICE@EXAMPLE.COM"},
    {"name": "", "age": "invalid", "email": "bob@example.com"},
    {"name": "Charlie", "age": "25", "email": None},
]
```

- 去除名字空白
- 转换年龄为整数
- 规范化邮箱

---

## 练习 8：去重

实现多字段去重：
- 按 (name, email) 去重
- 保留最后一条记录

---

## 练习 9：数据聚合

给定销售数据，计算：
- 每个城市的总销售额
- 每个产品的平均价格
- 每月订单数量

---

## 练习 10：数据质量报告

为给定数据集生成报告：
- 总记录数
- 每个字段的空值率
- 每个字段的唯一值数量
- 数据类型分布

---

## 练习 11：字段映射

将 camelCase 字段名转换为 snake_case：

```python
{"firstName": "Alice", "lastName": "Smith", "phoneNumber": "123"}
```

---

## 练习 12：处理验证错误

处理 pydantic 验证错误：
- 收集所有错误
- 生成用户友好的错误消息
- 返回部分有效数据

---

## 练习 13：日期时间处理

创建模型处理多种日期格式：
- "2024-01-15"
- "2024/01/15"
- "15-01-2024"
- Unix 时间戳

---

## 练习 14：配置文件解析

创建配置解析器，支持：
- TOML
- YAML
- JSON
- 环境变量覆盖

---

## 练习 15：数据清洗管道

实现完整的数据清洗管道：
1. 读取 dirty.csv
2. 清洗和验证
3. 输出 clean.jsonl
4. 生成 report.json

