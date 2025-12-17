# 01. AST 基础概念

> 抽象语法树：代码的结构化表示

---

## 📑 目录

1. [什么是 AST](#什么是-ast)
2. [为什么构建工具需要 AST](#为什么构建工具需要-ast)
3. [JS 代码到 AST 的转换](#js-代码到-ast-的转换)
4. [AST 节点类型](#ast-节点类型)
5. [常见 AST 规范](#常见-ast-规范)
6. [AST 工具链](#ast-工具链)

---

## 什么是 AST

**AST（Abstract Syntax Tree，抽象语法树）** 是源代码的结构化表示。它将代码转换成树形数据结构，便于程序分析和转换。

### 类比理解

```
源代码 (字符串)                    AST (树形结构)
─────────────────                 ─────────────────
const x = 1 + 2;                  Program
                                    └── VariableDeclaration
就像：                                  ├── kind: "const"
"今天天气很好"                          └── declarations
                                            └── VariableDeclarator
    │                                           ├── id: Identifier (x)
    ▼                                           └── init: BinaryExpression
                                                        ├── operator: "+"
语法分析                                               ├── left: Literal (1)
主语: 今天                                             └── right: Literal (2)
谓语: 是
宾语: 天气很好
```

### 为什么叫"抽象"

- **抽象**：忽略无关细节（空格、换行、注释位置等）
- 保留语法结构的**本质信息**

```javascript
// 这两段代码的 AST 结构相同
const x = 1 + 2;

const   x   =   1   +   2  ;
```

---

## 为什么构建工具需要 AST

```
┌─────────────────────────────────────────────────────────────────┐
│                     构建工具处理流程                             │
│                                                                 │
│   源代码         AST           转换后 AST        目标代码        │
│   (字符串)       (树)          (树)            (字符串)         │
│                                                                 │
│     ┌───┐      ┌─────┐      ┌─────┐      ┌───┐               │
│     │ JS│ ──►  │Parse│ ──►  │Trans│ ──►  │Gen│               │
│     └───┘      └─────┘      └─────┘      └───┘               │
│                                                                 │
│               解析器        转换器       生成器                  │
│               Babel/       Babel/       Babel/                 │
│               Acorn        Traverse     Generator              │
└─────────────────────────────────────────────────────────────────┘
```

### 应用场景

| 场景 | 工具 | AST 作用 |
|------|------|---------|
| **语法转换** | Babel | ES6+ → ES5 |
| **代码压缩** | Terser | 移除无用代码、重命名变量 |
| **静态分析** | ESLint | 检查代码规范 |
| **代码格式化** | Prettier | 重新生成格式化代码 |
| **类型检查** | TypeScript | 分析类型信息 |
| **打包优化** | Webpack/Rollup | Tree Shaking |

---

## JS 代码到 AST 的转换

### 示例代码

```javascript
const greeting = "Hello, World!";

function sayHello(name) {
  return greeting + " " + name;
}

sayHello("Alice");
```

### 对应的 AST 结构

```
Program
├── body: [
│   ├── VariableDeclaration
│   │   ├── kind: "const"
│   │   └── declarations: [
│   │       └── VariableDeclarator
│   │           ├── id: Identifier { name: "greeting" }
│   │           └── init: Literal { value: "Hello, World!" }
│   │   ]
│   │
│   ├── FunctionDeclaration
│   │   ├── id: Identifier { name: "sayHello" }
│   │   ├── params: [
│   │   │   └── Identifier { name: "name" }
│   │   │   ]
│   │   └── body: BlockStatement
│   │       └── body: [
│   │           └── ReturnStatement
│   │               └── argument: BinaryExpression
│   │                   ├── operator: "+"
│   │                   ├── left: BinaryExpression
│   │                   │   ├── operator: "+"
│   │                   │   ├── left: Identifier { name: "greeting" }
│   │                   │   └── right: Literal { value: " " }
│   │                   └── right: Identifier { name: "name" }
│   │           ]
│   │
│   └── ExpressionStatement
│       └── expression: CallExpression
│           ├── callee: Identifier { name: "sayHello" }
│           └── arguments: [
│               └── Literal { value: "Alice" }
│           ]
│   ]
└── sourceType: "module"
```

### 简化的 JSON 表示

```json
{
  "type": "Program",
  "body": [
    {
      "type": "VariableDeclaration",
      "kind": "const",
      "declarations": [
        {
          "type": "VariableDeclarator",
          "id": { "type": "Identifier", "name": "greeting" },
          "init": { "type": "Literal", "value": "Hello, World!" }
        }
      ]
    },
    {
      "type": "FunctionDeclaration",
      "id": { "type": "Identifier", "name": "sayHello" },
      "params": [{ "type": "Identifier", "name": "name" }],
      "body": {
        "type": "BlockStatement",
        "body": [
          {
            "type": "ReturnStatement",
            "argument": {
              "type": "BinaryExpression",
              "operator": "+",
              "left": { "type": "Identifier", "name": "greeting" },
              "right": { "type": "Identifier", "name": "name" }
            }
          }
        ]
      }
    }
  ]
}
```

---

## AST 节点类型

### 常见节点分类

```
┌─────────────────────────────────────────────────────────────────┐
│                       AST 节点类型                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  字面量 (Literals)                                              │
│  ├── Literal: 数字、字符串、布尔值、null                         │
│  ├── TemplateLiteral: 模板字符串                                │
│  └── RegExpLiteral: 正则表达式                                  │
│                                                                 │
│  标识符 (Identifiers)                                           │
│  └── Identifier: 变量名、函数名等                               │
│                                                                 │
│  表达式 (Expressions)                                           │
│  ├── BinaryExpression: a + b                                   │
│  ├── UnaryExpression: !a, -b                                   │
│  ├── CallExpression: fn()                                      │
│  ├── MemberExpression: obj.prop                                │
│  ├── ArrowFunctionExpression: () => {}                         │
│  ├── AssignmentExpression: a = 1                               │
│  └── ConditionalExpression: a ? b : c                          │
│                                                                 │
│  语句 (Statements)                                              │
│  ├── VariableDeclaration: const/let/var                        │
│  ├── FunctionDeclaration: function fn() {}                     │
│  ├── IfStatement: if/else                                      │
│  ├── ForStatement: for 循环                                    │
│  ├── ReturnStatement: return                                   │
│  └── ExpressionStatement: 表达式语句                            │
│                                                                 │
│  模式 (Patterns)                                                │
│  ├── ObjectPattern: { a, b }                                   │
│  └── ArrayPattern: [a, b]                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 示例：各种表达式的 AST

```javascript
// BinaryExpression (二元表达式)
a + b
// { type: "BinaryExpression", operator: "+", left: {...}, right: {...} }

// CallExpression (函数调用)
console.log("hello")
// { type: "CallExpression", callee: {...}, arguments: [...] }

// MemberExpression (成员访问)
obj.property
// { type: "MemberExpression", object: {...}, property: {...} }

// ArrowFunctionExpression (箭头函数)
(x) => x * 2
// { type: "ArrowFunctionExpression", params: [...], body: {...} }
```

---

## 常见 AST 规范

### ESTree

- **标准**：JavaScript 社区标准 AST 格式
- **使用者**：Acorn, Esprima, ESLint
- **特点**：简洁、通用

### Babel AST

- **基于**：ESTree 扩展
- **使用者**：Babel
- **特点**：支持更多语法（JSX、TypeScript、提案特性）

### 主要差异

```javascript
// 箭头函数体的表示

// ESTree: 直接是表达式
{
  type: "ArrowFunctionExpression",
  body: { type: "Literal", value: 1 }  // x => 1
}

// Babel AST: 可能有额外信息
{
  type: "ArrowFunctionExpression",
  body: { type: "NumericLiteral", value: 1 },
  extra: { parenthesized: false }
}
```

---

## AST 工具链

### 解析器 (Parser)

| 工具 | 语言 | 特点 |
|------|------|------|
| **@babel/parser** | JS | 支持最新语法、JSX、TS |
| **Acorn** | JS | 轻量、符合 ESTree |
| **Esprima** | JS | 老牌、稳定 |
| **SWC** | Rust | 极快 |

### 代码使用

```javascript
// 使用 @babel/parser
const parser = require('@babel/parser');

const code = 'const x = 1 + 2;';
const ast = parser.parse(code, {
  sourceType: 'module',
  plugins: ['jsx', 'typescript']
});

console.log(JSON.stringify(ast, null, 2));
```

### 遍历器 (Traverser)

```javascript
// 使用 @babel/traverse
const traverse = require('@babel/traverse').default;

traverse(ast, {
  // 访问所有标识符节点
  Identifier(path) {
    console.log('Found identifier:', path.node.name);
  },

  // 访问所有函数调用
  CallExpression(path) {
    console.log('Found call:', path.node.callee.name);
  }
});
```

### 生成器 (Generator)

```javascript
// 使用 @babel/generator
const generate = require('@babel/generator').default;

const output = generate(ast, {
  comments: true,
  compact: false
});

console.log(output.code);
```

---

## 实践：在线探索 AST

访问 [AST Explorer](https://astexplorer.net/)：

1. 选择解析器：`@babel/parser`
2. 粘贴代码，查看实时 AST
3. 尝试修改代码，观察 AST 变化

```javascript
// 试试这段代码
const add = (a, b) => a + b;
add(1, 2);
```

观察：
- `ArrowFunctionExpression` 的结构
- `CallExpression` 的 `arguments` 数组
- `BinaryExpression` 的嵌套关系

