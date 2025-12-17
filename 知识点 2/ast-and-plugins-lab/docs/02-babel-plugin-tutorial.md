# 02. Babel 插件开发教程

> 从零开始编写 Babel 插件

---

## 📑 目录

1. [Babel 工作流程](#babel-工作流程)
2. [插件基本结构](#插件基本结构)
3. [Visitor 模式](#visitor-模式)
4. [常用 AST 操作](#常用-ast-操作)
5. [实战插件 1：日志注入](#实战插件-1日志注入)
6. [实战插件 2：装饰器转换](#实战插件-2装饰器转换)
7. [配置与运行](#配置与运行)

---

## Babel 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     Babel 转换流程                              │
│                                                                 │
│   源代码          Parse          Transform         Generate     │
│   (Source)        (解析)          (转换)           (生成)       │
│                                                                 │
│     ┌───┐       ┌─────┐         ┌─────┐         ┌─────┐       │
│     │ JS│  ──►  │ AST │   ──►   │ AST │   ──►   │ JS  │       │
│     │   │       │     │         │ new │         │ new │       │
│     └───┘       └─────┘         └─────┘         └─────┘       │
│                                                                 │
│              @babel/parser   你的插件在这里!   @babel/generator │
│                              @babel/traverse                    │
└─────────────────────────────────────────────────────────────────┘
```

### 三个阶段

| 阶段 | 工具 | 作用 |
|------|------|------|
| **Parse** | @babel/parser | 代码 → AST |
| **Transform** | @babel/traverse + 插件 | 遍历 AST、应用转换 |
| **Generate** | @babel/generator | AST → 代码 |

---

## 插件基本结构

### 最简单的插件

```javascript
// my-plugin.js
module.exports = function (babel) {
  // babel 对象包含各种工具
  const { types: t } = babel;

  return {
    // 插件名称（可选）
    name: 'my-plugin',

    // visitor 对象：定义要访问的节点类型
    visitor: {
      // 访问标识符节点
      Identifier(path) {
        // path 是节点的路径对象，包含节点信息和操作方法
        console.log('访问到标识符:', path.node.name);
      }
    }
  };
};
```

### 使用 ES Module

```javascript
// my-plugin.mjs
export default function ({ types: t }) {
  return {
    name: 'my-plugin',
    visitor: {
      // ...
    }
  };
}
```

---

## Visitor 模式

### 什么是 Visitor

Visitor（访问者）模式是一种遍历 AST 的方式。你定义对哪些节点类型感兴趣，Babel 遍历时会在遇到这些节点时调用你的处理函数。

```javascript
visitor: {
  // 当遍历到 FunctionDeclaration 节点时调用
  FunctionDeclaration(path) {
    console.log('找到函数:', path.node.id.name);
  },

  // 当遍历到 CallExpression 节点时调用
  CallExpression(path) {
    console.log('找到函数调用');
  }
}
```

### 进入和退出

```javascript
visitor: {
  FunctionDeclaration: {
    // 进入节点时调用
    enter(path) {
      console.log('进入函数');
    },
    // 离开节点时调用
    exit(path) {
      console.log('离开函数');
    }
  }
}
```

### 访问多种节点类型

```javascript
visitor: {
  // 同时处理多种节点
  'FunctionDeclaration|ArrowFunctionExpression'(path) {
    console.log('找到函数');
  }
}
```

---

## 常用 AST 操作

### path 对象

```javascript
visitor: {
  Identifier(path) {
    // path.node: 当前节点
    console.log(path.node.name);

    // path.parent: 父节点
    console.log(path.parent.type);

    // path.parentPath: 父节点的 path
    console.log(path.parentPath.node);

    // path.scope: 当前作用域信息
    console.log(path.scope.bindings);
  }
}
```

### 判断节点类型

```javascript
const { types: t } = babel;

visitor: {
  CallExpression(path) {
    // 使用 types 工具判断
    if (t.isIdentifier(path.node.callee, { name: 'console' })) {
      console.log('找到 console 调用');
    }

    // 或使用 path.get() + is 方法
    if (path.get('callee').isIdentifier({ name: 'console' })) {
      console.log('找到 console 调用');
    }
  }
}
```

### 创建新节点

```javascript
const { types: t } = babel;

// 创建标识符
t.identifier('myVar');  // → myVar

// 创建字符串字面量
t.stringLiteral('hello');  // → "hello"

// 创建数字字面量
t.numericLiteral(42);  // → 42

// 创建函数调用
t.callExpression(
  t.identifier('console.log'),
  [t.stringLiteral('hello')]
);  // → console.log("hello")

// 创建成员表达式
t.memberExpression(
  t.identifier('console'),
  t.identifier('log')
);  // → console.log
```

### 替换节点

```javascript
visitor: {
  Identifier(path) {
    // 替换为另一个节点
    if (path.node.name === 'oldName') {
      path.replaceWith(t.identifier('newName'));
    }
  }
}
```

### 删除节点

```javascript
visitor: {
  // 删除所有 console.log
  CallExpression(path) {
    if (
      t.isMemberExpression(path.node.callee) &&
      t.isIdentifier(path.node.callee.object, { name: 'console' })
    ) {
      path.remove();
    }
  }
}
```

### 插入节点

```javascript
visitor: {
  FunctionDeclaration(path) {
    // 在函数体开头插入语句
    const logStatement = t.expressionStatement(
      t.callExpression(
        t.memberExpression(
          t.identifier('console'),
          t.identifier('log')
        ),
        [t.stringLiteral('函数被调用了')]
      )
    );

    path.get('body').unshiftContainer('body', logStatement);
  }
}
```

---

## 实战插件 1：日志注入

### 需求

为所有 `track()` 函数调用自动注入当前文件名作为参数。

```javascript
// 转换前
track('click');
track('pageview', { page: '/home' });

// 转换后
track('click', { __source: 'button.js' });
track('pageview', { page: '/home', __source: 'home.js' });
```

### 实现

```javascript
// log-inject-plugin.js
module.exports = function ({ types: t }) {
  return {
    name: 'log-inject-plugin',

    visitor: {
      CallExpression(path, state) {
        // 1. 判断是否是 track() 调用
        if (!t.isIdentifier(path.node.callee, { name: 'track' })) {
          return;
        }

        // 2. 获取当前文件名
        const filename = state.filename || 'unknown';
        const shortFilename = filename.split('/').pop();

        // 3. 创建 __source 属性
        const sourceProperty = t.objectProperty(
          t.identifier('__source'),
          t.stringLiteral(shortFilename)
        );

        // 4. 处理参数
        const args = path.node.arguments;

        if (args.length === 1) {
          // 只有一个参数，添加第二个对象参数
          args.push(
            t.objectExpression([sourceProperty])
          );
        } else if (args.length >= 2 && t.isObjectExpression(args[1])) {
          // 第二个参数是对象，添加属性
          args[1].properties.push(sourceProperty);
        } else if (args.length >= 2) {
          // 第二个参数不是对象，包装一下
          // 这里简化处理，实际可能需要更复杂的逻辑
          args.push(
            t.objectExpression([sourceProperty])
          );
        }
      }
    }
  };
};
```

---

## 实战插件 2：装饰器转换

### 需求

将简单的 `@log` 装饰器转换为等价的 JavaScript。

```javascript
// 转换前
class MyClass {
  @log
  myMethod() {
    return 'hello';
  }
}

// 转换后
class MyClass {
  myMethod() {
    console.log('myMethod called');
    return 'hello';
  }
}
```

### 实现

```javascript
// custom-decorator-transform.js
module.exports = function ({ types: t }) {
  return {
    name: 'custom-decorator-transform',

    visitor: {
      ClassMethod(path) {
        // 1. 检查是否有装饰器
        const decorators = path.node.decorators;
        if (!decorators || decorators.length === 0) {
          return;
        }

        // 2. 找到 @log 装饰器
        const logDecoratorIndex = decorators.findIndex(
          (d) => t.isIdentifier(d.expression, { name: 'log' })
        );

        if (logDecoratorIndex === -1) {
          return;
        }

        // 3. 移除装饰器
        decorators.splice(logDecoratorIndex, 1);
        if (decorators.length === 0) {
          path.node.decorators = null;
        }

        // 4. 获取方法名
        const methodName = path.node.key.name;

        // 5. 创建日志语句
        const logStatement = t.expressionStatement(
          t.callExpression(
            t.memberExpression(
              t.identifier('console'),
              t.identifier('log')
            ),
            [t.stringLiteral(`${methodName} called`)]
          )
        );

        // 6. 在方法体开头插入日志
        path.get('body').unshiftContainer('body', logStatement);
      }
    }
  };
};
```

---

## 配置与运行

### babel.config.js

```javascript
// babel.config.js
module.exports = {
  presets: [
    ['@babel/preset-env', { targets: { node: 'current' } }]
  ],
  plugins: [
    './babel-plugins/log-inject-plugin.js',
    ['@babel/plugin-proposal-decorators', { legacy: true }],
    './babel-plugins/custom-decorator-transform.js'
  ]
};
```

### .babelrc

```json
{
  "presets": ["@babel/preset-env"],
  "plugins": [
    "./babel-plugins/log-inject-plugin.js"
  ]
}
```

### 使用 CLI 运行

```bash
# 安装依赖
npm install @babel/core @babel/cli @babel/preset-env

# 转换单个文件
npx babel input.js --out-file output.js

# 转换目录
npx babel src --out-dir dist
```

### 使用 Node API

```javascript
const babel = require('@babel/core');

const code = `track('click');`;

const result = babel.transformSync(code, {
  plugins: ['./babel-plugins/log-inject-plugin.js'],
  filename: 'test.js'  // 传入文件名供插件使用
});

console.log(result.code);
// 输出: track('click', { __source: 'test.js' });
```

### 调试技巧

```javascript
// 在插件中打印 AST
visitor: {
  CallExpression(path) {
    // 打印节点结构
    console.log(JSON.stringify(path.node, null, 2));

    // 打印生成的代码
    const generate = require('@babel/generator').default;
    console.log(generate(path.node).code);
  }
}
```

