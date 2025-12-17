# 04. 面试问答 & 表达要点

> AST 和构建工具插件相关面试准备

---

## 📑 目录

1. [AST 相关问题](#ast-相关问题)
2. [Babel 插件问题](#babel-插件问题)
3. [Webpack 插件问题](#webpack-插件问题)
4. [Vite 插件问题](#vite-插件问题)
5. [综合能力问题](#综合能力问题)

---

## AST 相关问题

### Q1: 什么是 AST？它在前端工程化中有什么作用？

**答案框架**：

> **AST（抽象语法树）** 是源代码的结构化表示，将代码转换为树形数据结构。
>
> **作用**：
> 1. **语法转换**：Babel 将 ES6+ 转为 ES5
> 2. **代码检查**：ESLint 分析代码规范
> 3. **代码压缩**：Terser 移除无用代码
> 4. **打包优化**：Tree Shaking 识别未使用导出
>
> **为什么需要 AST**：字符串处理无法理解代码结构，而 AST 让我们能精确地分析和修改代码。
>
> **实际应用**：我在项目中写过一个 Babel 插件，用于自动为所有埋点函数注入文件名参数，方便追踪埋点来源。

---

### Q2: 你知道哪些 AST 解析器？它们有什么区别？

**答案框架**：

> | 解析器 | 语言 | 特点 |
> |--------|------|------|
> | **@babel/parser** | JS | 支持最新语法、JSX、TS，生态丰富 |
> | **Acorn** | JS | 轻量、符合 ESTree 标准 |
> | **SWC** | Rust | 极快，Babel 的替代方案 |
> | **TypeScript** | TS | 内置解析器，用于类型检查 |
>
> **选择依据**：
> - 需要转换现代语法 → Babel
> - 追求极致性能 → SWC
> - 只做代码分析 → Acorn
>
> **团队实践**：我们在构建时用 SWC 替换 Babel 做 TypeScript 转换，构建速度提升了约 3 倍。

---

## Babel 插件问题

### Q3: Babel 的工作原理是什么？

**答案框架**：

> Babel 的工作流程分三个阶段：
>
> 1. **Parse（解析）**
>    - 使用 `@babel/parser` 将代码解析为 AST
>
> 2. **Transform（转换）**
>    - 使用 `@babel/traverse` 遍历 AST
>    - 插件在这个阶段通过 Visitor 模式修改 AST
>
> 3. **Generate（生成）**
>    - 使用 `@babel/generator` 将 AST 转回代码
>
> **关键概念**：
> - **Visitor 模式**：定义对哪些节点类型感兴趣
> - **path 对象**：提供节点信息和操作方法
> - **types 工具**：创建和判断 AST 节点

---

### Q4: 你写过 Babel 插件吗？能举个例子吗？

**答案框架**：

> 写过，举两个例子：
>
> **例子 1：日志注入插件**
>
> 需求：为所有 `track()` 埋点调用自动添加文件名。
>
> ```javascript
> // 转换前
> track('click');
>
> // 转换后
> track('click', { __source: 'button.js' });
> ```
>
> 实现思路：
> 1. 在 visitor 中监听 `CallExpression`
> 2. 判断 callee 是否为 `track`
> 3. 从 `state.filename` 获取文件名
> 4. 创建新的对象参数节点并添加
>
> **例子 2：装饰器转换**
>
> 需求：将自定义 `@log` 装饰器转为等价代码。
>
> 实现思路：
> 1. 监听 `ClassMethod` 节点
> 2. 检查 decorators 数组
> 3. 移除装饰器，在方法体开头插入 `console.log`

---

### Q5: Babel 插件中 types (t) 工具有什么作用？

**答案框架**：

> `types` 工具提供两类功能：
>
> **1. 节点判断**
> ```javascript
> t.isIdentifier(node, { name: 'foo' })
> t.isCallExpression(node)
> t.isMemberExpression(node)
> ```
>
> **2. 节点创建**
> ```javascript
> t.identifier('myVar')           // 创建标识符
> t.stringLiteral('hello')        // 创建字符串
> t.callExpression(callee, args)  // 创建函数调用
> t.objectProperty(key, value)    // 创建对象属性
> ```
>
> **重要性**：直接操作 AST 容易出错，types 提供类型安全的 API。

---

## Webpack 插件问题

### Q6: Webpack 插件的工作原理是什么？

**答案框架**：

> Webpack 插件基于 **Tapable** 事件系统：
>
> **核心概念**：
> - **Compiler**：整个构建过程的控制器
> - **Compilation**：单次编译的产物和状态
> - **Hooks**：各阶段暴露的钩子
>
> **插件结构**：
> ```javascript
> class MyPlugin {
>   apply(compiler) {
>     compiler.hooks.emit.tap('MyPlugin', (compilation) => {
>       // 在输出资源前执行
>     });
>   }
> }
> ```
>
> **常用钩子**：
> - `compile`：开始编译
> - `emit`：生成资源前
> - `done`：构建完成

---

### Q7: 你写过 Webpack 插件吗？解决了什么问题？

**答案框架**：

> 写过两个插件：
>
> **1. 构建信息输出插件**
>
> 需求：每次构建后生成 `build-info.json`，记录构建时间、hash、文件清单。
>
> 实现：
> - 在 `emit` 钩子中收集 compilation 信息
> - 创建新的 asset 写入输出目录
>
> 解决问题：运维可以通过 JSON 文件确认部署版本。
>
> **2. 打包体积报告插件**
>
> 需求：构建时在控制台输出各文件体积，超阈值的文件标红。
>
> 实现：
> - 在 `done` 钩子中获取 stats
> - 遍历 assets 计算体积并排序
>
> 解决问题：及时发现体积异常的模块。

---

### Q8: Webpack 的 Tapable 是什么？

**答案框架**：

> Tapable 是 Webpack 的核心事件库，类似发布-订阅模式。
>
> **钩子类型**：
> | 类型 | 特点 |
> |------|------|
> | SyncHook | 同步顺序执行 |
> | AsyncSeriesHook | 异步顺序执行 |
> | AsyncParallelHook | 异步并行执行 |
> | SyncBailHook | 有返回值则中断 |
>
> **注册方式**：
> - `tap`: 同步钩子
> - `tapAsync`: 异步回调
> - `tapPromise`: 异步 Promise
>
> **为什么重要**：理解 Tapable 才能正确选择钩子和注册方式。

---

## Vite 插件问题

### Q9: Vite 插件和 Webpack 插件有什么区别？

**答案框架**：

> | 维度 | Vite 插件 | Webpack 插件 |
> |------|----------|-------------|
> | 基础 | Rollup 兼容 | Tapable |
> | 开发模式 | ESM 原生，按需编译 | 全量打包 |
> | 钩子风格 | 函数式，返回对象 | Class，apply 方法 |
> | 配置复杂度 | 较低 | 较高 |
>
> **Vite 特点**：
> - 开发时几乎不打包，直接 ESM
> - 插件同时支持 Rollup 钩子和 Vite 专属钩子
>
> **迁移经验**：从 Webpack 迁移到 Vite 时，大部分 loader 逻辑需要用 `transform` 钩子重写。

---

### Q10: Vite 插件常用的钩子有哪些？

**答案框架**：

> **Vite 专属钩子**：
> - `config`: 修改 Vite 配置
> - `configResolved`: 读取最终配置
> - `configureServer`: 配置开发服务器
> - `transformIndexHtml`: 转换 HTML
> - `handleHotUpdate`: 自定义 HMR
>
> **Rollup 兼容钩子**：
> - `resolveId`: 解析模块路径
> - `load`: 加载模块内容
> - `transform`: 转换代码
>
> **钩子执行顺序**：
> ```
> config → configResolved → configureServer
>        ↓
> resolveId → load → transform
>        ↓
> buildEnd → generateBundle
> ```

---

## 综合能力问题

### Q11: 什么逻辑放 Babel 插件，什么放构建工具插件？

**答案框架**：

> **Babel 插件负责**：
> - 语法转换（ES6/TS/JSX）
> - 代码级别的注入和修改
> - 静态分析（类型提取）
>
> **构建工具插件负责**：
> - 模块解析（别名、虚拟模块）
> - 资源加载（图片、CSS）
> - 构建流程控制
> - 产物处理（压缩、banner）
>
> **判断标准**：
> - 只关心代码内容 → Babel
> - 涉及文件/模块系统 → 构建工具
>
> **实际案例**：
> - 自动导入 React → Babel 插件
> - SVG 转 React 组件 → Vite 插件

---

### Q12: 如何调试 AST 相关的代码？

**答案框架**：

> **工具推荐**：
>
> 1. **AST Explorer**
>    - 在线可视化 AST
>    - 支持多种解析器
>    - 可以直接写插件预览效果
>
> 2. **console.log 大法**
>    ```javascript
>    visitor: {
>      Identifier(path) {
>        console.log(JSON.stringify(path.node, null, 2));
>      }
>    }
>    ```
>
> 3. **@babel/generator**
>    ```javascript
>    const generate = require('@babel/generator').default;
>    console.log(generate(path.node).code);
>    ```
>
> 4. **单元测试**
>    - 使用 Jest 测试插件输入输出

---

### Q13: 你在工程化方面有什么亮点或贡献？

**答案框架**：

> 举几个具体例子：
>
> **1. 埋点自动注入**
> - 开发 Babel 插件，为 track 调用注入文件名
> - 减少了手动添加 source 参数的工作
> - 埋点排查效率提升 50%
>
> **2. 构建优化**
> - 开发体积报告插件，集成到 CI
> - 设置阈值告警，防止大文件上线
> - 包体积从 2MB 优化到 800KB
>
> **3. 开发体验**
> - 开发 Vite 插件，自动生成路由
> - 减少了手动维护路由配置
>
> **表达技巧**：
> - 说明**问题背景**
> - 描述**技术方案**
> - 量化**业务收益**

