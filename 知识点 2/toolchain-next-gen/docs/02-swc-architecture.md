# 02. SWC 架构解析

> Speedy Web Compiler：Rust 实现的 Babel 替代品

---

## 📑 目录

1. [SWC 是什么](#swc-是什么)
2. [架构概览](#架构概览)
3. [与 Babel 的对比](#与-babel-的对比)
4. [配置详解](#配置详解)
5. [应用场景](#应用场景)
6. [面试问题与答案](#面试问题与答案)

---

## SWC 是什么

**SWC (Speedy Web Compiler)** 是一个用 Rust 编写的高性能 JavaScript/TypeScript 编译器。

### 核心能力

| 能力 | 说明 |
|------|------|
| **编译** | 将 TS/JSX/TSX 转换为 JS |
| **转译** | ES2022 → ES5（类似 Babel） |
| **压缩** | 代码压缩（类似 Terser） |
| **打包** | 模块打包（spack，较少使用） |

### 性能数据

```
Babel 编译 1000 个文件: ~30s
SWC 编译 1000 个文件:   ~1s

性能提升: 20-70 倍
```

---

## 架构概览

### 编译流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      SWC 编译流程                                │
│                                                                 │
│   源代码 (TS/JSX)                                               │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │ Parser  │  词法分析 + 语法分析 → AST                         │
│   └────┬────┘                                                   │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────┐                                               │
│   │ Transformer │  AST 转换（插件在这里介入）                     │
│   └──────┬──────┘                                               │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────┐                                                   │
│   │ Emitter │  AST → 目标代码                                   │
│   └────┬────┘                                                   │
│        │                                                        │
│        ▼                                                        │
│   目标代码 (JS)                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### AST 模型

SWC 使用自己的 AST 结构（非 ESTree 标准）：

```rust
// SWC AST 节点示例 (Rust 结构)
pub struct Function {
    pub params: Vec<Param>,
    pub decorators: Vec<Decorator>,
    pub span: Span,
    pub body: Option<BlockStmt>,
    pub is_generator: bool,
    pub is_async: bool,
    pub type_params: Option<TsTypeParamDecl>,
    pub return_type: Option<TsTypeAnn>,
}
```

与 Babel AST 的差异：
- 字段命名不同
- 类型系统不同（强类型 vs 动态类型）
- 节点结构略有差异

---

## 与 Babel 的对比

### 架构对比

| 维度 | Babel | SWC |
|------|-------|-----|
| **实现语言** | JavaScript | Rust |
| **执行模型** | 解释执行 + JIT | 编译为机器码 |
| **并行能力** | 单线程 | 多线程 |
| **插件语言** | JavaScript | Rust / WASM |
| **AST 标准** | ESTree (基本兼容) | 自定义 |
| **生态成熟度** | 非常成熟 | 快速成长中 |

### 功能对比

| 功能 | Babel | SWC |
|------|-------|-----|
| ES 语法转换 | ✅ 完整 | ✅ 完整 |
| TypeScript | ✅ @babel/preset-typescript | ✅ 原生支持 |
| JSX | ✅ @babel/preset-react | ✅ 原生支持 |
| 装饰器 | ✅ 插件 | ✅ 配置项 |
| 代码压缩 | ❌ (需 Terser) | ✅ 内置 |
| 自定义插件 | ✅ JS 编写 | ⚠️ Rust 编写（门槛高） |
| Polyfill | ✅ core-js | ⚠️ 需额外配置 |

### 配置文件对比

**Babel (.babelrc)**:
```json
{
  "presets": [
    ["@babel/preset-env", { "targets": "> 0.25%" }],
    "@babel/preset-react",
    "@babel/preset-typescript"
  ],
  "plugins": [
    "@babel/plugin-proposal-decorators",
    "@babel/plugin-proposal-class-properties"
  ]
}
```

**SWC (.swcrc)**:
```json
{
  "$schema": "https://json.schemastore.org/swcrc",
  "jsc": {
    "parser": {
      "syntax": "typescript",
      "tsx": true,
      "decorators": true
    },
    "transform": {
      "react": {
        "runtime": "automatic"
      },
      "decoratorMetadata": true
    },
    "target": "es2020"
  },
  "minify": true
}
```

---

## 配置详解

### 完整配置结构

```json
{
  // JSON Schema 支持 IDE 提示
  "$schema": "https://json.schemastore.org/swcrc",

  // JavaScript/TypeScript 编译器配置
  "jsc": {
    // 解析器配置
    "parser": {
      "syntax": "typescript",      // "ecmascript" | "typescript"
      "tsx": true,                 // 是否支持 TSX
      "decorators": true,          // 是否支持装饰器
      "dynamicImport": true        // 是否支持动态 import
    },

    // 转换配置
    "transform": {
      "react": {
        "runtime": "automatic",    // "classic" | "automatic" (React 17+)
        "importSource": "react",   // JSX 导入源
        "pragma": "React.createElement",
        "pragmaFrag": "React.Fragment"
      },
      "decoratorMetadata": true,   // 装饰器元数据
      "legacyDecorator": true      // 传统装饰器语法
    },

    // 编译目标
    "target": "es2020",            // "es3" | "es5" | "es2015" ... "es2022"

    // 是否保留 class 名称
    "keepClassNames": true,

    // 外部 helper（减少重复代码）
    "externalHelpers": true
  },

  // 模块系统
  "module": {
    "type": "es6",                 // "es6" | "commonjs" | "amd" | "umd"
    "strict": true,
    "noInterop": false
  },

  // 压缩配置
  "minify": true,

  // Source Map
  "sourceMaps": true,

  // 排除文件
  "exclude": ["node_modules"]
}
```

### 常用配置场景

**1. React + TypeScript 项目**:
```json
{
  "jsc": {
    "parser": { "syntax": "typescript", "tsx": true },
    "transform": { "react": { "runtime": "automatic" } },
    "target": "es2020"
  }
}
```

**2. Node.js 后端项目**:
```json
{
  "jsc": {
    "parser": { "syntax": "typescript", "decorators": true },
    "target": "es2021"
  },
  "module": { "type": "commonjs" }
}
```

**3. 库打包（保留 ES Modules）**:
```json
{
  "jsc": {
    "parser": { "syntax": "typescript" },
    "target": "es2018"
  },
  "module": { "type": "es6" },
  "minify": false
}
```

---

## 应用场景

### 1. Next.js 默认编译器

```javascript
// next.config.js
module.exports = {
  // Next.js 12+ 默认使用 SWC
  // 自动检测 .swcrc 配置

  // 也可以在这里配置
  experimental: {
    forceSwcTransforms: true,
  },

  compiler: {
    // SWC 特有的优化
    removeConsole: process.env.NODE_ENV === 'production',
    reactRemoveProperties: true,
  },
};
```

### 2. Vite 中使用 SWC

```javascript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc'; // 注意是 -swc 版本

export default defineConfig({
  plugins: [react()],
});
```

### 3. Jest 测试

```javascript
// jest.config.js
module.exports = {
  transform: {
    '^.+\\.(t|j)sx?$': '@swc/jest',
  },
};
```

### 4. 命令行使用

```bash
# 安装
npm install -D @swc/cli @swc/core

# 编译单个文件
npx swc src/index.ts -o dist/index.js

# 编译目录
npx swc src -d dist

# 监听模式
npx swc src -d dist --watch
```

---

## 面试问题与答案

### Q1: SWC 和 Babel 最本质的区别是什么？

**答案**：

> 最本质的区别是 **实现语言和执行模型**：
>
> - Babel 用 JavaScript 写，在 Node.js 中解释执行
> - SWC 用 Rust 写，编译为本地机器码执行
>
> 这导致了：
> 1. **性能差距**：SWC 快 20-70 倍
> 2. **插件开发门槛不同**：Babel 插件用 JS 写，SWC 需要 Rust
> 3. **AST 模型不同**：无法直接复用 Babel 插件
>
> 选型建议：如果不需要复杂的自定义转换，优先选 SWC；如果依赖特定 Babel 插件，可能还是要用 Babel。

### Q2: 为什么 SWC 不能完全替代 Babel？

**答案**：

> 主要原因是 **插件生态**：
>
> 1. Babel 有大量成熟的插件（装饰器、国际化、埋点等）
> 2. SWC 插件需要用 Rust 写，社区门槛高
> 3. 某些高级特性（如 Polyfill 注入）Babel 更成熟
>
> 但随着 SWC 支持 WASM 插件，这个差距在缩小。

### Q3: 在项目中如何从 Babel 迁移到 SWC？

**答案**：

> 迁移步骤：
>
> 1. **评估依赖**：检查项目用了哪些 Babel 插件，SWC 是否支持
> 2. **渐进迁移**：
>    - 如果用 Next.js，升级到 12+ 自动启用 SWC
>    - 如果用 Vite，换用 `@vitejs/plugin-react-swc`
>    - 如果用 Webpack，用 `swc-loader` 替换 `babel-loader`
> 3. **配置转换**：将 `.babelrc` 配置转换为 `.swcrc`
> 4. **测试验证**：确保编译结果一致
>
> 常见问题：装饰器语法、某些特殊插件可能需要调整。

