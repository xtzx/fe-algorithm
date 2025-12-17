# 04. Rolldown 与 Rollup 兼容性

> Vite 生态的下一代打包器：Rust 实现的 Rollup

---

## 📑 目录

1. [Rolldown 是什么](#rolldown-是什么)
2. [与 Rollup 的关系](#与-rollup-的关系)
3. [架构优势](#架构优势)
4. [插件兼容性](#插件兼容性)
5. [对 Vite 生态的影响](#对-vite-生态的影响)
6. [面试问题与答案](#面试问题与答案)

---

## Rolldown 是什么

**Rolldown** 是一个用 Rust 编写的 JavaScript 打包器，目标是成为 **Rollup 的高性能替代品**。

### 定位

```
┌─────────────────────────────────────────────────────────────────┐
│                     Vite 构建流程                               │
│                                                                 │
│                        开发模式                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Esbuild (快速转译)                          │  │
│   │              原生 ESM (浏览器直接加载)                    │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│                        生产构建                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │   当前:  Rollup (JS 实现，相对较慢)                       │  │
│   │   未来:  Rolldown (Rust 实现，更快)                       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 核心目标

| 目标 | 说明 |
|------|------|
| **API 兼容** | 与 Rollup API 尽可能兼容 |
| **插件兼容** | 现有 Rollup 插件可以直接使用 |
| **性能提升** | Rust 实现，多线程，更快的构建速度 |
| **统一工具链** | 最终成为 Vite 的核心打包器 |

---

## 与 Rollup 的关系

### Rollup 回顾

Rollup 是一个专注于 **ES Modules** 的打包器，特点：

| 特点 | 说明 |
|------|------|
| **Tree-shaking** | 业界最早支持，效果最好 |
| **ES Module 输出** | 支持输出真正的 ESM |
| **简洁的 API** | 配置相对 Webpack 更简单 |
| **适合库打包** | 生成的代码更干净 |

### Rolldown 的定位

```
Rolldown = Rollup 的 Rust 实现

目标:
├── 保持 Rollup 的 API 设计
├── 保持 Rollup 的插件接口
├── 大幅提升构建性能
└── 成为 Vite 的默认打包器
```

### 对比

| 维度 | Rollup | Rolldown |
|------|--------|----------|
| **实现语言** | JavaScript | Rust |
| **并行能力** | 单线程 | 多线程 |
| **构建速度** | 快 | 更快 (预计 10-100x) |
| **插件系统** | JS 插件 | 兼容 JS 插件 |
| **API** | 原创 | 兼容 Rollup |
| **状态** | 成熟稳定 | 开发中 |

---

## 架构优势

### 1. 多线程解析

```
Rollup (单线程):
┌──────────────────────────────────────────┐
│  Thread 1                                │
│  ├── Parse A.js (10ms)                   │
│  ├── Parse B.js (10ms)                   │
│  ├── Parse C.js (10ms)                   │
│  └── ...                                 │
│  总计: 10ms × N                          │
└──────────────────────────────────────────┘

Rolldown (多线程):
┌──────────────────────────────────────────┐
│  Thread 1: Parse A.js (10ms)             │
│  Thread 2: Parse B.js (10ms)             │
│  Thread 3: Parse C.js (10ms)             │
│  Thread 4: Parse D.js (10ms)             │
│  ...                                     │
│  总计: ~10ms (并行)                       │
└──────────────────────────────────────────┘
```

### 2. 原生 AST 处理

```
Rollup:
源码 → [JS Parser] → JS AST → [JS Transform] → 输出
        (纯 JS)      (JS 对象)   (纯 JS)

Rolldown:
源码 → [Rust Parser] → Rust AST → [Rust Transform] → 输出
       (编译优化)     (零拷贝)    (编译优化)
```

### 3. 优化的 Tree-shaking

Rolldown 继承了 Rollup 优秀的 Tree-shaking 能力，并通过 Rust 实现进一步优化：

- **更快的死代码分析**
- **并行的副作用检测**
- **优化的 Scope 分析**

---

## 插件兼容性

### 兼容策略

Rolldown 的核心承诺是 **尽可能兼容现有 Rollup 插件**。

```javascript
// Rollup 插件示例
export default function myPlugin() {
  return {
    name: 'my-plugin',

    // 这些钩子 Rolldown 都会支持
    resolveId(source, importer) { ... },
    load(id) { ... },
    transform(code, id) { ... },
    renderChunk(code, chunk) { ... },
    generateBundle(options, bundle) { ... },
  };
}
```

### 兼容性等级

| 钩子类型 | 兼容性 | 说明 |
|----------|--------|------|
| resolveId | ✅ 完全兼容 | 模块解析 |
| load | ✅ 完全兼容 | 模块加载 |
| transform | ✅ 完全兼容 | 代码转换 |
| renderChunk | ✅ 兼容 | Chunk 渲染 |
| generateBundle | ✅ 兼容 | Bundle 生成 |
| buildStart/End | ✅ 兼容 | 构建生命周期 |
| 自定义钩子 | ⚠️ 可能不兼容 | 需要适配 |

### 不兼容的场景

```javascript
// 可能不兼容的情况

// 1. 直接操作 Rollup 内部对象
function problematicPlugin() {
  return {
    buildStart() {
      // 直接访问 Rollup 内部 API
      this.getModuleInfo(); // 可能行为不同
    }
  };
}

// 2. 依赖 Rollup 特定的 AST 结构
function astPlugin() {
  return {
    transform(code) {
      const ast = this.parse(code);
      // Rolldown 的 AST 可能有细微差异
      traverse(ast, { ... });
    }
  };
}
```

---

## 对 Vite 生态的影响

### 当前 Vite 架构

```
开发模式:                    生产模式:
┌─────────────┐              ┌─────────────┐
│   Esbuild   │              │   Rollup    │
│  (转译快)    │              │  (输出优)   │
└─────────────┘              └─────────────┘
       ↑                            ↑
       └────── 不一致 ──────────────┘
```

**问题**：开发和生产使用不同的工具，可能导致行为不一致。

### 未来 Vite 架构（Rolldown）

```
开发模式 & 生产模式:
┌─────────────────────────────────────────┐
│              Rolldown                   │
│   (统一的 Rust 实现，快速 + 兼容)        │
└─────────────────────────────────────────┘
              ↑
        开发/生产一致
```

**优势**：
- 开发和生产行为完全一致
- 更快的生产构建速度
- 统一的插件系统

### 时间线（预估）

```
2023: Rolldown 开发中，内部测试
2024: Rolldown 公开测试，Vite 实验性支持
2025: Rolldown 稳定，Vite 默认使用
```

---

## 面试问题与答案

### Q1: Rolldown 和 Rollup 是什么关系？

**答案**：

> Rolldown 是 **Rollup 的 Rust 实现**，目标是完全兼容 Rollup 的 API 和插件系统。
>
> 可以理解为：
> - **接口层面**：Rolldown ≈ Rollup（API 兼容）
> - **实现层面**：Rolldown 用 Rust 重写，性能大幅提升
>
> 这类似于 SWC 之于 Babel 的关系：保持兼容性，但用系统语言重写以获得性能提升。

### Q2: 为什么 Vite 需要 Rolldown？

**答案**：

> Vite 目前有一个架构问题：
>
> - **开发模式**：使用 Esbuild 做转译，原生 ESM 提供模块
> - **生产模式**：使用 Rollup 打包
>
> 这导致开发和生产可能行为不一致。
>
> Rolldown 的意义：
> 1. **统一开发和生产**：用同一个工具链，避免行为差异
> 2. **更快的生产构建**：Rust 实现比 JS 快很多
> 3. **保持生态兼容**：现有 Rollup 插件可以继续使用

### Q3: 我现在的 Rollup 插件以后能用在 Rolldown 上吗？

**答案**：

> 大部分情况下可以。Rolldown 的设计目标就是兼容 Rollup 插件。
>
> 但要注意几点：
> 1. **标准钩子**（resolveId, load, transform 等）基本完全兼容
> 2. **内部 API**（如 this.getModuleInfo 的具体行为）可能有细微差异
> 3. **AST 操作**：如果插件直接操作 AST，需要验证兼容性
>
> 建议：等 Rolldown 稳定后，用你的插件跑一遍测试套件。

