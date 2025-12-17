# 🔧 AST & 插件开发实战

> 手写 Webpack/Vite 插件 & Babel 插件 — 深入理解前端工程化

## 📚 项目简介

本项目面向 **7-8 年经验的资深前端工程师**，通过实战帮助你：

1. 理解 **AST（抽象语法树）** 的核心概念
2. 掌握 **Babel 插件** 开发：代码分析与转换
3. 掌握 **Webpack/Vite 插件** 开发：构建流程扩展
4. 准备高级面试中的工程化问题

---

## 📁 项目结构

```
ast-and-plugins-lab/
├── README.md                               # 本文件
├── docs/
│   ├── 01-ast-basics.md                    # AST 基础概念
│   ├── 02-babel-plugin-tutorial.md         # Babel 插件开发教程
│   ├── 03-webpack-vite-plugin-tutorial.md  # Webpack/Vite 插件教程
│   └── 04-interview-qa-and-talking-points.md  # 面试问答
├── babel-plugins/
│   ├── log-inject-plugin.js                # 日志注入插件
│   ├── custom-decorator-transform.js       # 装饰器转换插件
│   └── examples/
│       ├── input-sample.js                 # 转换前代码
│       └── output-sample.js                # 转换后代码
├── webpack-plugins/
│   ├── simple-build-info-plugin.js         # 构建信息输出插件
│   └── bundle-size-report-plugin.js        # 打包体积报告插件
├── vite-plugins/
│   ├── banner-inject-plugin.ts             # Banner 注入插件
│   └── env-replace-plugin.ts               # 环境变量替换插件
└── scripts/
    ├── run-babel-transform.sh              # Babel 转换脚本
    └── run-webpack-with-plugin.sh          # Webpack 构建脚本
```

---

## 🎯 学习路线

```
Step 1: AST 基础
├── 什么是抽象语法树
├── JS 代码 → AST 的过程
└── AST 节点类型
        │
        ▼
Step 2: Babel 插件
├── Babel 工作流程
├── Visitor 模式
├── 节点操作 API
└── 实战：日志注入、装饰器转换
        │
        ▼
Step 3: Webpack 插件
├── Tapable 钩子机制
├── Compiler 与 Compilation
└── 实战：构建信息、体积报告
        │
        ▼
Step 4: Vite 插件
├── Rollup 插件兼容
├── Vite 专属钩子
└── 实战：Banner 注入、环境变量
        │
        ▼
Step 5: 综合应用
├── Babel + 构建工具组合
└── 面试准备
```

---

## 🔥 核心技能点

### AST 操作

| 技能点 | 重要性 | 说明 |
|--------|:------:|------|
| AST 结构理解 | ⭐⭐⭐⭐⭐ | 节点类型、树形结构 |
| Babel 插件开发 | ⭐⭐⭐⭐⭐ | Visitor 模式、节点操作 |
| AST 工具链 | ⭐⭐⭐⭐ | @babel/parser, @babel/traverse |

### 构建工具插件

| 技能点 | 重要性 | 说明 |
|--------|:------:|------|
| Webpack 插件 | ⭐⭐⭐⭐⭐ | Tapable、Compiler/Compilation |
| Vite 插件 | ⭐⭐⭐⭐⭐ | Rollup 兼容、专属钩子 |
| 构建流程理解 | ⭐⭐⭐⭐ | 各阶段的能力边界 |

---

## 🚀 快速开始

### 运行 Babel 转换

```bash
# 安装依赖
npm install @babel/core @babel/cli @babel/preset-env

# 运行转换
npx babel babel-plugins/examples/input-sample.js \
  --plugins ./babel-plugins/log-inject-plugin.js \
  --out-file babel-plugins/examples/output-sample.js
```

### 运行 Webpack 构建

```bash
# 安装依赖
npm install webpack webpack-cli

# 使用自定义插件构建
npx webpack --config webpack.config.js
```

### 运行 Vite 插件

```bash
# 在 vite.config.ts 中引入插件
import bannerPlugin from './vite-plugins/banner-inject-plugin'

export default {
  plugins: [bannerPlugin()]
}
```

---

## 📖 推荐阅读顺序

1. `docs/01-ast-basics.md` - AST 基础概念
2. `docs/02-babel-plugin-tutorial.md` - Babel 插件开发
3. `docs/03-webpack-vite-plugin-tutorial.md` - 构建工具插件
4. `docs/04-interview-qa-and-talking-points.md` - 面试准备

---

## 🔗 参考资源

- [AST Explorer](https://astexplorer.net/) - 在线 AST 可视化
- [Babel 插件手册](https://github.com/jamiebuilds/babel-handbook)
- [Webpack 插件 API](https://webpack.js.org/api/plugins/)
- [Vite 插件 API](https://vitejs.dev/guide/api-plugin.html)

