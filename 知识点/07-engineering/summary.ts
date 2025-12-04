/**
 * ============================================================
 * 📚 工程化体系 - 高频面试题汇总
 * ============================================================
 *
 * 按照面试频率和难度进行分类
 * 包含常见追问和易错点
 */

// ============================================================
// 🔥🔥🔥 高频必考题
// ============================================================

/**
 * 1. Webpack 的构建流程是什么？
 *
 * 流程：
 * 1. 初始化：读取配置，创建 Compiler
 * 2. 编译：从 Entry 开始，递归分析依赖
 * 3. 构建模块：调用 Loader 转换文件
 * 4. 生成 Chunk：根据依赖关系组合模块
 * 5. 输出：生成最终文件
 *
 * ⚠️ 易错点：
 * - Loader 从右到左执行
 * - Plugin 通过钩子介入各阶段
 *
 * 💡 追问：Loader 和 Plugin 的区别？
 * - Loader：文件转换器，作用于单个文件
 * - Plugin：扩展功能，作用于整个构建流程
 */

/**
 * 2. Vite 为什么比 Webpack 快？
 *
 * 开发模式：
 * - 不需要打包，利用浏览器原生 ESM
 * - 按需编译，请求时才编译
 * - 依赖预构建用 esbuild（Go 语言，快 10-100 倍）
 *
 * 生产模式：
 * - 使用 Rollup，速度相近
 *
 * ⚠️ 易错点：
 * - Vite 开发快，生产构建和 Webpack 差不多
 * - 依赖预构建解决 CommonJS 和请求数问题
 *
 * 💡 追问：Vite 的依赖预构建是做什么的？
 * - 将 CommonJS/UMD 转换为 ESM
 * - 合并小模块，减少请求数
 */

/**
 * 3. CommonJS 和 ESM 的区别？
 *
 * CommonJS：
 * - 同步加载，运行时加载
 * - 值的拷贝
 * - 不支持 Tree Shaking
 *
 * ESM：
 * - 异步加载，编译时确定依赖
 * - 值的引用
 * - 支持 Tree Shaking
 *
 * ⚠️ 易错点：
 * - ESM 的 import 是引用，修改会反映到原模块
 * - CommonJS 的 require 是拷贝，修改不影响原模块
 *
 * 💡 追问：为什么 ESM 支持 Tree Shaking？
 * - ESM 是静态的，编译时确定导入导出
 * - 可以静态分析哪些代码未使用
 */

/**
 * 4. 什么是 Tree Shaking？如何生效？
 *
 * 定义：移除未使用的代码
 *
 * 条件：
 * - 使用 ESM（不能是 CommonJS）
 * - 使用 production 模式
 * - package.json 配置 sideEffects
 *
 * ⚠️ 易错点：
 * - 第三方库可能不支持（如 lodash）
 * - CSS 需要标记为 sideEffects
 *
 * 💡 追问：sideEffects 有什么用？
 * - 告诉打包工具哪些文件有副作用
 * - 没有副作用的文件可以安全移除
 */

/**
 * 5. Babel 的工作原理？
 *
 * 流程：
 * 1. Parse：源码 → AST
 * 2. Transform：AST 转换（Plugin）
 * 3. Generate：AST → 代码
 *
 * ⚠️ 易错点：
 * - Babel 只转换语法，不添加 Polyfill
 * - Polyfill 需要 core-js
 *
 * 💡 追问：preset-env 和 transform-runtime 的区别？
 * - preset-env：转换语法 + 全局 Polyfill
 * - transform-runtime：复用 helper + 沙箱化 Polyfill
 * - 库开发用 transform-runtime，业务用 preset-env
 */

// ============================================================
// 🔥🔥 进阶深入题
// ============================================================

/**
 * 6. Webpack 如何优化构建速度？
 *
 * 优化手段：
 * - 缓存：cache: { type: 'filesystem' }
 * - 并行：thread-loader
 * - 减少范围：include/exclude
 * - 减少解析：noParse
 * - DLL：预编译不变的库
 *
 * ⚠️ 易错点：
 * - thread-loader 有通信开销，小项目可能更慢
 *
 * 💡 追问：如何分析构建性能？
 * - speed-measure-webpack-plugin：耗时分析
 * - webpack-bundle-analyzer：体积分析
 */

/**
 * 7. 如何实现代码分割？
 *
 * 方式：
 * - 入口分割：多入口
 * - 动态导入：import()
 * - SplitChunks：公共代码提取
 *
 * SplitChunks 配置：
 * - chunks: 'all'
 * - cacheGroups 定义分组
 *
 * ⚠️ 易错点：
 * - 分割太细会增加请求数
 * - HTTP/2 下影响较小
 *
 * 💡 追问：如何配置合理的代码分割？
 * - 按路由分割
 * - 第三方库独立 chunk
 * - 公共代码提取
 */

/**
 * 8. 什么是 Monorepo？有什么优势？
 *
 * 定义：单仓库管理多个项目/包
 *
 * 优势：
 * - 代码复用方便
 * - 统一规范
 * - 原子提交
 * - 依赖管理统一
 *
 * 工具：
 * - pnpm workspace + Turborepo（推荐）
 * - Nx
 * - Lerna
 *
 * ⚠️ 易错点：
 * - 仓库体积大
 * - 需要增量构建优化
 *
 * 💡 追问：pnpm 为什么比 npm 快？
 * - 硬链接：共享依赖
 * - 非扁平化：避免幽灵依赖
 */

/**
 * 9. 如何设计 CI/CD 流程？
 *
 * 流程：
 * 1. 代码提交：husky + lint-staged
 * 2. CI：lint → test → build
 * 3. CD：preview → deploy
 * 4. 监控：性能、错误监控
 *
 * ⚠️ 易错点：
 * - 缓存 node_modules 加速
 * - 敏感信息用 Secrets
 *
 * 💡 追问：如何优化 CI 速度？
 * - 缓存依赖
 * - 并行任务
 * - 增量构建
 * - 只测试变更部分
 */

/**
 * 10. HMR 热更新原理？
 *
 * 流程：
 * 1. 文件变化，Webpack 重新编译
 * 2. 生成新的 hash 和更新 manifest
 * 3. WebSocket 通知浏览器
 * 4. 浏览器请求更新的模块
 * 5. 执行 module.hot.accept 回调
 *
 * ⚠️ 易错点：
 * - 需要 module.hot.accept 处理更新
 * - React 用 React Refresh 插件
 *
 * 💡 追问：Vite 的 HMR 和 Webpack 有什么区别？
 * - Vite 基于 ESM，更新粒度更细
 * - 不需要重新打包整个 chunk
 */

// ============================================================
// 📝 场景题
// ============================================================

/**
 * 场景 1：如何从零搭建一个前端工程化项目？
 *
 * 1. 包管理：pnpm
 * 2. 构建工具：Vite
 * 3. 代码规范：ESLint + Prettier
 * 4. Git 规范：husky + commitlint
 * 5. 测试：Vitest
 * 6. CI/CD：GitHub Actions
 */

/**
 * 场景 2：打包体积太大怎么优化？
 *
 * 分析：
 * - webpack-bundle-analyzer 分析
 *
 * 优化：
 * - Tree Shaking
 * - 代码分割
 * - 按需引入
 * - 压缩（terser、gzip、brotli）
 * - 外部化大依赖（externals）
 */

/**
 * 场景 3：首次构建太慢怎么办？
 *
 * 优化：
 * - 开启缓存
 * - 并行编译
 * - 减少构建范围
 * - 使用 Vite（开发环境）
 * - esbuild 替代 babel-loader
 */

// ============================================================
// 🎯 资深追问清单
// ============================================================

/**
 * 追问 1：如何开发一个 Webpack Plugin？
 *
 * - 创建一个类，有 apply 方法
 * - 通过 compiler.hooks 注册钩子
 * - 在钩子中处理 compilation 对象
 */

/**
 * 追问 2：如何开发一个 Babel Plugin？
 *
 * - 分析输入输出 AST（astexplorer.net）
 * - 编写 visitor 处理节点
 * - 使用 @babel/types 创建/修改节点
 */

/**
 * 追问 3：循环依赖如何处理？
 *
 * CommonJS：
 * - 返回部分执行的结果
 *
 * ESM：
 * - 变量提升 + 暂时性死区
 * - 可能 ReferenceError
 *
 * 解决：
 * - 重构代码
 * - 延迟访问（函数包装）
 */

/**
 * 追问 4：如何实现微前端？
 *
 * 方案：
 * - qiankun：基于 single-spa
 * - Module Federation：Webpack 5
 * - iframe：隔离性好但体验差
 *
 * 关键问题：
 * - 样式隔离
 * - JS 沙箱
 * - 路由管理
 * - 通信机制
 */

// ============================================================
// 📋 面试准备 Checklist
// ============================================================

/**
 * ✅ 构建工具
 *    - [ ] Webpack 构建流程
 *    - [ ] Vite 原理
 *    - [ ] Loader vs Plugin
 *    - [ ] HMR 原理
 *
 * ✅ 模块化
 *    - [ ] CommonJS vs ESM
 *    - [ ] Tree Shaking
 *    - [ ] 循环依赖
 *
 * ✅ Babel
 *    - [ ] 工作原理
 *    - [ ] Polyfill 策略
 *    - [ ] 插件开发
 *
 * ✅ CI/CD
 *    - [ ] 流程设计
 *    - [ ] 自动化测试
 *    - [ ] 部署策略
 *
 * ✅ Monorepo
 *    - [ ] 优势和挑战
 *    - [ ] 工具选择
 *    - [ ] 版本管理
 */

export {};

