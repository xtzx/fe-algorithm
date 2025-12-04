# 🎯 前端面试知识点通关指南

> 面向有一定基础的前端开发者，系统性准备面试八股文

## 📊 学习路线图

```
Step 01 ──► Step 02 ──► Step 03 ──► Step 04
JavaScript   CSS深入      浏览器原理    网络协议
    │           │           │           │
    └───────────┴───────────┴───────────┘
                   基础篇
                      │
                      ▼
Step 05 ──► Step 06 ──► Step 07 ──► Step 08
 性能优化    框架原理     工程化       手写代码
    │           │           │           │
    └───────────┴───────────┴───────────┘
                   进阶篇
```

---

## 📚 学习大纲

### 基础篇（Step 01 - 04）

| 步骤 | 主题 | 核心内容 | 面试频率 | 预计时间 |
|:---:|------|----------|:-------:|:-------:|
| 01 | [JavaScript 核心](./01-javascript/) | 数据类型、闭包、原型链、this、事件循环、Promise | ⭐⭐⭐⭐⭐ | 4天 |
| 02 | [CSS 深入](./02-css/) | 盒模型、BFC、Flex/Grid、定位、动画、响应式 | ⭐⭐⭐⭐ | 2天 |
| 03 | [浏览器原理](./03-browser/) | 渲染流程、回流重绘、存储、安全、垃圾回收 | ⭐⭐⭐⭐⭐ | 3天 |
| 04 | [网络协议](./04-network/) | HTTP/HTTPS、TCP、缓存策略、跨域、WebSocket | ⭐⭐⭐⭐⭐ | 3天 |

### 进阶篇（Step 05 - 08）

| 步骤 | 主题 | 核心内容 | 面试频率 | 预计时间 |
|:---:|------|----------|:-------:|:-------:|
| 05 | [性能优化](./05-performance/) | 加载优化、渲染优化、代码优化、监控指标 | ⭐⭐⭐⭐⭐ | 2天 |
| 06 | [框架原理](./06-framework/) | Vue/React 响应式、虚拟DOM、Diff、生命周期、Hooks | ⭐⭐⭐⭐⭐ | 4天 |
| 07 | [工程化](./07-engineering/) | Webpack/Vite、Babel、模块化、CI/CD、微前端 | ⭐⭐⭐⭐ | 3天 |
| 08 | [手写代码](./08-handwriting/) | 防抖节流、深拷贝、Promise、bind/call/apply、发布订阅 | ⭐⭐⭐⭐⭐ | 3天 |

**总计约 24 天**，每天 1-2 小时

---

## 🎯 详细内容大纲

### Step 01: JavaScript 核心

```
📁 01-javascript/
├── 01-data-types.ts          # 数据类型与类型转换
│   ├── 基本类型 vs 引用类型
│   ├── 类型判断（typeof、instanceof、Object.prototype.toString）
│   ├── 类型转换（隐式/显式）
│   └── == vs ===
│
├── 02-scope-closure.ts       # 作用域与闭包
│   ├── 作用域链
│   ├── 词法作用域 vs 动态作用域
│   ├── 闭包原理与应用场景
│   └── 内存泄漏问题
│
├── 03-prototype.ts           # 原型与原型链
│   ├── prototype vs __proto__ vs constructor
│   ├── 原型链查找机制
│   ├── 继承的多种实现方式
│   └── new 操作符原理
│
├── 04-this.ts                # this 指向
│   ├── 默认绑定、隐式绑定、显式绑定、new绑定
│   ├── 箭头函数的 this
│   ├── 优先级规则
│   └── 常见面试题
│
├── 05-event-loop.ts          # 事件循环
│   ├── 调用栈、任务队列
│   ├── 宏任务 vs 微任务
│   ├── Node.js 事件循环差异
│   └── 经典输出题
│
├── 06-promise.ts             # Promise 与异步
│   ├── Promise 状态与链式调用
│   ├── async/await 原理
│   ├── 错误处理
│   └── 并发控制（all、race、allSettled）
│
├── 07-es6-plus.ts            # ES6+ 新特性
│   ├── let/const vs var
│   ├── 解构赋值、展开运算符
│   ├── Symbol、BigInt
│   ├── Proxy、Reflect
│   └── Map/Set/WeakMap/WeakSet
│
└── summary.ts                # 高频面试题汇总
```

### Step 02: CSS 深入

```
📁 02-css/
├── 01-box-model.ts           # 盒模型
│   ├── content-box vs border-box
│   ├── margin 合并与塌陷
│   └── 负 margin 应用
│
├── 02-bfc.ts                 # BFC
│   ├── 什么是 BFC
│   ├── 触发条件
│   └── 应用场景（清除浮动、防止margin重叠）
│
├── 03-layout.ts              # 布局方案
│   ├── Flex 布局详解
│   ├── Grid 布局详解
│   ├── 常见布局实现（圣杯、双飞翼、两栏、三栏）
│   └── 水平垂直居中（N 种方法）
│
├── 04-position.ts            # 定位与层叠
│   ├── position 五种值
│   ├── 层叠上下文
│   └── z-index 失效场景
│
├── 05-responsive.ts          # 响应式设计
│   ├── 媒体查询
│   ├── rem/em/vw/vh
│   ├── 移动端适配方案
│   └── 1px 问题
│
├── 06-animation.ts           # 动画与过渡
│   ├── transition vs animation
│   ├── transform 与 GPU 加速
│   ├── 性能优化
│   └── 常见动画效果实现
│
└── summary.ts                # 高频面试题汇总
```

### Step 03: 浏览器原理

```
📁 03-browser/
├── 01-render-process.ts      # 渲染流程
│   ├── 从 URL 输入到页面展示
│   ├── DOM 树、CSSOM 树、渲染树
│   ├── 布局与绘制
│   └── 合成层
│
├── 02-reflow-repaint.ts      # 回流与重绘
│   ├── 什么是回流（重排）
│   ├── 什么是重绘
│   ├── 如何触发
│   └── 优化策略
│
├── 03-storage.ts             # 浏览器存储
│   ├── Cookie（属性、安全、跨域）
│   ├── LocalStorage vs SessionStorage
│   ├── IndexedDB
│   └── 各存储方案对比
│
├── 04-cache.ts               # 浏览器缓存
│   ├── 强缓存（Expires、Cache-Control）
│   ├── 协商缓存（ETag、Last-Modified）
│   ├── 缓存策略最佳实践
│   └── Service Worker 缓存
│
├── 05-security.ts            # 前端安全
│   ├── XSS（类型、防御）
│   ├── CSRF（原理、防御）
│   ├── 点击劫持
│   └── HTTPS、CSP
│
├── 06-gc.ts                  # 垃圾回收
│   ├── 引用计数 vs 标记清除
│   ├── V8 垃圾回收机制
│   ├── 内存泄漏场景
│   └── 如何排查内存问题
│
└── summary.ts                # 高频面试题汇总
```

### Step 04: 网络协议

```
📁 04-network/
├── 01-http.ts                # HTTP 协议
│   ├── HTTP 报文结构
│   ├── 常见状态码
│   ├── HTTP 方法（GET vs POST）
│   ├── HTTP/1.0 vs 1.1 vs 2.0 vs 3.0
│   └── 常见请求头/响应头
│
├── 02-https.ts               # HTTPS
│   ├── 对称加密 vs 非对称加密
│   ├── SSL/TLS 握手过程
│   ├── 证书验证
│   └── HTTP vs HTTPS
│
├── 03-tcp-udp.ts             # TCP/UDP
│   ├── 三次握手、四次挥手
│   ├── TCP 可靠传输机制
│   ├── TCP vs UDP
│   └── WebSocket 原理
│
├── 04-dns.ts                 # DNS
│   ├── DNS 解析过程
│   ├── DNS 缓存
│   └── DNS 优化
│
├── 05-cors.ts                # 跨域
│   ├── 同源策略
│   ├── CORS（简单请求、预检请求）
│   ├── JSONP
│   ├── 代理
│   └── 其他跨域方案
│
├── 06-cdn.ts                 # CDN
│   ├── CDN 原理
│   ├── CDN 缓存策略
│   └── CDN 回源
│
└── summary.ts                # 高频面试题汇总
```

### Step 05: 性能优化

```
📁 05-performance/
├── 01-metrics.ts             # 性能指标
│   ├── FCP、LCP、FID、CLS
│   ├── TTFB、TTI
│   ├── Core Web Vitals
│   └── 性能监控方案
│
├── 02-loading.ts             # 加载优化
│   ├── 资源压缩（Gzip、Brotli）
│   ├── 代码分割与懒加载
│   ├── 预加载（prefetch、preload）
│   ├── 图片优化（格式、懒加载、响应式）
│   └── 字体优化
│
├── 03-rendering.ts           # 渲染优化
│   ├── 关键渲染路径优化
│   ├── 减少回流重绘
│   ├── 虚拟列表
│   ├── 骨架屏
│   └── SSR vs CSR vs SSG
│
├── 04-runtime.ts             # 运行时优化
│   ├── 防抖节流
│   ├── Web Worker
│   ├── requestAnimationFrame
│   ├── requestIdleCallback
│   └── 大数据量处理
│
├── 05-bundle.ts              # 打包优化
│   ├── Tree Shaking
│   ├── 代码分割
│   ├── 按需加载
│   └── 构建分析与优化
│
└── summary.ts                # 性能优化清单
```

### Step 06: 框架原理

```
📁 06-framework/
├── 01-virtual-dom.ts         # 虚拟 DOM
│   ├── 什么是虚拟 DOM
│   ├── 为什么需要虚拟 DOM
│   ├── 虚拟 DOM 实现
│   └── 虚拟 DOM 的优缺点
│
├── 02-diff.ts                # Diff 算法
│   ├── React Diff（双端对比）
│   ├── Vue2 Diff（双端对比）
│   ├── Vue3 Diff（最长递增子序列）
│   └── key 的作用
│
├── 03-vue-reactivity.ts      # Vue 响应式
│   ├── Vue2 响应式（Object.defineProperty）
│   ├── Vue3 响应式（Proxy）
│   ├── 依赖收集与派发更新
│   └── computed vs watch
│
├── 04-vue-lifecycle.ts       # Vue 生命周期
│   ├── Vue2 生命周期
│   ├── Vue3 生命周期
│   ├── 父子组件生命周期顺序
│   └── 常见场景与钩子选择
│
├── 05-react-hooks.ts         # React Hooks
│   ├── useState、useEffect、useRef
│   ├── useMemo、useCallback
│   ├── 自定义 Hooks
│   ├── Hooks 原理（链表）
│   └── 闭包陷阱
│
├── 06-react-fiber.ts         # React Fiber
│   ├── 为什么需要 Fiber
│   ├── Fiber 架构
│   ├── 时间切片
│   └── 优先级调度
│
├── 07-state-management.ts    # 状态管理
│   ├── Vuex/Pinia
│   ├── Redux/Mobx
│   ├── 状态管理最佳实践
│   └── 何时需要状态管理
│
└── summary.ts                # 框架对比与选型
```

### Step 07: 工程化

```
📁 07-engineering/
├── 01-module.ts              # 模块化
│   ├── CommonJS vs ESM
│   ├── AMD、UMD
│   ├── 循环依赖
│   └── Tree Shaking 原理
│
├── 02-webpack.ts             # Webpack
│   ├── 核心概念（Entry、Output、Loader、Plugin）
│   ├── 构建流程
│   ├── HMR 原理
│   ├── 常用优化配置
│   └── Loader vs Plugin
│
├── 03-vite.ts                # Vite
│   ├── 为什么快
│   ├── ESM 原理
│   ├── 依赖预构建
│   └── Webpack vs Vite
│
├── 04-babel.ts               # Babel
│   ├── 编译原理（AST）
│   ├── 预设与插件
│   ├── polyfill
│   └── 按需加载
│
├── 05-typescript.ts          # TypeScript
│   ├── 类型系统
│   ├── 泛型
│   ├── 工具类型
│   ├── 类型体操
│   └── 最佳实践
│
├── 06-ci-cd.ts               # CI/CD
│   ├── Git 工作流
│   ├── 自动化测试
│   ├── 持续集成
│   └── 持续部署
│
├── 07-micro-frontend.ts      # 微前端
│   ├── 为什么需要微前端
│   ├── 实现方案（qiankun、single-spa、Module Federation）
│   ├── 沙箱隔离
│   └── 通信机制
│
└── summary.ts                # 工程化最佳实践
```

### Step 08: 手写代码

```
📁 08-handwriting/
├── 01-debounce-throttle.ts   # 防抖节流
│   ├── 防抖实现与应用
│   ├── 节流实现与应用
│   └── 带取消功能
│
├── 02-deep-clone.ts          # 深拷贝
│   ├── JSON 方案局限
│   ├── 递归实现
│   ├── 循环引用处理
│   └── 完整版深拷贝
│
├── 03-promise.ts             # Promise
│   ├── Promise/A+ 规范
│   ├── Promise 实现
│   ├── Promise.all/race/allSettled
│   └── 并发控制
│
├── 04-bind-call-apply.ts     # bind/call/apply
│   ├── call 实现
│   ├── apply 实现
│   ├── bind 实现
│   └── 软绑定
│
├── 05-new-instanceof.ts      # new/instanceof
│   ├── new 实现
│   ├── instanceof 实现
│   └── Object.create 实现
│
├── 06-inherit.ts             # 继承
│   ├── 原型链继承
│   ├── 构造函数继承
│   ├── 组合继承
│   ├── 寄生组合继承
│   └── ES6 class 继承
│
├── 07-event-emitter.ts       # 发布订阅
│   ├── EventEmitter 实现
│   ├── once 实现
│   └── 应用场景
│
├── 08-array-methods.ts       # 数组方法
│   ├── forEach/map/filter/reduce
│   ├── flat/flatMap
│   ├── 数组去重
│   └── 数组乱序
│
├── 09-async-utils.ts         # 异步工具
│   ├── sleep
│   ├── retry（重试机制）
│   ├── 并发限制
│   └── 串行/并行执行
│
├── 10-dom-utils.ts           # DOM 相关
│   ├── 事件委托
│   ├── 懒加载
│   ├── 虚拟列表
│   └── 无限滚动
│
└── summary.ts                # 手写代码清单
```

---

## 📈 学习进度

- [ ] Step 01: JavaScript 核心
- [ ] Step 02: CSS 深入
- [ ] Step 03: 浏览器原理
- [ ] Step 04: 网络协议
- [ ] Step 05: 性能优化
- [ ] Step 06: 框架原理
- [ ] Step 07: 工程化
- [ ] Step 08: 手写代码

---

## 🎯 学习建议

### 八股文学习三步法

```
1️⃣ 理解原理 ──► 2️⃣ 能够表达 ──► 3️⃣ 举例说明
     │               │               │
  为什么这样      用自己的话      结合项目经验
  底层机制        清晰地讲出来    或具体例子
```

### 面试技巧

1. **先总后分** - 先给出结论，再展开细节
2. **结合实践** - 结合项目经验，展示落地能力
3. **知道边界** - 知道自己不知道什么，不要乱说
4. **追问准备** - 准备面试官可能的追问

---

## 🔗 推荐资源

- [MDN Web Docs](https://developer.mozilla.org/)
- [JavaScript.info](https://javascript.info/)
- [前端面试之道](https://juejin.cn/book/6844733763675488269)
- [图解 HTTP](https://book.douban.com/subject/25863515/)

---

