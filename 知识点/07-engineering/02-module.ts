/**
 * ============================================================
 * 📚 模块化方案
 * ============================================================
 *
 * 面试考察重点：
 * 1. 模块化的发展历程
 * 2. CommonJS vs ESM 的区别
 * 3. 循环依赖的处理
 * 4. 模块化最佳实践
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 为什么需要模块化？
 *
 * 1. 避免全局污染
 * 2. 依赖管理
 * 3. 代码复用
 * 4. 按需加载
 *
 * 📊 模块化发展历程
 *
 * 1. 全局函数时代：直接定义全局函数
 * 2. 命名空间：对象封装（如 jQuery）
 * 3. IIFE：立即执行函数
 * 4. CommonJS：Node.js 模块系统
 * 5. AMD：RequireJS，异步加载
 * 6. UMD：兼容 CommonJS 和 AMD
 * 7. ESM：ES6 原生模块系统（标准）
 */

// ============================================================
// 2. CommonJS
// ============================================================

/**
 * 📊 CommonJS 特点
 *
 * 1. 同步加载（适合服务端）
 * 2. 运行时加载
 * 3. 值的拷贝
 * 4. 单例模式（缓存）
 *
 * 语法：
 * - 导出：module.exports / exports
 * - 导入：require()
 */

const commonjsExample = `
// math.js
const PI = 3.14159;

function add(a, b) {
  return a + b;
}

module.exports = {
  PI,
  add,
};

// 或者
exports.PI = PI;
exports.add = add;

// ⚠️ 注意：不能直接给 exports 赋值
exports = { PI, add }; // ❌ 错误！切断了引用

// main.js
const math = require('./math');
console.log(math.PI);
console.log(math.add(1, 2));

// 解构导入
const { PI, add } = require('./math');
`;

/**
 * 📊 require 的实现原理（简化版）
 *
 * 1. 解析路径
 * 2. 检查缓存
 * 3. 读取文件
 * 4. 包装成函数执行
 * 5. 返回 module.exports
 */

function myRequire(modulePath: string) {
  // 1. 解析绝对路径
  const absolutePath = resolveModulePath(modulePath);

  // 2. 检查缓存
  if (myRequire.cache[absolutePath]) {
    return myRequire.cache[absolutePath].exports;
  }

  // 3. 创建 module 对象
  const module = {
    id: absolutePath,
    exports: {},
  };

  // 4. 缓存
  myRequire.cache[absolutePath] = module;

  // 5. 读取文件并执行
  const code = readFileSync(absolutePath);
  const wrapper = `
    (function(module, exports, require, __dirname, __filename) {
      ${code}
    })
  `;

  const fn = eval(wrapper);
  fn(module, module.exports, myRequire, getDirname(absolutePath), absolutePath);

  // 6. 返回 exports
  return module.exports;
}

myRequire.cache = {} as Record<string, any>;

// 模拟函数
function resolveModulePath(p: string) { return p; }
function readFileSync(p: string) { return ''; }
function getDirname(p: string) { return ''; }

// ============================================================
// 3. ESM（ES Modules）
// ============================================================

/**
 * 📊 ESM 特点
 *
 * 1. 静态分析（编译时确定依赖）
 * 2. 异步加载
 * 3. 值的引用（不是拷贝）
 * 4. 自动严格模式
 * 5. 支持 Tree Shaking
 *
 * 语法：
 * - 导出：export / export default
 * - 导入：import
 */

const esmExample = `
// math.js

// 命名导出
export const PI = 3.14159;

export function add(a, b) {
  return a + b;
}

// 默认导出
export default class Calculator {
  // ...
}

// main.js

// 命名导入
import { PI, add } from './math.js';

// 默认导入
import Calculator from './math.js';

// 命名空间导入
import * as math from './math.js';

// 动态导入
const module = await import('./math.js');

// 混合导入
import Calculator, { PI, add } from './math.js';
`;

// ============================================================
// 4. CommonJS vs ESM 区别（重要！）
// ============================================================

/**
 * 📊 CommonJS vs ESM 对比
 *
 * ┌─────────────────────┬────────────────────────┬────────────────────────┐
 * │ 特性                 │ CommonJS               │ ESM                    │
 * ├─────────────────────┼────────────────────────┼────────────────────────┤
 * │ 加载方式             │ 同步，运行时           │ 异步，编译时           │
 * │ 导出                 │ 值的拷贝               │ 值的引用               │
 * │ this                │ 当前模块               │ undefined              │
 * │ Tree Shaking        │ 不支持                 │ 支持                   │
 * │ 循环依赖             │ 部分执行               │ 变量提升 + 暂时性死区  │
 * │ 顶层 await          │ 不支持                 │ 支持                   │
 * │ 文件扩展名           │ 可省略                 │ 必须（严格模式）       │
 * └─────────────────────┴────────────────────────┴────────────────────────┘
 */

/**
 * 📊 值的拷贝 vs 值的引用
 */

const valueDifferenceExample = `
// CommonJS：值的拷贝
// counter.js
let count = 0;
function increment() {
  count++;
}
module.exports = { count, increment };

// main.js
const { count, increment } = require('./counter');
console.log(count); // 0
increment();
console.log(count); // 0 ← 还是 0！因为是拷贝

// ESM：值的引用
// counter.js
export let count = 0;
export function increment() {
  count++;
}

// main.js
import { count, increment } from './counter.js';
console.log(count); // 0
increment();
console.log(count); // 1 ← 变成 1！因为是引用
`;

// ============================================================
// 5. 循环依赖
// ============================================================

/**
 * 📊 CommonJS 循环依赖
 *
 * 特点：返回部分执行的结果
 */

const commonjsCyclicExample = `
// a.js
console.log('a.js 开始');
exports.done = false;
const b = require('./b.js');
console.log('在 a.js 中，b.done =', b.done);
exports.done = true;
console.log('a.js 结束');

// b.js
console.log('b.js 开始');
exports.done = false;
const a = require('./a.js'); // 此时 a.js 只执行了一部分
console.log('在 b.js 中，a.done =', a.done); // false
exports.done = true;
console.log('b.js 结束');

// main.js
require('./a.js');

// 输出：
// a.js 开始
// b.js 开始
// 在 b.js 中，a.done = false  ← 部分执行
// b.js 结束
// 在 a.js 中，b.done = true
// a.js 结束
`;

/**
 * 📊 ESM 循环依赖
 *
 * 特点：变量提升 + 暂时性死区
 */

const esmCyclicExample = `
// a.js
import { b } from './b.js';
console.log('a.js', b);
export const a = 'a';

// b.js
import { a } from './a.js';
console.log('b.js', a); // ReferenceError: Cannot access 'a' before initialization
export const b = 'b';

// 解决方案：使用函数延迟访问
// a.js
import { getB } from './b.js';
export const a = 'a';
console.log('a.js', getB()); // 'b'

// b.js
import { a } from './a.js';
export const b = 'b';
export function getB() { return b; }
console.log('b.js', a); // 'a'
`;

// ============================================================
// 6. UMD（Universal Module Definition）
// ============================================================

/**
 * 📊 UMD 兼容多种模块系统
 */

const umdExample = `
(function(root, factory) {
  if (typeof define === 'function' && define.amd) {
    // AMD
    define(['jquery'], factory);
  } else if (typeof module === 'object' && module.exports) {
    // CommonJS
    module.exports = factory(require('jquery'));
  } else {
    // 全局变量
    root.myModule = factory(root.jQuery);
  }
})(typeof self !== 'undefined' ? self : this, function($) {
  // 模块代码
  return {
    init: function() {
      // ...
    }
  };
});
`;

// ============================================================
// 7. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. CommonJS 和 ESM 混用问题
 *    - Node.js 中 require 不能导入 ESM
 *    - ESM 可以 import CommonJS（但有限制）
 *
 * 2. export default 的误解
 *    - export default 是导出一个叫 default 的变量
 *    - import x from 'module' 是导入 default
 *
 * 3. 动态 import 的返回值
 *    - 返回 Promise
 *    - default 导出在 result.default 上
 *
 * 4. __dirname 在 ESM 中不可用
 *    - ESM 中用 import.meta.url
 *    - const __dirname = path.dirname(fileURLToPath(import.meta.url))
 *
 * 5. package.json 的 type 字段
 *    - "type": "module" 整个包用 ESM
 *    - "type": "commonjs" 或不写用 CommonJS
 */

// ============================================================
// 8. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 为什么 ESM 支持 Tree Shaking 而 CommonJS 不支持？
 * A:
 *    - ESM 是静态的，编译时确定导入导出
 *    - CommonJS 是动态的，运行时才知道
 *    - 静态分析可以确定哪些代码未使用
 *
 * Q2: Node.js 如何判断文件是 ESM 还是 CommonJS？
 * A:
 *    1. .mjs 文件 → ESM
 *    2. .cjs 文件 → CommonJS
 *    3. .js 文件 → 看 package.json 的 type 字段
 *
 * Q3: ESM 的 import 是同步还是异步？
 * A:
 *    - 静态 import：异步加载，但像同步一样使用
 *    - 动态 import()：返回 Promise
 *
 * Q4: 如何在 Node.js 中同时支持 CommonJS 和 ESM？
 * A:
 *    package.json 配置 exports 字段：
 *    {
 *      "exports": {
 *        "import": "./dist/esm/index.js",
 *        "require": "./dist/cjs/index.js"
 *      }
 *    }
 */

// ============================================================
// 9. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：发布支持双模式的 npm 包
 */

const dualModePackage = `
// package.json
{
  "name": "my-package",
  "main": "./dist/cjs/index.js",
  "module": "./dist/esm/index.js",
  "types": "./dist/types/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/esm/index.js",
      "require": "./dist/cjs/index.js",
      "types": "./dist/types/index.d.ts"
    }
  },
  "files": ["dist"],
  "sideEffects": false
}
`;

/**
 * 🏢 场景 2：迁移 CommonJS 到 ESM
 *
 * 步骤：
 * 1. package.json 添加 "type": "module"
 * 2. require → import
 * 3. module.exports → export
 * 4. __dirname → import.meta.url
 * 5. 文件扩展名补全
 */

/**
 * 🏢 场景 3：处理循环依赖
 *
 * 检测：
 * - eslint-plugin-import
 * - circular-dependency-plugin
 *
 * 解决：
 * - 重构代码，提取公共模块
 * - 延迟访问（函数包装）
 * - 依赖注入
 */

export {
  myRequire,
  commonjsExample,
  esmExample,
  valueDifferenceExample,
  commonjsCyclicExample,
  esmCyclicExample,
  umdExample,
  dualModePackage,
};

