/**
 * ============================================================
 * 📚 JavaScript 核心 - 高频面试题汇总
 * ============================================================
 *
 * 按照面试频率和难度进行分类
 */

// ============================================================
// 🔥🔥🔥 高频必考题
// ============================================================

/**
 * 1. typeof 和 instanceof 的区别？如何判断数据类型？
 *
 * typeof：
 * - 返回字符串，表示操作数的类型
 * - 能准确判断：undefined、boolean、number、string、symbol、bigint、function
 * - 缺点：typeof null === 'object'，无法区分数组和对象
 *
 * instanceof：
 * - 检查构造函数的 prototype 是否在对象的原型链上
 * - 只能判断引用类型，不能判断基本类型
 * - 缺点：跨 iframe 失效
 *
 * 最准确的方式：Object.prototype.toString.call()
 *
 * ⚠️ 易错点：
 * - typeof function 返回 "function"，但函数也是对象
 * - typeof NaN 返回 "number"
 * - [] instanceof Object 也是 true
 *
 * 💡 面试追问：为什么 typeof null === 'object'？
 * - 历史遗留 bug，JS 早期用低位标记类型，null 全零被误判为对象
 * - 修复会破坏现有代码，所以保留
 */

/**
 * 2. == 和 === 的区别？
 *
 * ==（抽象相等）：会进行类型转换
 * - null == undefined → true
 * - 布尔值转数字比较
 * - 字符串和数字比较，字符串转数字
 * - 对象和基本类型比较，对象调用 ToPrimitive
 *
 * ===（严格相等）：不进行类型转换
 * - 类型和值都必须相等
 * - NaN !== NaN
 *
 * 推荐：始终使用 ===
 */

/**
 * 3. 什么是闭包？闭包的应用场景？
 *
 * 定义：函数能够访问其词法作用域外的变量
 *
 * 应用场景：
 * - 数据私有化（模块模式）
 * - 函数工厂
 * - 缓存（记忆化）
 * - 防抖节流
 * - 柯里化
 * - React Hooks
 *
 * ⚠️ 易错点：
 * - 循环中 var 配合闭包的陷阱（let 可解决）
 * - 闭包引用的是变量本身，不是值的快照
 * - 不必要的闭包会占用内存
 *
 * 💡 面试追问：
 * Q: 闭包为什么会导致内存泄漏？
 * A: 函数引用外部变量，即使函数不再使用，
 *    变量也不会被回收，需要手动解除引用
 *
 * Q: React Hooks 依赖数组和闭包的关系？
 * A: Hooks 回调形成闭包，依赖数组决定是否重新创建闭包
 */

/**
 * 4. 原型链是什么？如何实现继承？
 *
 * 原型链：每个对象都有 __proto__ 指向其原型对象，
 * 原型对象也有 __proto__，形成链式结构。
 * 查找属性时会沿着原型链向上查找。
 *
 * 继承方式：
 * 1. 原型链继承（引用类型共享问题）
 * 2. 构造函数继承（无法继承原型方法）
 * 3. 组合继承（调用两次父构造函数）
 * 4. 寄生组合继承（最佳实践）
 * 5. ES6 class extends（语法糖）
 */

/**
 * 5. this 的指向规则？
 *
 * 四种绑定规则（优先级从低到高）：
 * 1. 默认绑定：独立函数调用，this 指向全局/undefined
 * 2. 隐式绑定：obj.fn()，this 指向 obj
 * 3. 显式绑定：call/apply/bind，this 指向指定对象
 * 4. new 绑定：this 指向新创建的对象
 *
 * 箭头函数：没有自己的 this，继承外层作用域
 */

/**
 * 6. 事件循环机制？宏任务和微任务的区别？
 *
 * 事件循环：
 * 1. 执行同步代码（调用栈）
 * 2. 清空微任务队列
 * 3. 可能渲染页面
 * 4. 取一个宏任务执行
 * 5. 重复 2-4
 *
 * 宏任务：setTimeout、setInterval、I/O、UI 渲染
 * 微任务：Promise.then、MutationObserver、queueMicrotask
 *
 * 关键：微任务优先级高于宏任务
 *
 * ⚠️ 易错点：
 * - setTimeout(fn, 0) 不是立即执行，最小延迟 4ms
 * - Promise 构造函数中的代码是同步执行的
 * - async/await 后面的代码相当于 then 回调（微任务）
 * - Node.js 事件循环和浏览器有区别（process.nextTick、setImmediate）
 *
 * 💡 面试追问：
 * Q: 微任务执行过程中产生的新微任务如何处理？
 * A: 会在当前微任务队列中继续执行，直到清空
 *
 * Q: requestAnimationFrame 是宏任务还是微任务？
 * A: 都不是，它在渲染前执行，每帧一次
 */

/**
 * 7. Promise 的原理？async/await 的本质？
 *
 * Promise：
 * - 三种状态：pending、fulfilled、rejected
 * - 状态不可逆
 * - then 返回新的 Promise
 *
 * async/await：
 * - async 函数返回 Promise
 * - await 相当于 Promise.then
 * - 本质是 Generator + Promise 的语法糖
 */

/**
 * 8. var、let、const 的区别？
 *
 * var：函数作用域、变量提升、可重复声明
 * let：块级作用域、暂时性死区、不可重复声明
 * const：块级作用域、声明时必须初始化、引用不可变
 */

// ============================================================
// 🔥🔥 进阶深入题
// ============================================================

/**
 * 9. 为什么 0.1 + 0.2 !== 0.3？
 *
 * 原因：JavaScript 使用 IEEE 754 双精度浮点数，
 * 0.1 和 0.2 无法精确表示，存在精度损失。
 *
 * 解决方案：
 * - 使用整数运算（乘以倍数后除回）
 * - 使用 Number.EPSILON 判断相等
 * - 使用 BigInt 或第三方库（decimal.js）
 */

/**
 * 10. new 操作符做了什么？
 *
 * 1. 创建一个空对象
 * 2. 将空对象的 __proto__ 指向构造函数的 prototype
 * 3. 将构造函数的 this 指向这个对象，执行构造函数
 * 4. 如果构造函数返回对象，则返回该对象；否则返回创建的对象
 */

/**
 * 11. 箭头函数和普通函数的区别？
 *
 * 1. 没有自己的 this，继承外层
 * 2. 没有 arguments 对象
 * 3. 不能用作构造函数（不能 new）
 * 4. 没有 prototype 属性
 * 5. 不能用作 Generator 函数
 */

/**
 * 12. ES Module 和 CommonJS 的区别？
 *
 * 加载时机：
 * - ESM：编译时加载（静态分析，支持 Tree Shaking）
 * - CJS：运行时加载
 *
 * 输出方式：
 * - ESM：值的引用
 * - CJS：值的拷贝
 *
 * 循环依赖处理也不同
 */

/**
 * 13. 什么是防抖和节流？区别是什么？
 *
 * 防抖（debounce）：
 * - 事件触发后延迟执行，如果期间再次触发则重新计时
 * - 场景：搜索输入、窗口 resize
 *
 * 节流（throttle）：
 * - 在指定时间内只执行一次
 * - 场景：滚动加载、按钮点击
 */

/**
 * 14. 深拷贝和浅拷贝的区别？如何实现深拷贝？
 *
 * 浅拷贝：只复制第一层，嵌套对象还是引用
 * - Object.assign()
 * - 展开运算符 {...obj}
 * - Array.prototype.slice()
 *
 * 深拷贝：递归复制所有层级
 * - JSON.parse(JSON.stringify())（有局限性）
 * - 递归实现（处理循环引用、特殊对象）
 * - structuredClone()（现代浏览器）
 */

// ============================================================
// 📝 经典代码输出题
// ============================================================

/**
 * 题目 1：作用域
 */
var scope = 'global';
function checkScope() {
  var scope = 'local';
  function f() {
    return scope;
  }
  return f();
}
// checkScope(); // 'local'

/**
 * 题目 2：变量提升
 */
console.log(a); // undefined
var a = 1;
console.log(a); // 1

/**
 * 题目 3：闭包经典问题
 */
for (var i = 0; i < 3; i++) {
  setTimeout(() => console.log(i), 0);
}
// 输出：3, 3, 3

// 解决方案
for (let i = 0; i < 3; i++) {
  setTimeout(() => console.log(i), 0);
}
// 输出：0, 1, 2

/**
 * 题目 4：this 指向
 */
const obj = {
  name: 'obj',
  fn1: function () {
    console.log(this.name);
  },
  fn2: () => {
    console.log(this);
  },
};
// obj.fn1(); // 'obj'
// obj.fn2(); // window 或 undefined

const fn1 = obj.fn1;
// fn1(); // undefined（this 丢失）

/**
 * 题目 5：事件循环
 */
console.log('1');
setTimeout(() => console.log('2'), 0);
Promise.resolve().then(() => console.log('3'));
console.log('4');
// 输出：1, 4, 3, 2

/**
 * 题目 6：async/await
 */
async function async1() {
  console.log('async1 start');
  await async2();
  console.log('async1 end');
}
async function async2() {
  console.log('async2');
}
console.log('script start');
setTimeout(() => console.log('setTimeout'), 0);
async1();
new Promise((resolve) => {
  console.log('promise1');
  resolve(undefined);
}).then(() => console.log('promise2'));
console.log('script end');

// 输出顺序：
// script start
// async1 start
// async2
// promise1
// script end
// async1 end
// promise2
// setTimeout

/**
 * 题目 7：[] == ![] 的结果？
 *
 * 解析：
 * 1. ![] → false（空数组是真值）
 * 2. [] == false
 * 3. [] → ''（ToPrimitive）
 * 4. '' → 0
 * 5. false → 0
 * 6. 0 == 0 → true
 */

/**
 * 题目 8：typeof null === 'object' 的原因？
 *
 * 这是 JavaScript 的历史遗留 bug。
 * 在最初实现中，值以类型标签 + 实际数据存储，
 * object 的类型标签是 000，而 null 是空指针（全 0），
 * 所以 typeof null 返回 'object'。
 */

// ============================================================
// 🎯 资深面试追问清单
// ============================================================

/**
 * 追问 1：为什么 JavaScript 是单线程的？
 *
 * JavaScript 最初是为浏览器设计的，用于操作 DOM。
 * 如果多线程同时操作 DOM，会产生竞态条件和同步问题。
 * 单线程 + 事件循环是更简单、更安全的设计。
 */

/**
 * 追问 2：为什么微任务优先级高于宏任务？
 *
 * 微任务是为了处理需要尽快执行的异步操作，
 * 如 Promise 的状态变化需要立即响应。
 * 这样可以保证 Promise 链的连续执行，
 * 避免被其他任务（如渲染）打断。
 */

/**
 * 追问 3：V8 引擎是如何优化 JavaScript 执行的？
 *
 * 1. 隐藏类（Hidden Classes）：相同结构的对象共享类型信息
 * 2. 内联缓存（Inline Caching）：缓存属性访问信息
 * 3. JIT 编译：热点代码编译为机器码
 * 4. 垃圾回收优化：分代回收、增量标记
 */

/**
 * 追问 4：如何写出 V8 友好的代码？
 *
 * 1. 避免改变对象结构（保持隐藏类稳定）
 * 2. 避免使用 delete 删除属性
 * 3. 数组避免使用稀疏数组
 * 4. 函数参数类型保持一致
 * 5. 避免大对象（会直接进入老生代）
 */

/**
 * 追问 5：为什么 Vue 3 用 Proxy 替代 Object.defineProperty？
 *
 * 1. 可以监听数组的变化（不需要重写数组方法）
 * 2. 可以监听对象属性的新增和删除
 * 3. 可以监听更多操作（has、deleteProperty 等）
 * 4. 性能更好（惰性响应式）
 *
 * 缺点：不支持 IE11
 */

/**
 * 追问 6：WeakMap 和 Map 的区别？使用场景？
 *
 * WeakMap：
 * - 键必须是对象
 * - 弱引用，不阻止垃圾回收
 * - 不可迭代，没有 size
 *
 * 使用场景：
 * - 存储 DOM 节点相关数据
 * - 对象的私有数据
 * - 缓存（自动清理）
 */

// ============================================================
// 📋 面试准备 Checklist
// ============================================================

/**
 * ✅ 数据类型
 *    - [ ] 8 种数据类型
 *    - [ ] 类型判断方法
 *    - [ ] 类型转换规则
 *
 * ✅ 作用域与闭包
 *    - [ ] 作用域链
 *    - [ ] 闭包原理与应用
 *    - [ ] 变量提升与暂时性死区
 *
 * ✅ 原型与继承
 *    - [ ] 原型链查找机制
 *    - [ ] 继承实现方式
 *    - [ ] new 操作符原理
 *
 * ✅ this
 *    - [ ] 四种绑定规则
 *    - [ ] 箭头函数的 this
 *    - [ ] call/apply/bind 实现
 *
 * ✅ 异步编程
 *    - [ ] 事件循环机制
 *    - [ ] Promise 原理
 *    - [ ] async/await 原理
 *
 * ✅ ES6+
 *    - [ ] 核心特性
 *    - [ ] 模块化
 *    - [ ] Proxy/Reflect
 *
 * ✅ 手写代码
 *    - [ ] 防抖节流
 *    - [ ] 深拷贝
 *    - [ ] Promise
 *    - [ ] call/apply/bind
 */

export {};

