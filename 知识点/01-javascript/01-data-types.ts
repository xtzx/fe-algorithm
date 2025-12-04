/**
 * ============================================================
 * 📚 JavaScript 数据类型与类型系统
 * ============================================================
 *
 * 面试考察重点：
 * 1. 基本类型 vs 引用类型的区别
 * 2. 类型判断的多种方式
 * 3. 类型转换规则（隐式/显式）
 * 4. == vs === 的区别
 */

// ============================================================
// 1. 数据类型概览
// ============================================================

/**
 * 📊 JavaScript 8 种数据类型
 *
 * 基本类型（7种）：
 * ┌─────────────┬───────────────────────────────────────────────────┐
 * │   类型       │   说明                                            │
 * ├─────────────┼───────────────────────────────────────────────────┤
 * │ undefined   │ 未定义，变量声明但未赋值                            │
 * │ null        │ 空值，表示"无"                                     │
 * │ boolean     │ 布尔值 true/false                                 │
 * │ number      │ 数字，包括整数、浮点数、NaN、Infinity              │
 * │ string      │ 字符串                                            │
 * │ symbol      │ ES6，唯一标识符                                   │
 * │ bigint      │ ES2020，任意精度整数                              │
 * └─────────────┴───────────────────────────────────────────────────┘
 *
 * 引用类型（1种）：
 * ┌─────────────┬───────────────────────────────────────────────────┐
 * │ object      │ 对象，包括普通对象、数组、函数、Date、RegExp 等    │
 * └─────────────┴───────────────────────────────────────────────────┘
 */

// ============================================================
// 2. 基本类型 vs 引用类型
// ============================================================

/**
 * 🔑 核心区别：存储方式和赋值行为
 *
 * 【基本类型】
 * - 存储在栈（Stack）中
 * - 按值访问，赋值时复制值
 * - 不可变（immutable）
 *
 * 【引用类型】
 * - 值存储在堆（Heap）中，栈中存储指向堆的引用（指针）
 * - 按引用访问，赋值时复制引用
 * - 可变（mutable）
 *
 * 📊 内存示意图：
 *
 * 栈 Stack              堆 Heap
 * ┌──────────────┐      ┌──────────────────────┐
 * │ a = 1        │      │                      │
 * │ b = 1        │      │  ┌────────────────┐  │
 * │              │      │  │ { name: 'Tom' }│  │
 * │ obj1 ───────────────│──┤                │  │
 * │ obj2 ───────────────│──┘                │  │
 * └──────────────┘      └──────────────────────┘
 *
 * obj1 和 obj2 指向同一个对象！
 */

// 示例：基本类型赋值
let a = 1;
let b = a; // 复制值
b = 2;
console.log(a); // 1，a 不受影响

// 示例：引用类型赋值
let obj1 = { name: 'Tom' };
let obj2 = obj1; // 复制引用
obj2.name = 'Jerry';
console.log(obj1.name); // 'Jerry'，obj1 也被修改了！

/**
 * 💡 追问：为什么基本类型存在栈中，引用类型存在堆中？
 *
 * 答：
 * 1. 栈内存：大小固定，由系统自动分配和释放，速度快
 *    - 基本类型大小固定（如 number 是 64 位），适合存在栈中
 *
 * 2. 堆内存：大小不固定，由程序员控制（JS 中由 GC 回收），速度慢
 *    - 引用类型大小不固定（对象可以有任意多属性），适合存在堆中
 */

// ============================================================
// 3. 类型判断
// ============================================================

/**
 * 📊 四种类型判断方式对比
 *
 * ┌─────────────────────┬────────────┬────────────┬─────────────────────┐
 * │       方式          │  能判断     │  不能判断   │       说明           │
 * ├─────────────────────┼────────────┼────────────┼─────────────────────┤
 * │ typeof              │ 基本类型    │ null,数组  │ null 返回 'object'  │
 * │ instanceof          │ 引用类型    │ 基本类型   │ 检查原型链          │
 * │ constructor         │ 大部分类型  │ null,undef │ 可被修改            │
 * │ Object.prototype.   │ 所有类型    │ 无         │ 最准确，推荐        │
 * │   toString.call()   │            │            │                     │
 * └─────────────────────┴────────────┴────────────┴─────────────────────┘
 */

// 3.1 typeof
console.log(typeof undefined); // 'undefined'
console.log(typeof null); // 'object' ⚠️ 历史遗留 bug
console.log(typeof true); // 'boolean'
console.log(typeof 42); // 'number'
console.log(typeof 'str'); // 'string'
console.log(typeof Symbol()); // 'symbol'
console.log(typeof 42n); // 'bigint'
console.log(typeof {}); // 'object'
console.log(typeof []); // 'object' ⚠️ 无法区分数组
console.log(typeof function () {}); // 'function'

/**
 * 💡 追问：为什么 typeof null === 'object'？
 *
 * 答：这是 JavaScript 的一个历史遗留 bug。
 *
 * 在 JS 最初的实现中，值是由一个类型标签和实际数据组成的：
 * - 000: object
 * - 1: int
 * - 010: double
 * - 100: string
 * - 110: boolean
 *
 * null 的值是机器码 NULL 指针（全是 0），所以类型标签也是 000，
 * 被判断为 object。
 *
 * 这个 bug 无法修复，因为修复会破坏大量现有代码。
 */

// 3.2 instanceof
console.log([] instanceof Array); // true
console.log([] instanceof Object); // true
console.log({} instanceof Object); // true
console.log(function () {} instanceof Function); // true

// instanceof 原理：检查右边构造函数的 prototype 是否在左边对象的原型链上
function myInstanceof(left: any, right: any): boolean {
  if (left === null || typeof left !== 'object') return false;
  let proto = Object.getPrototypeOf(left);
  while (proto !== null) {
    if (proto === right.prototype) return true;
    proto = Object.getPrototypeOf(proto);
  }
  return false;
}

// 3.3 Object.prototype.toString.call() - 最准确的方式
const getType = (value: unknown): string => {
  return Object.prototype.toString.call(value).slice(8, -1).toLowerCase();
};

console.log(getType(undefined)); // 'undefined'
console.log(getType(null)); // 'null'
console.log(getType(true)); // 'boolean'
console.log(getType(42)); // 'number'
console.log(getType('str')); // 'string'
console.log(getType(Symbol())); // 'symbol'
console.log(getType(42n)); // 'bigint'
console.log(getType({})); // 'object'
console.log(getType([])); // 'array'
console.log(getType(function () {})); // 'function'
console.log(getType(new Date())); // 'date'
console.log(getType(/regex/)); // 'regexp'

// ============================================================
// 4. 类型转换
// ============================================================

/**
 * 📊 类型转换规则
 *
 * JavaScript 中有三种类型转换：
 * 1. 转布尔值（ToBoolean）
 * 2. 转数字（ToNumber）
 * 3. 转字符串（ToString）
 */

// 4.1 转布尔值
// 假值（falsy）：undefined, null, false, 0, -0, NaN, ''
// 其他都是真值（truthy），包括 [], {}

console.log(Boolean(undefined)); // false
console.log(Boolean(null)); // false
console.log(Boolean(0)); // false
console.log(Boolean('')); // false
console.log(Boolean(NaN)); // false
console.log(Boolean([])); // true ⚠️ 空数组是真值
console.log(Boolean({})); // true ⚠️ 空对象是真值

// 4.2 转数字
console.log(Number(undefined)); // NaN
console.log(Number(null)); // 0
console.log(Number(true)); // 1
console.log(Number(false)); // 0
console.log(Number('')); // 0
console.log(Number('123')); // 123
console.log(Number('123abc')); // NaN
console.log(Number([])); // 0
console.log(Number([1])); // 1
console.log(Number([1, 2])); // NaN
console.log(Number({})); // NaN

// 4.3 转字符串
console.log(String(undefined)); // 'undefined'
console.log(String(null)); // 'null'
console.log(String(true)); // 'true'
console.log(String(123)); // '123'
console.log(String([])); // ''
console.log(String([1, 2])); // '1,2'
console.log(String({})); // '[object Object]'

// ============================================================
// 5. 隐式类型转换
// ============================================================

/**
 * 📊 隐式类型转换发生的场景：
 *
 * 1. 算术运算符（+, -, *, /, %）
 * 2. 比较运算符（==, <, >, <=, >=）
 * 3. 逻辑运算符（!, &&, ||）
 * 4. 条件语句（if, while, for, ? :）
 */

// 5.1 + 运算符
// 规则：如果有字符串，转字符串拼接；否则转数字相加
console.log(1 + '2'); // '12'
console.log(1 + 2); // 3
console.log('1' + 2); // '12'
console.log(1 + true); // 2
console.log(1 + null); // 1
console.log(1 + undefined); // NaN
console.log([] + []); // ''
console.log([] + {}); // '[object Object]'
console.log({} + []); // '[object Object]' 或 0（取决于解析为语句还是表达式）

// 5.2 == 运算符（抽象相等）
// 规则：会进行类型转换
console.log(1 == '1'); // true
console.log(1 == true); // true
console.log(0 == false); // true
console.log(0 == ''); // true
console.log(null == undefined); // true
console.log([] == false); // true
console.log([] == 0); // true
console.log([] == ''); // true

/**
 * 📊 == 类型转换规则：
 *
 * 1. null == undefined → true（特殊规则）
 * 2. null/undefined 和其他值比较 → false
 * 3. NaN == 任何值（包括 NaN）→ false
 * 4. 布尔值 → 转数字后比较
 * 5. 字符串 vs 数字 → 字符串转数字
 * 6. 对象 vs 基本类型 → 对象调用 ToPrimitive
 *
 * ToPrimitive 规则：
 * - 如果有 Symbol.toPrimitive 方法，调用它
 * - 否则先调用 valueOf()，如果返回基本类型，使用它
 * - 否则调用 toString()，如果返回基本类型，使用它
 * - 否则报错
 */

// [] == false 的过程：
// 1. false → 0
// 2. [] → '' (ToPrimitive，调用 toString)
// 3. '' → 0
// 4. 0 == 0 → true

// 5.3 === 运算符（严格相等）
// 规则：不进行类型转换，类型和值都必须相等
console.log(1 === '1'); // false
console.log(1 === 1); // true
console.log(null === undefined); // false
console.log(NaN === NaN); // false ⚠️

/**
 * 💡 追问：如何判断一个值是 NaN？
 *
 * 1. Number.isNaN(value) - 推荐
 * 2. value !== value（NaN 是唯一不等于自身的值）
 * 3. Object.is(value, NaN)
 *
 * 注意：全局 isNaN() 会先转数字，isNaN('abc') 返回 true
 */

// ============================================================
// 6. 特殊值
// ============================================================

// 6.1 null vs undefined
/**
 * 设计意图：
 * - undefined: "缺少值"，变量声明了但没赋值
 * - null: "空值"，主动赋值为"无"
 *
 * 使用建议：
 * - 不要主动给变量赋值 undefined
 * - 需要表示"空"时使用 null
 */

// 6.2 NaN
/**
 * NaN = Not a Number
 * - typeof NaN === 'number'
 * - NaN !== NaN（唯一不等于自身的值）
 * - 任何包含 NaN 的运算结果都是 NaN
 */

// 6.3 BigInt
/**
 * ES2020 引入，用于表示任意精度整数
 * - 不能和 Number 直接运算
 * - 不能使用 Math 方法
 */
const big1 = 9007199254740991n; // 字面量
const big2 = BigInt('9007199254740991'); // 构造函数
// console.log(big1 + 1); // Error!
console.log(big1 + 1n); // 9007199254740992n

// 6.4 Symbol
/**
 * ES6 引入，表示唯一标识符
 * - 主要用于对象属性名，避免命名冲突
 * - Symbol.for() 可创建共享的 Symbol
 */
const sym1 = Symbol('desc');
const sym2 = Symbol('desc');
console.log(sym1 === sym2); // false

const sym3 = Symbol.for('shared');
const sym4 = Symbol.for('shared');
console.log(sym3 === sym4); // true

// ============================================================
// 7. 高频面试题
// ============================================================

/**
 * 题目 1：[] == ![] 的结果？
 *
 * 解析：
 * 1. ![] → false（空数组是真值，取反为 false）
 * 2. [] == false
 * 3. false → 0
 * 4. [] → ''（ToPrimitive）
 * 5. '' → 0
 * 6. 0 == 0 → true
 *
 * 答案：true
 */

/**
 * 题目 2：实现一个完整的类型判断函数
 */
function getTypeComplete(value: unknown): string {
  // null 特殊处理
  if (value === null) return 'null';

  // 基本类型使用 typeof
  const type = typeof value;
  if (type !== 'object') return type;

  // 引用类型使用 toString
  return Object.prototype.toString.call(value).slice(8, -1).toLowerCase();
}

/**
 * 题目 3：如何判断一个变量是数组？
 */
// 方法 1：Array.isArray() - 推荐
Array.isArray([]);

// 方法 2：instanceof（跨 iframe 会失效）
[] instanceof Array;

// 方法 3：Object.prototype.toString.call()
Object.prototype.toString.call([]) === '[object Array]';

// 方法 4：constructor（可被修改）
[].constructor === Array;

/**
 * 题目 4：Object.is() vs === 的区别？
 *
 * Object.is() 修复了 === 的两个"bug"：
 * - Object.is(NaN, NaN) → true（=== 返回 false）
 * - Object.is(+0, -0) → false（=== 返回 true）
 */
console.log(NaN === NaN); // false
console.log(Object.is(NaN, NaN)); // true
console.log(+0 === -0); // true
console.log(Object.is(+0, -0)); // false

// ============================================================
// 8. 实战应用
// ============================================================

/**
 * 场景 1：类型安全的工具函数
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isFunction(value: unknown): value is Function {
  return typeof value === 'function';
}

function isEmpty(value: unknown): boolean {
  if (value == null) return true;
  if (typeof value === 'string' || Array.isArray(value)) return value.length === 0;
  if (value instanceof Map || value instanceof Set) return value.size === 0;
  if (isObject(value)) return Object.keys(value).length === 0;
  return false;
}

/**
 * 场景 2：安全的类型转换
 */
function safeToNumber(value: unknown, defaultValue = 0): number {
  if (value === null || value === undefined) return defaultValue;
  const num = Number(value);
  return Number.isNaN(num) ? defaultValue : num;
}

function safeToString(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

export {
  getType,
  getTypeComplete,
  myInstanceof,
  isObject,
  isFunction,
  isEmpty,
  safeToNumber,
  safeToString,
};

