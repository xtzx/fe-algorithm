/**
 * ============================================================
 * 📚 ES6+ 新特性
 * ============================================================
 *
 * 面试考察重点：
 * 1. ES6 核心特性（let/const、解构、箭头函数、class、模块化等）
 * 2. ES6+ 常用特性（可选链、空值合并、Promise.allSettled 等）
 * 3. 各特性的原理和使用场景
 */

// ============================================================
// 1. let / const / var 区别
// ============================================================

/**
 * 已在 02-scope-closure.ts 详细讲解
 *
 * 核心区别：
 * - var：函数作用域、变量提升、可重复声明
 * - let：块级作用域、暂时性死区、不可重复声明
 * - const：块级作用域、声明时必须初始化、不可重新赋值
 */

// ============================================================
// 2. 解构赋值
// ============================================================

// 2.1 数组解构
const [a, b, ...rest] = [1, 2, 3, 4, 5];
console.log(a, b, rest); // 1, 2, [3, 4, 5]

// 默认值
const [x = 1, y = 2] = [undefined, null];
console.log(x, y); // 1, null（只有 undefined 才会使用默认值）

// 交换变量
let m = 1,
  n = 2;
[m, n] = [n, m];

// 2.2 对象解构
const { name, age: userAge, job = 'engineer' } = { name: 'Tom', age: 18 };
console.log(name, userAge, job); // 'Tom', 18, 'engineer'

// 嵌套解构
const {
  user: { name: userName },
} = { user: { name: 'Jerry' } };
console.log(userName); // 'Jerry'

// 2.3 函数参数解构
function greet({ name, greeting = 'Hello' }: { name: string; greeting?: string }) {
  console.log(`${greeting}, ${name}!`);
}

// ============================================================
// 3. 展开运算符 / 剩余参数
// ============================================================

// 3.1 展开数组
const arr1 = [1, 2, 3];
const arr2 = [...arr1, 4, 5]; // [1, 2, 3, 4, 5]

// 数组浅拷贝
const arrCopy = [...arr1];

// 3.2 展开对象（ES2018）
const obj1 = { a: 1, b: 2 };
const obj2 = { ...obj1, c: 3 }; // { a: 1, b: 2, c: 3 }

// 对象合并
const merged = { ...obj1, ...obj2 };

// 3.3 剩余参数
function sum(...nums: number[]) {
  return nums.reduce((a, b) => a + b, 0);
}

// ============================================================
// 4. 箭头函数
// ============================================================

/**
 * 箭头函数特点：
 * 1. 没有自己的 this（继承外层）
 * 2. 没有 arguments 对象
 * 3. 不能用作构造函数（不能 new）
 * 4. 没有 prototype 属性
 * 5. 不能用作 Generator 函数
 */

// 简写形式
const add = (a: number, b: number) => a + b;

// 返回对象需要加括号
const createObj = (name: string) => ({ name });

// 💡 追问：什么时候不适合用箭头函数？
// 1. 需要动态 this 的场景（事件处理、对象方法）
// 2. 需要 arguments 的场景
// 3. 需要构造函数的场景

// ============================================================
// 5. 模板字符串
// ============================================================

const name1 = 'World';
const greeting1 = `Hello, ${name1}!`; // Hello, World!

// 多行字符串
const multiLine = `
  line 1
  line 2
`;

// 标签模板（Tagged Template）
function highlight(strings: TemplateStringsArray, ...values: any[]) {
  return strings.reduce((result, str, i) => {
    return result + str + (values[i] !== undefined ? `<mark>${values[i]}</mark>` : '');
  }, '');
}

const name2 = 'Tom';
const age1 = 18;
const result = highlight`My name is ${name2} and I'm ${age1} years old.`;
// "My name is <mark>Tom</mark> and I'm <mark>18</mark> years old."

// ============================================================
// 6. class 类
// ============================================================

class Animal {
  // 公有字段
  name: string;
  // 私有字段（ES2022）
  #privateField = 'private';
  // 静态属性
  static species = 'Animal';

  constructor(name: string) {
    this.name = name;
  }

  // 实例方法
  speak() {
    console.log(`${this.name} makes a sound.`);
  }

  // 私有方法（ES2022）
  #privateMethod() {
    return this.#privateField;
  }

  // 静态方法
  static create(name: string) {
    return new Animal(name);
  }

  // getter/setter
  get info() {
    return `Animal: ${this.name}`;
  }

  set info(value: string) {
    this.name = value;
  }
}

// 继承
class Dog extends Animal {
  breed: string;

  constructor(name: string, breed: string) {
    super(name); // 必须先调用 super
    this.breed = breed;
  }

  // 方法重写
  speak() {
    console.log(`${this.name} barks.`);
  }
}

// ============================================================
// 7. Symbol
// ============================================================

// 创建唯一标识符
const sym1 = Symbol('description');
const sym2 = Symbol('description');
console.log(sym1 === sym2); // false

// 全局 Symbol 注册表
const globalSym1 = Symbol.for('global');
const globalSym2 = Symbol.for('global');
console.log(globalSym1 === globalSym2); // true

// 内置 Symbol
// Symbol.iterator - 定义迭代器
// Symbol.toStringTag - 定义 toString 标签
// Symbol.toPrimitive - 定义类型转换行为

const obj = {
  [Symbol.toPrimitive](hint: string) {
    if (hint === 'number') return 42;
    if (hint === 'string') return 'hello';
    return true;
  },
};

console.log(+obj); // 42
console.log(`${obj}`); // 'hello'

// ============================================================
// 8. Iterator 与 Generator
// ============================================================

// 8.1 迭代器协议
const iterable = {
  [Symbol.iterator]() {
    let i = 0;
    return {
      next() {
        if (i < 3) {
          return { value: i++, done: false };
        }
        return { value: undefined, done: true };
      },
    };
  },
};

for (const value of iterable) {
  console.log(value); // 0, 1, 2
}

// 8.2 Generator 函数
function* gen() {
  yield 1;
  yield 2;
  yield 3;
}

const g = gen();
console.log(g.next()); // { value: 1, done: false }
console.log(g.next()); // { value: 2, done: false }
console.log(g.next()); // { value: 3, done: false }
console.log(g.next()); // { value: undefined, done: true }

// Generator 实现无限序列
function* fibonacci() {
  let [prev, curr] = [0, 1];
  while (true) {
    yield curr;
    [prev, curr] = [curr, prev + curr];
  }
}

// ============================================================
// 9. Map / Set / WeakMap / WeakSet
// ============================================================

// 9.1 Map
const map = new Map<string, number>();
map.set('a', 1);
map.set('b', 2);
console.log(map.get('a')); // 1
console.log(map.has('b')); // true
console.log(map.size); // 2

// Map vs Object
// - Map 的键可以是任意类型
// - Map 保持插入顺序
// - Map 有 size 属性
// - Map 更适合频繁增删

// 9.2 Set
const set = new Set([1, 2, 2, 3]);
console.log([...set]); // [1, 2, 3]（自动去重）

// 数组去重
const unique = [...new Set([1, 2, 2, 3])];

// 9.3 WeakMap / WeakSet
// - 键必须是对象
// - 弱引用，不阻止垃圾回收
// - 不可迭代，没有 size 属性

// 使用场景：存储 DOM 节点相关数据
const wm = new WeakMap<object, any>();
// const element = document.querySelector('#id');
// wm.set(element, { clicks: 0 });
// 当 element 被移除时，WeakMap 中的数据也会被回收

// ============================================================
// 10. Proxy / Reflect
// ============================================================

// 10.1 Proxy - 代理对象
const target = { name: 'Tom', age: 18 };

const handler: ProxyHandler<typeof target> = {
  get(target, prop, receiver) {
    console.log(`Getting ${String(prop)}`);
    return Reflect.get(target, prop, receiver);
  },
  set(target, prop, value, receiver) {
    console.log(`Setting ${String(prop)} to ${value}`);
    return Reflect.set(target, prop, value, receiver);
  },
};

const proxy = new Proxy(target, handler);
proxy.name; // Getting name
proxy.age = 20; // Setting age to 20

// 10.2 Proxy 应用：响应式系统（Vue 3）
function reactive<T extends object>(target: T): T {
  return new Proxy(target, {
    get(target, prop, receiver) {
      const result = Reflect.get(target, prop, receiver);
      // 收集依赖（track）
      console.log('track', prop);
      // 递归处理嵌套对象
      if (typeof result === 'object' && result !== null) {
        return reactive(result);
      }
      return result;
    },
    set(target, prop, value, receiver) {
      const result = Reflect.set(target, prop, value, receiver);
      // 触发更新（trigger）
      console.log('trigger', prop);
      return result;
    },
  });
}

// 10.3 Reflect
// Reflect 提供了操作对象的方法，与 Proxy handler 一一对应
// - Reflect.get(target, prop)
// - Reflect.set(target, prop, value)
// - Reflect.has(target, prop)
// - Reflect.deleteProperty(target, prop)
// - Reflect.ownKeys(target)

// ============================================================
// 11. ES2017+ 重要特性
// ============================================================

// 11.1 async/await（ES2017）
// 详见 06-promise-async.ts

// 11.2 Object.values / Object.entries（ES2017）
const obj3 = { a: 1, b: 2, c: 3 };
console.log(Object.values(obj3)); // [1, 2, 3]
console.log(Object.entries(obj3)); // [['a', 1], ['b', 2], ['c', 3]]

// 11.3 String padding（ES2017）
console.log('5'.padStart(3, '0')); // '005'
console.log('5'.padEnd(3, '0')); // '500'

// 11.4 Object.getOwnPropertyDescriptors（ES2017）
const descriptors = Object.getOwnPropertyDescriptors(obj3);

// 11.5 可选链 ?.（ES2020）
const user = { profile: { name: 'Tom' } };
console.log(user?.profile?.name); // 'Tom'
console.log(user?.profile?.age); // undefined（不会报错）

// 函数调用
const fn = null;
fn?.(); // 不会报错

// 11.6 空值合并 ??（ES2020）
const value1 = null ?? 'default'; // 'default'
const value2 = 0 ?? 'default'; // 0（只有 null/undefined 才使用默认值）
const value3 = '' ?? 'default'; // ''

// ?? vs ||
const value4 = 0 || 'default'; // 'default'（0 是假值）
const value5 = 0 ?? 'default'; // 0（0 不是 null/undefined）

// 11.7 BigInt（ES2020）
const big = 9007199254740991n;
console.log(big + 1n); // 9007199254740992n

// 11.8 Promise.allSettled（ES2020）
// 详见 06-promise-async.ts

// 11.9 String.prototype.replaceAll（ES2021）
const str = 'hello hello hello';
console.log(str.replaceAll('hello', 'hi')); // 'hi hi hi'

// 11.10 逻辑赋值运算符（ES2021）
let a1 = null;
a1 ||= 'default'; // a1 = a1 || 'default'
a1 &&= 'changed'; // a1 = a1 && 'changed'
a1 ??= 'fallback'; // a1 = a1 ?? 'fallback'

// 11.11 数字分隔符（ES2021）
const billion = 1_000_000_000;

// 11.12 Array.prototype.at（ES2022）
const arr = [1, 2, 3, 4, 5];
console.log(arr.at(-1)); // 5（支持负索引）

// 11.13 Object.hasOwn（ES2022）
console.log(Object.hasOwn(obj3, 'a')); // true
// 比 obj.hasOwnProperty 更安全（不会被覆盖）

// 11.14 类私有字段和方法（ES2022）
// 已在 class 部分展示

// 11.15 Top-level await（ES2022）
// 在模块顶层直接使用 await
// const data = await fetch('/api/data');

// 11.16 Array.prototype.toSorted / toReversed / toSpliced / with（ES2023）
const arr2 = [3, 1, 2];
const sorted = arr2.toSorted(); // [1, 2, 3]，原数组不变
const reversed = arr2.toReversed(); // [2, 1, 3]，原数组不变

// ============================================================
// 12. 模块化
// ============================================================

/**
 * 📊 模块化方案对比
 *
 * ┌──────────┬─────────────────────────────────────────────────────┐
 * │  方案     │  特点                                              │
 * ├──────────┼─────────────────────────────────────────────────────┤
 * │ CommonJS │ 同步加载、运行时加载、值拷贝、Node.js 默认           │
 * │ AMD      │ 异步加载、运行时加载、浏览器端                       │
 * │ UMD      │ 兼容 CommonJS 和 AMD                               │
 * │ ESM      │ 静态分析、编译时加载、值引用、官方标准               │
 * └──────────┴─────────────────────────────────────────────────────┘
 */

// ES Module
// export { xxx };
// export default xxx;
// import { xxx } from 'module';
// import xxx from 'module';
// import * as xxx from 'module';

// 动态导入（返回 Promise）
// const module = await import('./module.js');

/**
 * 💡 追问：ESM 和 CommonJS 的区别？
 *
 * 1. 加载时机：
 *    - ESM：编译时加载（静态分析）
 *    - CJS：运行时加载
 *
 * 2. 输出：
 *    - ESM：值的引用（原模块变化会反映）
 *    - CJS：值的拷贝
 *
 * 3. this：
 *    - ESM：undefined
 *    - CJS：module.exports
 *
 * 4. 循环依赖处理不同
 */

// CommonJS 值拷贝示例
// counter.js
// let count = 0;
// module.exports = { count, increment: () => count++ };

// main.js
// const { count, increment } = require('./counter');
// increment();
// console.log(count); // 0（值拷贝，不会变）

// ESM 值引用示例
// counter.mjs
// export let count = 0;
// export const increment = () => count++;

// main.mjs
// import { count, increment } from './counter.mjs';
// increment();
// console.log(count); // 1（值引用，会变）

export { reactive, highlight };

