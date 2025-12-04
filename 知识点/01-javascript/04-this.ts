/**
 * ============================================================
 * 📚 this 指向
 * ============================================================
 *
 * 面试考察重点：
 * 1. this 的绑定规则
 * 2. 箭头函数的 this
 * 3. call/apply/bind 的使用和实现
 * 4. 各种场景下 this 的判断
 */

// ============================================================
// 1. this 是什么？
// ============================================================

/**
 * 📖 this 是执行上下文的一个属性
 *
 * this 的值在函数调用时确定，取决于函数的调用方式，而非定义位置。
 * （箭头函数除外，箭头函数的 this 在定义时确定）
 *
 * 💡 为什么需要 this？
 *
 * this 提供了一种更优雅的方式来隐式"传递"一个对象引用，
 * 让 API 设计更加简洁，避免显式传递上下文对象。
 */

// ============================================================
// 2. this 绑定规则
// ============================================================

/**
 * 📊 四种绑定规则（优先级从低到高）
 *
 * ┌─────────────┬────────────────────────────────────────────────────┐
 * │ 规则         │ 说明                                              │
 * ├─────────────┼────────────────────────────────────────────────────┤
 * │ 默认绑定     │ 独立函数调用，this 指向全局对象（严格模式为 undefined）│
 * │ 隐式绑定     │ 作为对象方法调用，this 指向调用的对象                │
 * │ 显式绑定     │ call/apply/bind，this 指向指定的对象               │
 * │ new 绑定     │ 构造函数调用，this 指向新创建的对象                  │
 * └─────────────┴────────────────────────────────────────────────────┘
 *
 * 优先级：new > 显式绑定 > 隐式绑定 > 默认绑定
 */

// 2.1 默认绑定
function defaultBinding() {
  console.log(this); // 非严格模式：window/global；严格模式：undefined
}
defaultBinding();

// 2.2 隐式绑定
const obj1 = {
  name: 'obj1',
  sayName() {
    console.log(this.name);
  },
};
obj1.sayName(); // 'obj1'

// 隐式丢失问题
const fn = obj1.sayName;
fn(); // undefined（this 指向全局或 undefined）

// 回调函数中的隐式丢失
function doCallback(callback: Function) {
  callback(); // 默认绑定
}
doCallback(obj1.sayName); // undefined

// 2.3 显式绑定
function greet(greeting: string, punctuation: string) {
  console.log(`${greeting}, ${this.name}${punctuation}`);
}

const person1 = { name: 'Tom' };

// call：参数逐个传递
greet.call(person1, 'Hello', '!'); // 'Hello, Tom!'

// apply：参数以数组传递
greet.apply(person1, ['Hi', '?']); // 'Hi, Tom?'

// bind：返回绑定后的新函数
const boundGreet = greet.bind(person1, 'Hey');
boundGreet('~'); // 'Hey, Tom~'

// 2.4 new 绑定
function PersonConstructor(this: any, name: string) {
  this.name = name;
  console.log(this); // 新创建的对象 { name: 'Tom' }
}
const p = new (PersonConstructor as any)('Tom');

// ============================================================
// 3. 箭头函数的 this
// ============================================================

/**
 * 📖 箭头函数没有自己的 this
 *
 * 箭头函数的 this 继承自外层作用域，在定义时就确定了（词法绑定）。
 * - 不能用 call/apply/bind 改变 this
 * - 不能用作构造函数
 */

const obj2 = {
  name: 'obj2',
  // 普通函数
  regularFn() {
    console.log('regular:', this.name);
  },
  // 箭头函数
  arrowFn: () => {
    console.log('arrow:', this); // 外层作用域的 this
  },
  // 嵌套情况
  nested() {
    const inner = () => {
      console.log('nested arrow:', this.name); // 继承 nested 的 this
    };
    inner();
  },
};

obj2.regularFn(); // 'regular: obj2'
obj2.arrowFn(); // 'arrow: undefined' 或 window
obj2.nested(); // 'nested arrow: obj2'

// 常见用途：避免 this 丢失
class Timer {
  seconds = 0;

  start() {
    // 使用箭头函数，this 指向 Timer 实例
    setInterval(() => {
      this.seconds++;
      console.log(this.seconds);
    }, 1000);
  }
}

// ============================================================
// 4. 手写 call/apply/bind
// ============================================================

// 4.1 手写 call
Function.prototype.myCall = function (context: any, ...args: any[]) {
  // 如果 context 为 null/undefined，默认为全局对象
  context = context ?? globalThis;
  // 基本类型转为对象
  context = Object(context);

  // 用 Symbol 避免属性名冲突
  const fnKey = Symbol('fn');
  context[fnKey] = this;

  // 调用函数
  const result = context[fnKey](...args);

  // 删除临时属性
  delete context[fnKey];

  return result;
};

// 4.2 手写 apply
Function.prototype.myApply = function (context: any, args: any[] = []) {
  context = context ?? globalThis;
  context = Object(context);

  const fnKey = Symbol('fn');
  context[fnKey] = this;

  const result = context[fnKey](...args);

  delete context[fnKey];

  return result;
};

// 4.3 手写 bind
Function.prototype.myBind = function (context: any, ...args: any[]) {
  const fn = this;

  const boundFn = function (this: any, ...innerArgs: any[]) {
    // 判断是否作为构造函数调用（new 绑定优先级高于显式绑定）
    const isNew = this instanceof boundFn;

    return fn.apply(isNew ? this : context, [...args, ...innerArgs]);
  };

  // 维护原型关系
  if (fn.prototype) {
    boundFn.prototype = Object.create(fn.prototype);
  }

  return boundFn;
};

// 声明类型
declare global {
  interface Function {
    myCall(context: any, ...args: any[]): any;
    myApply(context: any, args?: any[]): any;
    myBind(context: any, ...args: any[]): Function;
  }
}

// ============================================================
// 5. 特殊场景的 this
// ============================================================

// 5.1 DOM 事件处理函数
/**
 * <button onclick="console.log(this)">Click</button>
 * // this 指向 button 元素
 *
 * button.addEventListener('click', function() {
 *   console.log(this); // this 指向 button 元素
 * });
 *
 * button.addEventListener('click', () => {
 *   console.log(this); // 箭头函数，this 指向外层作用域
 * });
 */

// 5.2 定时器
const obj3 = {
  name: 'obj3',
  // 普通函数作为回调，this 丢失
  delayLog1() {
    setTimeout(function () {
      console.log(this.name); // undefined
    }, 100);
  },
  // 箭头函数保持 this
  delayLog2() {
    setTimeout(() => {
      console.log(this.name); // 'obj3'
    }, 100);
  },
  // 手动绑定
  delayLog3() {
    setTimeout(
      function () {
        console.log(this.name); // 'obj3'
      }.bind(this),
      100
    );
  },
};

// 5.3 类中的 this
class MyClass {
  name = 'MyClass';

  // 普通方法
  regularMethod() {
    console.log(this.name);
  }

  // 箭头函数属性（每个实例都有自己的一份）
  arrowMethod = () => {
    console.log(this.name);
  };
}

const instance = new MyClass();
const { regularMethod, arrowMethod } = instance;

// regularMethod(); // Error: Cannot read property 'name' of undefined
arrowMethod(); // 'MyClass'（箭头函数绑定了实例）

// ============================================================
// 6. React 中的 this 问题
// ============================================================

/**
 * 📊 React 类组件中 this 丢失的解决方案
 *
 * class MyComponent extends React.Component {
 *   constructor() {
 *     super();
 *     // 方案 1：构造函数中 bind
 *     this.handleClick1 = this.handleClick1.bind(this);
 *   }
 *
 *   // 方案 2：箭头函数（推荐）
 *   handleClick2 = () => {
 *     console.log(this);
 *   }
 *
 *   handleClick1() {
 *     console.log(this);
 *   }
 *
 *   render() {
 *     return (
 *       <>
 *         <button onClick={this.handleClick1}>方案 1</button>
 *         <button onClick={this.handleClick2}>方案 2</button>
 *         {// 方案 3：render 中 bind（每次渲染创建新函数，不推荐）}
 *         <button onClick={this.handleClick1.bind(this)}>方案 3</button>
 *         {// 方案 4：render 中箭头函数（每次渲染创建新函数，不推荐）}
 *         <button onClick={() => this.handleClick1()}>方案 4</button>
 *       </>
 *     );
 *   }
 * }
 *
 * 推荐方案 2：箭头函数属性
 * - 语法简洁
 * - 不会每次渲染创建新函数
 * - 缺点：每个实例都有一份，不在原型上
 */

// ============================================================
// 7. 高频面试题
// ============================================================

/**
 * 题目 1：下面代码输出什么？
 */
var name = 'global';

const obj4 = {
  name: 'obj4',
  fn1: function () {
    console.log(this.name);
  },
  fn2: () => {
    console.log(this.name);
  },
  fn3: function () {
    return function () {
      console.log(this.name);
    };
  },
  fn4: function () {
    return () => {
      console.log(this.name);
    };
  },
};

// obj4.fn1();      // 'obj4'（隐式绑定）
// obj4.fn2();      // 'global' 或 undefined（箭头函数，外层 this）
// obj4.fn3()();    // 'global' 或 undefined（返回的函数独立调用）
// obj4.fn4()();    // 'obj4'（箭头函数继承 fn4 的 this）

/**
 * 题目 2：下面代码输出什么？
 */
function Foo2(this: any) {
  this.name = 'Foo2';
  return {
    name: 'returned',
    getName: () => {
      console.log(this.name);
    },
  };
}

const foo2 = new (Foo2 as any)();
// foo2.getName(); // 'Foo2'
// 解析：箭头函数的 this 在 Foo2 内部定义时确定，指向 new 创建的对象
// 虽然 new 返回了另一个对象，但箭头函数的 this 仍然指向原来 new 创建的对象

/**
 * 题目 3：实现 softBind
 *
 * softBind：如果 this 指向全局或 undefined，则使用绑定的 context；
 * 否则使用调用时的 this（允许隐式绑定覆盖）
 */
Function.prototype.softBind = function (context: any, ...args: any[]) {
  const fn = this;

  const boundFn = function (this: any, ...innerArgs: any[]) {
    // 如果 this 是全局对象或 undefined，使用绑定的 context
    const useContext =
      !this || this === globalThis || this === (typeof window !== 'undefined' ? window : global)
        ? context
        : this;

    return fn.apply(useContext, [...args, ...innerArgs]);
  };

  boundFn.prototype = Object.create(fn.prototype);

  return boundFn;
};

declare global {
  interface Function {
    softBind(context: any, ...args: any[]): Function;
  }
}

/**
 * 题目 4：实现一个能绑定多次的 bind（链式绑定）
 */
function chainBind(this: Function, ...contexts: any[]) {
  const fn = this;

  return function (this: any, ...args: any[]) {
    // 从后往前应用 context
    return contexts.reduceRight((acc, ctx) => {
      return fn.call(ctx, ...args);
    }, undefined);
  };
}

export {};

