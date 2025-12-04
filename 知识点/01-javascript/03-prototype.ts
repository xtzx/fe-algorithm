/**
 * ============================================================
 * 📚 原型与原型链
 * ============================================================
 *
 * 面试考察重点：
 * 1. prototype、__proto__、constructor 的关系
 * 2. 原型链的查找机制
 * 3. 继承的多种实现方式
 * 4. new 操作符的原理
 */

// ============================================================
// 1. 原型基础概念
// ============================================================

/**
 * 📖 三个核心属性
 *
 * 1. prototype（原型对象）
 *    - 只有函数才有 prototype 属性
 *    - 指向一个对象，这个对象是通过该函数创建的实例的原型
 *
 * 2. __proto__（隐式原型）
 *    - 所有对象都有 __proto__ 属性
 *    - 指向该对象的原型（即创建该对象的构造函数的 prototype）
 *    - 实际上是 Object.getPrototypeOf() 的 getter
 *
 * 3. constructor（构造函数）
 *    - 原型对象的 constructor 指向构造函数本身
 *
 * 📊 关系图：
 *
 * ┌─────────────────┐           ┌─────────────────────────────┐
 * │  构造函数 Person │           │  Person.prototype          │
 * │  ─────────────  │  prototype │  ─────────────────────     │
 * │                 │ ─────────► │  constructor ──────────────┼───┐
 * │                 │            │  sayHello()                │   │
 * └────────┬────────┘ ◄───────── └──────────────┬──────────────┘   │
 *          │           constructor              │                  │
 *          │                                    │ __proto__        │
 *          │ new                                │                  │
 *          ▼                                    │                  │
 * ┌─────────────────┐                          │                  │
 * │  实例对象 person │ ─────────────────────────┘                  │
 * │  ─────────────  │                                             │
 * │  name: 'Tom'    │                                             │
 * │  __proto__ ─────┼─────────────────────────────────────────────┘
 * └─────────────────┘
 */

// 示例代码
function Person(this: any, name: string) {
  this.name = name;
}

Person.prototype.sayHello = function () {
  console.log(`Hello, I'm ${this.name}`);
};

const person = new (Person as any)('Tom');

// 验证关系
console.log(person.__proto__ === Person.prototype); // true
console.log(Person.prototype.constructor === Person); // true
console.log(person.constructor === Person); // true（通过原型链找到）

// ============================================================
// 2. 原型链
// ============================================================

/**
 * 📖 原型链
 *
 * 当访问对象的属性时，JS 会沿着原型链查找：
 * 1. 先在对象自身查找
 * 2. 找不到就去 __proto__ 指向的原型对象查找
 * 3. 还找不到就继续往上查找
 * 4. 直到 Object.prototype（终点，其 __proto__ 为 null）
 *
 * 📊 原型链示意图：
 *
 * person 实例
 *    │
 *    │ __proto__
 *    ▼
 * Person.prototype
 *    │
 *    │ __proto__
 *    ▼
 * Object.prototype
 *    │
 *    │ __proto__
 *    ▼
 *   null
 */

// 原型链查找示例
console.log(person.name); // 'Tom'，自身属性
console.log(person.sayHello); // function，来自 Person.prototype
console.log(person.toString); // function，来自 Object.prototype
console.log(person.notExist); // undefined，原型链上都没有

// hasOwnProperty 检查自身属性
console.log(person.hasOwnProperty('name')); // true
console.log(person.hasOwnProperty('sayHello')); // false

// in 操作符检查原型链
console.log('name' in person); // true
console.log('sayHello' in person); // true

// ============================================================
// 3. 函数与对象的原型关系
// ============================================================

/**
 * 📊 完整的原型关系图
 *
 * 函数也是对象，所以函数也有 __proto__
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                                                                 │
 * │     Function                    Object                          │
 * │        │                           │                            │
 * │        │ prototype                 │ prototype                  │
 * │        ▼                           ▼                            │
 * │   Function.prototype ──────► Object.prototype ──────► null      │
 * │        ▲                           ▲                            │
 * │        │ __proto__                 │ __proto__                  │
 * │        │                           │                            │
 * │   Person(函数) ─────────────────────                            │
 * │        │                           │                            │
 * │        │ prototype                 │                            │
 * │        ▼                           │                            │
 * │   Person.prototype ────────────────┘                            │
 * │        ▲                                                        │
 * │        │ __proto__                                              │
 * │        │                                                        │
 * │   person(实例)                                                   │
 * │                                                                 │
 * └─────────────────────────────────────────────────────────────────┘
 */

// 验证
console.log(Person.__proto__ === Function.prototype); // true
console.log(Function.prototype.__proto__ === Object.prototype); // true
console.log(Object.prototype.__proto__ === null); // true

// 特殊情况：Function 和 Object 本身
console.log(Function.__proto__ === Function.prototype); // true（Function 是自己的实例）
console.log(Object.__proto__ === Function.prototype); // true（Object 也是函数）

// ============================================================
// 4. 继承的多种实现方式
// ============================================================

// 4.1 原型链继承
function Animal(this: any, name: string) {
  this.name = name;
  this.colors = ['white'];
}
Animal.prototype.sayName = function () {
  console.log(this.name);
};

function Dog1(this: any) {}
Dog1.prototype = new (Animal as any)('dog');

const dog1 = new (Dog1 as any)();
const dog2 = new (Dog1 as any)();

dog1.colors.push('black');
console.log(dog2.colors); // ['white', 'black'] ⚠️ 引用类型共享！

/**
 * 原型链继承的问题：
 * 1. 引用类型的属性被所有实例共享
 * 2. 创建子类实例时，不能向父类传参
 */

// 4.2 构造函数继承
function Cat(this: any, name: string) {
  Animal.call(this, name); // 调用父类构造函数
}

const cat1 = new (Cat as any)('Tom');
const cat2 = new (Cat as any)('Jerry');

cat1.colors.push('black');
console.log(cat2.colors); // ['white'] ✓ 不共享了

/**
 * 构造函数继承的问题：
 * 1. 只能继承父类实例属性，不能继承原型属性和方法
 * 2. 方法都在构造函数中定义，无法复用
 */

// 4.3 组合继承（最常用）
function Bird(this: any, name: string) {
  Animal.call(this, name); // 继承实例属性
}
Bird.prototype = new (Animal as any)(); // 继承原型方法
Bird.prototype.constructor = Bird; // 修复 constructor

const bird1 = new (Bird as any)('Tweety');

/**
 * 组合继承的问题：
 * 调用了两次父类构造函数，产生了多余的属性
 */

// 4.4 寄生组合继承（最佳实践）
function inheritPrototype(child: Function, parent: Function) {
  const prototype = Object.create(parent.prototype); // 创建父类原型的副本
  prototype.constructor = child; // 修复 constructor
  child.prototype = prototype; // 赋值给子类原型
}

function Fish(this: any, name: string) {
  Animal.call(this, name);
}
inheritPrototype(Fish, Animal);

const fish = new (Fish as any)('Nemo');

/**
 * 寄生组合继承的优点：
 * 1. 只调用一次父类构造函数
 * 2. 原型链保持完整
 * 3. 可以向父类传参
 */

// 4.5 ES6 class 继承
class AnimalClass {
  name: string;
  colors: string[];

  constructor(name: string) {
    this.name = name;
    this.colors = ['white'];
  }

  sayName() {
    console.log(this.name);
  }
}

class DogClass extends AnimalClass {
  breed: string;

  constructor(name: string, breed: string) {
    super(name); // 必须先调用 super
    this.breed = breed;
  }

  bark() {
    console.log('Woof!');
  }
}

const myDog = new DogClass('Buddy', 'Golden');

/**
 * 💡 追问：ES6 class 继承的本质是什么？
 *
 * 答：ES6 class 是语法糖，本质还是基于原型的继承。
 *
 * class 继承 ≈ 寄生组合继承 + 一些增强：
 * 1. 子类 __proto__ 指向父类（可以继承静态方法）
 * 2. 子类 prototype.__proto__ 指向父类 prototype
 * 3. 必须先调用 super() 才能使用 this
 */

// 验证 class 继承的原型关系
console.log(DogClass.__proto__ === AnimalClass); // true（继承静态方法）
console.log(DogClass.prototype.__proto__ === AnimalClass.prototype); // true

// ============================================================
// 5. new 操作符
// ============================================================

/**
 * 📖 new 的执行过程
 *
 * 1. 创建一个空对象
 * 2. 将空对象的 __proto__ 指向构造函数的 prototype
 * 3. 将构造函数的 this 指向这个空对象，执行构造函数
 * 4. 如果构造函数返回一个对象，则返回该对象；否则返回创建的对象
 */

// 手写 new
function myNew<T>(constructor: new (...args: any[]) => T, ...args: any[]): T {
  // 1. 创建空对象，并将其 __proto__ 指向构造函数的 prototype
  const obj = Object.create(constructor.prototype);

  // 2. 执行构造函数，绑定 this
  const result = constructor.apply(obj, args);

  // 3. 如果构造函数返回对象，则返回该对象；否则返回创建的对象
  return result instanceof Object ? result : obj;
}

// 测试
function TestClass(this: any, name: string) {
  this.name = name;
}
TestClass.prototype.sayName = function () {
  console.log(this.name);
};

const test = myNew(TestClass as any, 'test');
console.log(test.name); // 'test'
test.sayName(); // 'test'

// ============================================================
// 6. instanceof 原理
// ============================================================

/**
 * 📖 instanceof 的原理
 *
 * 检查右边构造函数的 prototype 是否在左边对象的原型链上
 */

function myInstanceof(left: any, right: Function): boolean {
  // 基本类型直接返回 false
  if (left === null || (typeof left !== 'object' && typeof left !== 'function')) {
    return false;
  }

  let proto = Object.getPrototypeOf(left);

  while (proto !== null) {
    if (proto === right.prototype) {
      return true;
    }
    proto = Object.getPrototypeOf(proto);
  }

  return false;
}

// 测试
console.log(myInstanceof([], Array)); // true
console.log(myInstanceof([], Object)); // true
console.log(myInstanceof({}, Array)); // false

// ============================================================
// 7. 高频面试题
// ============================================================

/**
 * 题目 1：下面代码输出什么？
 */
function Foo() {}
const foo1 = new (Foo as any)();
const foo2 = new (Foo as any)();

Foo.prototype.bar = 'bar';
console.log(foo1.bar); // 'bar'
console.log(foo2.bar); // 'bar'

Foo.prototype = { baz: 'baz' };
console.log(foo1.bar); // 'bar'（foo1 的 __proto__ 仍指向旧的原型）
console.log(foo1.baz); // undefined

const foo3 = new (Foo as any)();
console.log(foo3.bar); // undefined
console.log(foo3.baz); // 'baz'

/**
 * 题目 2：实现 Object.create
 */
function objectCreate(proto: object | null, propertiesObject?: PropertyDescriptorMap) {
  if (typeof proto !== 'object' && proto !== null) {
    throw new TypeError('Object prototype may only be an Object or null');
  }

  function F() {}
  F.prototype = proto;
  const obj = new (F as any)();

  if (propertiesObject !== undefined) {
    Object.defineProperties(obj, propertiesObject);
  }

  return obj;
}

/**
 * 题目 3：如何判断一个属性是自身的还是原型链上的？
 */
const obj = { a: 1 };
console.log(obj.hasOwnProperty('a')); // true
console.log(obj.hasOwnProperty('toString')); // false

// 更安全的写法（避免 hasOwnProperty 被覆盖）
console.log(Object.prototype.hasOwnProperty.call(obj, 'a')); // true
// 或 ES2022+
console.log(Object.hasOwn(obj, 'a')); // true

/**
 * 题目 4：如何实现一个不能被继承的类？
 */
class FinalClass {
  constructor() {
    if (new.target !== FinalClass) {
      throw new Error('FinalClass cannot be inherited');
    }
  }
}

// class ChildClass extends FinalClass {
//   constructor() {
//     super(); // Error!
//   }
// }

export {
  myNew,
  myInstanceof,
  objectCreate,
  inheritPrototype,
};

