/**
 * ============================================================
 * 📚 常用函数手写实现
 * ============================================================
 *
 * 面试考察重点：
 * 1. call / apply / bind
 * 2. debounce / throttle
 * 3. new / instanceof
 * 4. 柯里化 / 组合函数
 */

// ============================================================
// 1. call / apply / bind
// ============================================================

/**
 * 📊 call 实现
 *
 * 原理：将函数作为对象的方法调用，从而改变 this
 */

function myCall<T, R>(
  this: (this: T, ...args: any[]) => R,
  context: T,
  ...args: any[]
): R {
  // 处理 null/undefined
  const ctx = context ?? globalThis;
  
  // 使用 Symbol 避免属性名冲突
  const key = Symbol('fn');
  
  // 将函数作为对象的方法
  (ctx as any)[key] = this;
  
  // 调用并获取结果
  const result = (ctx as any)[key](...args);
  
  // 删除临时属性
  delete (ctx as any)[key];
  
  return result;
}

// 挂载到 Function.prototype
// Function.prototype.myCall = myCall;

/**
 * 📊 apply 实现
 *
 * 与 call 的区别：参数是数组
 */

function myApply<T, R>(
  this: (this: T, ...args: any[]) => R,
  context: T,
  args?: any[]
): R {
  const ctx = context ?? globalThis;
  const key = Symbol('fn');
  
  (ctx as any)[key] = this;
  const result = args ? (ctx as any)[key](...args) : (ctx as any)[key]();
  delete (ctx as any)[key];
  
  return result;
}

/**
 * 📊 bind 实现
 *
 * 特点：
 * 1. 返回一个新函数
 * 2. 可以预设参数（柯里化）
 * 3. new 调用时 this 指向新对象
 */

function myBind<T, R>(
  this: (this: T, ...args: any[]) => R,
  context: T,
  ...args: any[]
): (...newArgs: any[]) => R {
  const fn = this;
  
  const boundFn = function(this: any, ...newArgs: any[]) {
    // 判断是否是 new 调用
    const isNew = this instanceof boundFn;
    
    // new 调用时 this 指向新对象，否则使用绑定的 context
    return fn.apply(isNew ? this : context, [...args, ...newArgs]);
  };
  
  // 保持原型链
  if (fn.prototype) {
    boundFn.prototype = Object.create(fn.prototype);
  }
  
  return boundFn;
}

// ============================================================
// 2. debounce / throttle（完整版）
// ============================================================

/**
 * 📊 防抖（Debounce）
 *
 * 延迟执行，期间再次触发则重新计时
 */

interface DebounceOptions {
  leading?: boolean;   // 开始时立即执行
  trailing?: boolean;  // 结束后执行
  maxWait?: number;    // 最大等待时间
}

function debounce<T extends (...args: any[]) => any>(
  fn: T,
  wait: number,
  options: DebounceOptions = {}
): T & { cancel: () => void; flush: () => void } {
  const { leading = false, trailing = true, maxWait } = options;
  
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let lastCallTime: number | undefined;
  let lastInvokeTime = 0;
  let lastArgs: Parameters<T> | null = null;
  let lastThis: any = null;
  let result: ReturnType<T>;

  function invokeFunc(time: number): ReturnType<T> {
    const args = lastArgs!;
    const thisArg = lastThis;
    lastArgs = lastThis = null;
    lastInvokeTime = time;
    result = fn.apply(thisArg, args);
    return result;
  }

  function shouldInvoke(time: number): boolean {
    const timeSinceLastCall = lastCallTime === undefined ? 0 : time - lastCallTime;
    const timeSinceLastInvoke = time - lastInvokeTime;

    return (
      lastCallTime === undefined ||
      timeSinceLastCall >= wait ||
      timeSinceLastCall < 0 ||
      (maxWait !== undefined && timeSinceLastInvoke >= maxWait)
    );
  }

  function trailingEdge(time: number): ReturnType<T> | undefined {
    timeoutId = null;
    if (trailing && lastArgs) {
      return invokeFunc(time);
    }
    lastArgs = lastThis = null;
    return result;
  }

  function timerExpired() {
    const time = Date.now();
    if (shouldInvoke(time)) {
      return trailingEdge(time);
    }
    const timeSinceLastCall = time - (lastCallTime || 0);
    const timeSinceLastInvoke = time - lastInvokeTime;
    const timeWaiting = wait - timeSinceLastCall;
    const remainingWait = maxWait !== undefined
      ? Math.min(timeWaiting, maxWait - timeSinceLastInvoke)
      : timeWaiting;

    timeoutId = setTimeout(timerExpired, remainingWait);
  }

  function leadingEdge(time: number): ReturnType<T> | undefined {
    lastInvokeTime = time;
    timeoutId = setTimeout(timerExpired, wait);
    return leading ? invokeFunc(time) : result;
  }

  function debounced(this: any, ...args: Parameters<T>): ReturnType<T> | undefined {
    const time = Date.now();
    const isInvoking = shouldInvoke(time);

    lastArgs = args;
    lastThis = this;
    lastCallTime = time;

    if (isInvoking) {
      if (timeoutId === null) {
        return leadingEdge(time);
      }
      if (maxWait !== undefined) {
        timeoutId = setTimeout(timerExpired, wait);
        return invokeFunc(time);
      }
    }
    if (timeoutId === null) {
      timeoutId = setTimeout(timerExpired, wait);
    }
    return result;
  }

  debounced.cancel = function() {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
    }
    lastInvokeTime = 0;
    lastArgs = lastCallTime = lastThis = timeoutId = null;
  };

  debounced.flush = function() {
    if (timeoutId !== null) {
      return trailingEdge(Date.now());
    }
    return result;
  };

  return debounced as T & { cancel: () => void; flush: () => void };
}

/**
 * 📊 节流（Throttle）
 *
 * 固定时间间隔内只执行一次
 */

interface ThrottleOptions {
  leading?: boolean;
  trailing?: boolean;
}

function throttle<T extends (...args: any[]) => any>(
  fn: T,
  wait: number,
  options: ThrottleOptions = {}
): T & { cancel: () => void } {
  const { leading = true, trailing = true } = options;
  
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let lastTime = 0;
  let lastArgs: Parameters<T> | null = null;
  let lastThis: any = null;

  function invokeFunc() {
    const args = lastArgs!;
    const thisArg = lastThis;
    lastArgs = lastThis = null;
    lastTime = Date.now();
    fn.apply(thisArg, args);
  }

  function throttled(this: any, ...args: Parameters<T>): void {
    const now = Date.now();
    
    if (!lastTime && !leading) {
      lastTime = now;
    }

    const remaining = wait - (now - lastTime);
    lastArgs = args;
    lastThis = this;

    if (remaining <= 0 || remaining > wait) {
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      invokeFunc();
    } else if (!timeoutId && trailing) {
      timeoutId = setTimeout(() => {
        lastTime = leading ? Date.now() : 0;
        timeoutId = null;
        invokeFunc();
      }, remaining);
    }
  }

  throttled.cancel = function() {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    lastTime = 0;
    timeoutId = lastArgs = lastThis = null;
  };

  return throttled as T & { cancel: () => void };
}

// ============================================================
// 3. new / instanceof
// ============================================================

/**
 * 📊 new 操作符实现
 *
 * new 做了什么：
 * 1. 创建新对象
 * 2. 链接原型
 * 3. 绑定 this 并执行构造函数
 * 4. 返回对象
 */

function myNew<T>(
  constructor: new (...args: any[]) => T,
  ...args: any[]
): T {
  // 1. 创建新对象，链接原型
  const obj = Object.create(constructor.prototype);
  
  // 2. 执行构造函数，绑定 this
  const result = constructor.apply(obj, args);
  
  // 3. 如果构造函数返回对象，则返回该对象
  return result instanceof Object ? result : obj;
}

/**
 * 📊 instanceof 实现
 *
 * 检查构造函数的 prototype 是否在对象的原型链上
 */

function myInstanceof(obj: any, constructor: Function): boolean {
  if (obj === null || typeof obj !== 'object') {
    return false;
  }
  
  let proto = Object.getPrototypeOf(obj);
  const prototype = constructor.prototype;
  
  while (proto !== null) {
    if (proto === prototype) {
      return true;
    }
    proto = Object.getPrototypeOf(proto);
  }
  
  return false;
}

// ============================================================
// 4. 柯里化 / 组合函数
// ============================================================

/**
 * 📊 柯里化（Curry）
 *
 * 将多参数函数转为一系列单参数函数
 */

function curry<T extends (...args: any[]) => any>(fn: T): any {
  return function curried(...args: any[]): any {
    if (args.length >= fn.length) {
      return fn.apply(this, args);
    }
    return function(...newArgs: any[]) {
      return curried.apply(this, [...args, ...newArgs]);
    };
  };
}

// 使用示例
const curriedAdd = curry((a: number, b: number, c: number) => a + b + c);
// curriedAdd(1)(2)(3) === 6
// curriedAdd(1, 2)(3) === 6
// curriedAdd(1)(2, 3) === 6

/**
 * 📊 组合函数（Compose）
 *
 * 从右到左执行函数
 */

function compose<T>(...fns: ((arg: T) => T)[]): (arg: T) => T {
  if (fns.length === 0) {
    return (arg: T) => arg;
  }
  if (fns.length === 1) {
    return fns[0];
  }
  return fns.reduce((a, b) => (arg: T) => a(b(arg)));
}

/**
 * 📊 管道函数（Pipe）
 *
 * 从左到右执行函数
 */

function pipe<T>(...fns: ((arg: T) => T)[]): (arg: T) => T {
  if (fns.length === 0) {
    return (arg: T) => arg;
  }
  if (fns.length === 1) {
    return fns[0];
  }
  return fns.reduce((a, b) => (arg: T) => b(a(arg)));
}

// ============================================================
// 5. 数组方法实现
// ============================================================

/**
 * 📊 Array.prototype.flat
 */

function myFlat<T>(arr: T[], depth: number = 1): T[] {
  if (depth <= 0) {
    return arr.slice();
  }
  
  return arr.reduce((acc: T[], item) => {
    if (Array.isArray(item)) {
      return [...acc, ...myFlat(item, depth - 1)];
    }
    return [...acc, item];
  }, []);
}

/**
 * 📊 Array.prototype.reduce
 */

function myReduce<T, U>(
  arr: T[],
  callback: (acc: U, item: T, index: number, array: T[]) => U,
  initialValue?: U
): U {
  let acc: U;
  let startIndex: number;
  
  if (initialValue !== undefined) {
    acc = initialValue;
    startIndex = 0;
  } else {
    if (arr.length === 0) {
      throw new TypeError('Reduce of empty array with no initial value');
    }
    acc = arr[0] as unknown as U;
    startIndex = 1;
  }
  
  for (let i = startIndex; i < arr.length; i++) {
    acc = callback(acc, arr[i], i, arr);
  }
  
  return acc;
}

/**
 * 📊 Array.prototype.map
 */

function myMap<T, U>(
  arr: T[],
  callback: (item: T, index: number, array: T[]) => U
): U[] {
  const result: U[] = [];
  for (let i = 0; i < arr.length; i++) {
    result.push(callback(arr[i], i, arr));
  }
  return result;
}

/**
 * 📊 Array.prototype.filter
 */

function myFilter<T>(
  arr: T[],
  callback: (item: T, index: number, array: T[]) => boolean
): T[] {
  const result: T[] = [];
  for (let i = 0; i < arr.length; i++) {
    if (callback(arr[i], i, arr)) {
      result.push(arr[i]);
    }
  }
  return result;
}

// ============================================================
// 6. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. call/apply 的 this 处理
 *    - null/undefined 时指向全局对象
 *
 * 2. bind 返回的函数可以被 new
 *    - 此时 this 应该指向新对象
 *
 * 3. 防抖节流的 this 和参数传递
 *    - 需要正确传递给原函数
 *
 * 4. new 操作符的返回值
 *    - 构造函数返回对象时使用该对象
 *
 * 5. instanceof 处理基本类型
 *    - 基本类型直接返回 false
 */

// ============================================================
// 7. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: call 和 apply 的区别？
 * A: 参数传递方式不同
 *    call：逐个传递
 *    apply：数组传递
 *
 * Q2: bind 返回的函数可以被 new 吗？
 * A: 可以。new 调用时 this 指向新对象，而不是绑定的 context
 *
 * Q3: 防抖和节流如何选择？
 * A:
 *    防抖：只关心最终结果（搜索框）
 *    节流：需要固定频率响应（滚动事件）
 *
 * Q4: 柯里化有什么用？
 * A:
 *    - 参数复用
 *    - 延迟执行
 *    - 函数组合
 */

export {
  myCall,
  myApply,
  myBind,
  debounce,
  throttle,
  myNew,
  myInstanceof,
  curry,
  compose,
  pipe,
  myFlat,
  myReduce,
  myMap,
  myFilter,
};

