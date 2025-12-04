/**
 * ============================================================
 * 📚 Promise 与异步编程
 * ============================================================
 *
 * 面试考察重点：
 * 1. Promise 的状态和基本用法
 * 2. Promise 的链式调用
 * 3. Promise 的静态方法
 * 4. 手写 Promise
 * 5. async/await 的原理
 */

// ============================================================
// 1. Promise 基础
// ============================================================

/**
 * 📖 什么是 Promise？
 *
 * Promise 是异步编程的一种解决方案，用于解决回调地狱问题。
 *
 * 三种状态：
 * - pending（进行中）
 * - fulfilled（已成功）
 * - rejected（已失败）
 *
 * 特点：
 * - 状态只能从 pending 变为 fulfilled 或 rejected
 * - 状态一旦改变，就不会再变
 * - Promise 会立即执行，then/catch 是异步的（微任务）
 */

// 基本用法
const promise = new Promise((resolve, reject) => {
  // 执行异步操作
  setTimeout(() => {
    const success = true;
    if (success) {
      resolve('成功');
    } else {
      reject(new Error('失败'));
    }
  }, 1000);
});

promise
  .then((value) => {
    console.log(value); // '成功'
  })
  .catch((error) => {
    console.error(error);
  });

// ============================================================
// 2. Promise 链式调用
// ============================================================

/**
 * 📖 then 方法返回新的 Promise
 *
 * - return 普通值：新 Promise 状态为 fulfilled，值为返回值
 * - return Promise：新 Promise 状态和值跟随返回的 Promise
 * - throw Error：新 Promise 状态为 rejected，值为错误
 */

Promise.resolve(1)
  .then((value) => {
    console.log(value); // 1
    return value + 1;
  })
  .then((value) => {
    console.log(value); // 2
    return Promise.resolve(value + 1);
  })
  .then((value) => {
    console.log(value); // 3
    throw new Error('出错了');
  })
  .catch((error) => {
    console.error(error.message); // '出错了'
    return '恢复';
  })
  .then((value) => {
    console.log(value); // '恢复'
  });

// ============================================================
// 3. Promise 静态方法
// ============================================================

// 3.1 Promise.resolve / Promise.reject
const p1 = Promise.resolve('成功');
const p2 = Promise.reject(new Error('失败'));

// 3.2 Promise.all - 全部成功才成功
const all = Promise.all([Promise.resolve(1), Promise.resolve(2), Promise.resolve(3)]);
// all → [1, 2, 3]

// 3.3 Promise.race - 第一个完成的结果（无论成功失败）
const race = Promise.race([
  new Promise((resolve) => setTimeout(() => resolve('slow'), 1000)),
  new Promise((resolve) => setTimeout(() => resolve('fast'), 500)),
]);
// race → 'fast'

// 3.4 Promise.allSettled - 全部完成，返回所有结果
const allSettled = Promise.allSettled([Promise.resolve(1), Promise.reject('error'), Promise.resolve(3)]);
// allSettled → [
//   { status: 'fulfilled', value: 1 },
//   { status: 'rejected', reason: 'error' },
//   { status: 'fulfilled', value: 3 }
// ]

// 3.5 Promise.any - 第一个成功的结果
const any = Promise.any([Promise.reject('error1'), Promise.resolve('success'), Promise.reject('error2')]);
// any → 'success'

// ============================================================
// 4. 手写 Promise（符合 Promise/A+ 规范）
// ============================================================

type Resolve<T> = (value: T | PromiseLike<T>) => void;
type Reject = (reason?: any) => void;
type Executor<T> = (resolve: Resolve<T>, reject: Reject) => void;

class MyPromise<T> {
  private state: 'pending' | 'fulfilled' | 'rejected' = 'pending';
  private value: T | undefined = undefined;
  private reason: any = undefined;
  private onFulfilledCallbacks: Array<() => void> = [];
  private onRejectedCallbacks: Array<() => void> = [];

  constructor(executor: Executor<T>) {
    const resolve: Resolve<T> = (value) => {
      // 处理 Promise 类型的 value
      if (value instanceof MyPromise) {
        value.then(resolve, reject);
        return;
      }

      if (this.state === 'pending') {
        this.state = 'fulfilled';
        this.value = value as T;
        this.onFulfilledCallbacks.forEach((fn) => fn());
      }
    };

    const reject: Reject = (reason) => {
      if (this.state === 'pending') {
        this.state = 'rejected';
        this.reason = reason;
        this.onRejectedCallbacks.forEach((fn) => fn());
      }
    };

    try {
      executor(resolve, reject);
    } catch (error) {
      reject(error);
    }
  }

  then<TResult1 = T, TResult2 = never>(
    onFulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | null,
    onRejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | null
  ): MyPromise<TResult1 | TResult2> {
    // 处理默认值（值穿透）
    const realOnFulfilled = typeof onFulfilled === 'function' ? onFulfilled : (value: T) => value as unknown as TResult1;
    const realOnRejected =
      typeof onRejected === 'function'
        ? onRejected
        : (reason: any) => {
            throw reason;
          };

    const promise2 = new MyPromise<TResult1 | TResult2>((resolve, reject) => {
      const fulfilledMicrotask = () => {
        queueMicrotask(() => {
          try {
            const x = realOnFulfilled(this.value as T);
            this.resolvePromise(promise2, x, resolve, reject);
          } catch (error) {
            reject(error);
          }
        });
      };

      const rejectedMicrotask = () => {
        queueMicrotask(() => {
          try {
            const x = realOnRejected(this.reason);
            this.resolvePromise(promise2, x, resolve, reject);
          } catch (error) {
            reject(error);
          }
        });
      };

      if (this.state === 'fulfilled') {
        fulfilledMicrotask();
      } else if (this.state === 'rejected') {
        rejectedMicrotask();
      } else {
        this.onFulfilledCallbacks.push(fulfilledMicrotask);
        this.onRejectedCallbacks.push(rejectedMicrotask);
      }
    });

    return promise2;
  }

  private resolvePromise<U>(
    promise2: MyPromise<U>,
    x: U | PromiseLike<U>,
    resolve: Resolve<U>,
    reject: Reject
  ): void {
    // 防止循环引用
    if (promise2 === x) {
      reject(new TypeError('Chaining cycle detected'));
      return;
    }

    if (x instanceof MyPromise) {
      x.then(resolve, reject);
      return;
    }

    // 处理 thenable
    if (x !== null && (typeof x === 'object' || typeof x === 'function')) {
      let called = false;
      try {
        const then = (x as any).then;
        if (typeof then === 'function') {
          then.call(
            x,
            (y: U) => {
              if (called) return;
              called = true;
              this.resolvePromise(promise2, y, resolve, reject);
            },
            (r: any) => {
              if (called) return;
              called = true;
              reject(r);
            }
          );
        } else {
          resolve(x);
        }
      } catch (error) {
        if (called) return;
        reject(error);
      }
    } else {
      resolve(x);
    }
  }

  catch<TResult = never>(
    onRejected?: ((reason: any) => TResult | PromiseLike<TResult>) | null
  ): MyPromise<T | TResult> {
    return this.then(null, onRejected);
  }

  finally(onFinally?: (() => void) | null): MyPromise<T> {
    return this.then(
      (value) => MyPromise.resolve(onFinally?.()).then(() => value),
      (reason) =>
        MyPromise.resolve(onFinally?.()).then(() => {
          throw reason;
        })
    );
  }

  // 静态方法
  static resolve<U>(value?: U | PromiseLike<U>): MyPromise<U> {
    if (value instanceof MyPromise) {
      return value;
    }
    return new MyPromise((resolve) => resolve(value as U));
  }

  static reject<U = never>(reason?: any): MyPromise<U> {
    return new MyPromise((_, reject) => reject(reason));
  }

  static all<T extends readonly unknown[]>(
    promises: T
  ): MyPromise<{ -readonly [K in keyof T]: Awaited<T[K]> }> {
    return new MyPromise((resolve, reject) => {
      const result: any[] = [];
      let count = 0;
      const len = promises.length;

      if (len === 0) {
        resolve(result as any);
        return;
      }

      promises.forEach((p, index) => {
        MyPromise.resolve(p).then(
          (value) => {
            result[index] = value;
            count++;
            if (count === len) {
              resolve(result as any);
            }
          },
          (reason) => {
            reject(reason);
          }
        );
      });
    });
  }

  static race<T>(promises: Iterable<T | PromiseLike<T>>): MyPromise<Awaited<T>> {
    return new MyPromise((resolve, reject) => {
      for (const p of promises) {
        MyPromise.resolve(p).then(resolve as any, reject);
      }
    });
  }

  static allSettled<T extends readonly unknown[]>(
    promises: T
  ): MyPromise<{ -readonly [K in keyof T]: PromiseSettledResult<Awaited<T[K]>> }> {
    return new MyPromise((resolve) => {
      const result: PromiseSettledResult<any>[] = [];
      let count = 0;
      const len = promises.length;

      if (len === 0) {
        resolve(result as any);
        return;
      }

      promises.forEach((p, index) => {
        MyPromise.resolve(p).then(
          (value) => {
            result[index] = { status: 'fulfilled', value };
            count++;
            if (count === len) resolve(result as any);
          },
          (reason) => {
            result[index] = { status: 'rejected', reason };
            count++;
            if (count === len) resolve(result as any);
          }
        );
      });
    });
  }

  static any<T extends readonly unknown[]>(promises: T): MyPromise<Awaited<T[number]>> {
    return new MyPromise((resolve, reject) => {
      const errors: any[] = [];
      let count = 0;
      const len = promises.length;

      if (len === 0) {
        reject(new AggregateError(errors, 'All promises were rejected'));
        return;
      }

      promises.forEach((p, index) => {
        MyPromise.resolve(p).then(
          (value) => {
            resolve(value as any);
          },
          (reason) => {
            errors[index] = reason;
            count++;
            if (count === len) {
              reject(new AggregateError(errors, 'All promises were rejected'));
            }
          }
        );
      });
    });
  }
}

// ============================================================
// 5. async/await
// ============================================================

/**
 * 📖 async/await 是 Generator + Promise 的语法糖
 *
 * async 函数：
 * - 返回 Promise
 * - 内部可以使用 await
 *
 * await：
 * - 暂停 async 函数执行
 * - 等待 Promise 完成
 * - await 后面的代码相当于 then 的回调（微任务）
 */

// async/await 等价于 Generator + Promise
function* generatorFn() {
  const result1 = yield Promise.resolve(1);
  console.log(result1);
  const result2 = yield Promise.resolve(2);
  console.log(result2);
  return 3;
}

// 自动执行 Generator
function asyncToGenerator(generatorFn: () => Generator): () => Promise<any> {
  return function () {
    const gen = generatorFn();

    return new Promise((resolve, reject) => {
      function step(key: 'next' | 'throw', arg?: any) {
        let result;
        try {
          result = gen[key](arg);
        } catch (error) {
          reject(error);
          return;
        }

        if (result.done) {
          resolve(result.value);
        } else {
          Promise.resolve(result.value).then(
            (value) => step('next', value),
            (reason) => step('throw', reason)
          );
        }
      }

      step('next');
    });
  };
}

// 使用
const asyncFn = asyncToGenerator(generatorFn);
asyncFn().then(console.log); // 输出 1, 2, 3

// ============================================================
// 6. 常见 Promise 应用
// ============================================================

// 6.1 超时控制
function promiseWithTimeout<T>(promise: Promise<T>, timeout: number): Promise<T> {
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error('Timeout')), timeout);
  });
  return Promise.race([promise, timeoutPromise]);
}

// 6.2 重试机制
async function retry<T>(fn: () => Promise<T>, times: number, delay: number = 0): Promise<T> {
  let lastError: Error | undefined;

  for (let i = 0; i < times; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      if (i < times - 1 && delay > 0) {
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError;
}

// 6.3 并发控制
async function asyncPool<T, R>(
  limit: number,
  items: T[],
  fn: (item: T) => Promise<R>
): Promise<R[]> {
  const results: R[] = [];
  const executing: Promise<void>[] = [];

  for (const [index, item] of items.entries()) {
    const p = Promise.resolve().then(() => fn(item));

    results[index] = undefined as any;

    const e = p.then((result) => {
      results[index] = result;
      executing.splice(executing.indexOf(e), 1);
    }) as Promise<void>;

    executing.push(e);

    if (executing.length >= limit) {
      await Promise.race(executing);
    }
  }

  await Promise.all(executing);
  return results;
}

// 6.4 串行执行
async function serial<T>(tasks: (() => Promise<T>)[]): Promise<T[]> {
  const results: T[] = [];
  for (const task of tasks) {
    results.push(await task());
  }
  return results;
}

// 或者使用 reduce
function serialReduce<T>(tasks: (() => Promise<T>)[]): Promise<T[]> {
  return tasks.reduce((promise, task) => {
    return promise.then((results) => task().then((result) => [...results, result]));
  }, Promise.resolve([] as T[]));
}

// ============================================================
// 7. 高频面试题
// ============================================================

/**
 * 题目 1：实现 Promise.finally
 */
Promise.prototype.myFinally = function (callback: () => void) {
  return this.then(
    (value) => Promise.resolve(callback()).then(() => value),
    (reason) =>
      Promise.resolve(callback()).then(() => {
        throw reason;
      })
  );
};

/**
 * 题目 2：实现红绿灯（红3秒，绿1秒，黄2秒，循环）
 */
function red() {
  console.log('red');
}
function green() {
  console.log('green');
}
function yellow() {
  console.log('yellow');
}

function light(cb: () => void, timer: number) {
  return new Promise<void>((resolve) => {
    setTimeout(() => {
      cb();
      resolve();
    }, timer);
  });
}

async function trafficLight() {
  while (true) {
    await light(red, 3000);
    await light(green, 1000);
    await light(yellow, 2000);
  }
}

/**
 * 题目 3：实现 Promise 调度器
 * 要求：最多同时执行 N 个任务
 */
class Scheduler {
  private queue: Array<() => Promise<any>> = [];
  private running = 0;
  private maxConcurrent: number;

  constructor(maxConcurrent: number) {
    this.maxConcurrent = maxConcurrent;
  }

  add<T>(promiseCreator: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      const task = () =>
        promiseCreator()
          .then(resolve)
          .catch(reject)
          .finally(() => {
            this.running--;
            this.runNext();
          });

      this.queue.push(task);
      this.runNext();
    });
  }

  private runNext() {
    while (this.running < this.maxConcurrent && this.queue.length > 0) {
      const task = this.queue.shift();
      if (task) {
        this.running++;
        task();
      }
    }
  }
}

// 使用示例
const scheduler = new Scheduler(2);

const addTask = (time: number, order: string) => {
  scheduler
    .add(() => new Promise((resolve) => setTimeout(resolve, time)))
    .then(() => console.log(order));
};

addTask(1000, '1');
addTask(500, '2');
addTask(300, '3');
addTask(400, '4');
// 输出：2, 3, 1, 4

/**
 * 题目 4：使用 Promise 实现每隔 1 秒输出 1, 2, 3
 */
const arr = [1, 2, 3];

// 方法 1：reduce
arr.reduce((promise, num) => {
  return promise.then(() => {
    return new Promise((resolve) => {
      setTimeout(() => {
        console.log(num);
        resolve(undefined);
      }, 1000);
    });
  });
}, Promise.resolve());

// 方法 2：async/await
async function printNumbers() {
  for (const num of arr) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    console.log(num);
  }
}

export {
  MyPromise,
  asyncToGenerator,
  promiseWithTimeout,
  retry,
  asyncPool,
  serial,
  Scheduler,
};

