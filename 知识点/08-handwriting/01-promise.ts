/**
 * ============================================================
 * 📚 Promise 手写实现
 * ============================================================
 *
 * 面试考察重点：
 * 1. Promise 基础实现
 * 2. Promise.all / race / allSettled / any
 * 3. 并发控制
 * 4. 异步调度
 */

// ============================================================
// 1. Promise 完整实现
// ============================================================

/**
 * 📊 Promise 状态
 *
 * - pending: 等待中
 * - fulfilled: 已成功
 * - rejected: 已失败
 *
 * 状态一旦改变，不可逆转
 */

type PromiseState = 'pending' | 'fulfilled' | 'rejected';
type Resolve<T> = (value: T | PromiseLike<T>) => void;
type Reject = (reason?: any) => void;
type Executor<T> = (resolve: Resolve<T>, reject: Reject) => void;

class MyPromise<T> {
  private state: PromiseState = 'pending';
  private value: T | undefined;
  private reason: any;
  private onFulfilledCallbacks: Function[] = [];
  private onRejectedCallbacks: Function[] = [];

  constructor(executor: Executor<T>) {
    const resolve: Resolve<T> = (value) => {
      // 处理 value 是 Promise 的情况
      if (value instanceof MyPromise) {
        value.then(resolve, reject);
        return;
      }

      if (this.state === 'pending') {
        this.state = 'fulfilled';
        this.value = value as T;
        this.onFulfilledCallbacks.forEach(fn => fn());
      }
    };

    const reject: Reject = (reason) => {
      if (this.state === 'pending') {
        this.state = 'rejected';
        this.reason = reason;
        this.onRejectedCallbacks.forEach(fn => fn());
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
    // 处理可选参数
    const realOnFulfilled = typeof onFulfilled === 'function'
      ? onFulfilled
      : (v: T) => v as unknown as TResult1;
    const realOnRejected = typeof onRejected === 'function'
      ? onRejected
      : (e: any) => { throw e; };

    const promise2 = new MyPromise<TResult1 | TResult2>((resolve, reject) => {
      const handleFulfilled = () => {
        // 异步执行，确保在 promise2 初始化后执行
        queueMicrotask(() => {
          try {
            const x = realOnFulfilled(this.value!);
            this.resolvePromise(promise2, x, resolve, reject);
          } catch (error) {
            reject(error);
          }
        });
      };

      const handleRejected = () => {
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
        handleFulfilled();
      } else if (this.state === 'rejected') {
        handleRejected();
      } else {
        // pending 状态，存储回调
        this.onFulfilledCallbacks.push(handleFulfilled);
        this.onRejectedCallbacks.push(handleRejected);
      }
    });

    return promise2;
  }

  // 处理 then 返回值
  private resolvePromise<R>(
    promise2: MyPromise<R>,
    x: any,
    resolve: Resolve<R>,
    reject: Reject
  ) {
    // 不能返回自身
    if (promise2 === x) {
      return reject(new TypeError('Chaining cycle detected'));
    }

    // 处理 Promise 或 thenable
    if (x instanceof MyPromise) {
      x.then(resolve, reject);
    } else if (x !== null && (typeof x === 'object' || typeof x === 'function')) {
      let called = false;
      try {
        const then = x.then;
        if (typeof then === 'function') {
          then.call(
            x,
            (y: any) => {
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
      value => MyPromise.resolve(onFinally?.()).then(() => value),
      reason => MyPromise.resolve(onFinally?.()).then(() => { throw reason; })
    );
  }

  // ==================== 静态方法 ====================

  static resolve<T>(value?: T | PromiseLike<T>): MyPromise<T> {
    if (value instanceof MyPromise) {
      return value;
    }
    return new MyPromise(resolve => resolve(value as T));
  }

  static reject<T = never>(reason?: any): MyPromise<T> {
    return new MyPromise((_, reject) => reject(reason));
  }

  static all<T>(promises: Iterable<T | PromiseLike<T>>): MyPromise<Awaited<T>[]> {
    return new MyPromise((resolve, reject) => {
      const arr = Array.from(promises);
      if (arr.length === 0) {
        resolve([]);
        return;
      }

      const results: Awaited<T>[] = new Array(arr.length);
      let count = 0;

      arr.forEach((promise, index) => {
        MyPromise.resolve(promise).then(
          value => {
            results[index] = value as Awaited<T>;
            count++;
            if (count === arr.length) {
              resolve(results);
            }
          },
          reject // 任一失败则整体失败
        );
      });
    });
  }

  static race<T>(promises: Iterable<T | PromiseLike<T>>): MyPromise<Awaited<T>> {
    return new MyPromise((resolve, reject) => {
      const arr = Array.from(promises);
      arr.forEach(promise => {
        MyPromise.resolve(promise).then(resolve, reject);
      });
    });
  }

  static allSettled<T>(
    promises: Iterable<T | PromiseLike<T>>
  ): MyPromise<PromiseSettledResult<Awaited<T>>[]> {
    return new MyPromise((resolve) => {
      const arr = Array.from(promises);
      if (arr.length === 0) {
        resolve([]);
        return;
      }

      const results: PromiseSettledResult<Awaited<T>>[] = new Array(arr.length);
      let count = 0;

      arr.forEach((promise, index) => {
        MyPromise.resolve(promise).then(
          value => {
            results[index] = { status: 'fulfilled', value: value as Awaited<T> };
            count++;
            if (count === arr.length) resolve(results);
          },
          reason => {
            results[index] = { status: 'rejected', reason };
            count++;
            if (count === arr.length) resolve(results);
          }
        );
      });
    });
  }

  static any<T>(promises: Iterable<T | PromiseLike<T>>): MyPromise<Awaited<T>> {
    return new MyPromise((resolve, reject) => {
      const arr = Array.from(promises);
      if (arr.length === 0) {
        reject(new AggregateError([], 'All promises were rejected'));
        return;
      }

      const errors: any[] = new Array(arr.length);
      let count = 0;

      arr.forEach((promise, index) => {
        MyPromise.resolve(promise).then(
          resolve, // 任一成功则整体成功
          reason => {
            errors[index] = reason;
            count++;
            if (count === arr.length) {
              reject(new AggregateError(errors, 'All promises were rejected'));
            }
          }
        );
      });
    });
  }
}

// ============================================================
// 2. 并发控制
// ============================================================

/**
 * 📊 限制并发数的 Promise
 *
 * 场景：批量请求但不想一次性发送太多
 */

class PromisePool {
  private limit: number;
  private running: number = 0;
  private queue: (() => Promise<any>)[] = [];

  constructor(limit: number) {
    this.limit = limit;
  }

  add<T>(task: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      const wrappedTask = () => {
        return task().then(resolve, reject);
      };

      this.queue.push(wrappedTask);
      this.run();
    });
  }

  private run() {
    while (this.running < this.limit && this.queue.length > 0) {
      const task = this.queue.shift()!;
      this.running++;

      task().finally(() => {
        this.running--;
        this.run();
      });
    }
  }
}

// 使用示例
async function poolExample() {
  const pool = new PromisePool(3); // 最多同时 3 个请求

  const urls = ['/api/1', '/api/2', '/api/3', '/api/4', '/api/5'];
  const results = await Promise.all(
    urls.map(url => pool.add(() => fetch(url)))
  );

  return results;
}

/**
 * 📊 另一种实现：asyncPool
 */

async function asyncPool<T, R>(
  limit: number,
  items: T[],
  iteratorFn: (item: T) => Promise<R>
): Promise<R[]> {
  const results: R[] = [];
  const executing: Promise<void>[] = [];

  for (const item of items) {
    const p = Promise.resolve().then(() => iteratorFn(item));
    results.push(p as any);

    if (items.length >= limit) {
      const e: Promise<void> = p.then(() => {
        executing.splice(executing.indexOf(e), 1);
      });
      executing.push(e);

      if (executing.length >= limit) {
        await Promise.race(executing);
      }
    }
  }

  return Promise.all(results);
}

// ============================================================
// 3. 重试机制
// ============================================================

/**
 * 📊 Promise 重试
 */

function retryPromise<T>(
  fn: () => Promise<T>,
  retries: number = 3,
  delay: number = 1000
): Promise<T> {
  return new Promise((resolve, reject) => {
    const attempt = (remaining: number) => {
      fn()
        .then(resolve)
        .catch(error => {
          if (remaining <= 0) {
            reject(error);
          } else {
            console.log(`Retry... (${retries - remaining + 1}/${retries})`);
            setTimeout(() => attempt(remaining - 1), delay);
          }
        });
    };

    attempt(retries);
  });
}

// 带指数退避的重试
function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: {
    retries?: number;
    initialDelay?: number;
    maxDelay?: number;
    factor?: number;
  } = {}
): Promise<T> {
  const {
    retries = 3,
    initialDelay = 1000,
    maxDelay = 30000,
    factor = 2,
  } = options;

  return new Promise((resolve, reject) => {
    const attempt = (remaining: number, delay: number) => {
      fn()
        .then(resolve)
        .catch(error => {
          if (remaining <= 0) {
            reject(error);
          } else {
            const nextDelay = Math.min(delay * factor, maxDelay);
            setTimeout(() => attempt(remaining - 1, nextDelay), delay);
          }
        });
    };

    attempt(retries, initialDelay);
  });
}

// ============================================================
// 4. 超时控制
// ============================================================

/**
 * 📊 Promise 超时
 */

function promiseWithTimeout<T>(
  promise: Promise<T>,
  timeout: number
): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) => {
      setTimeout(() => reject(new Error('Timeout')), timeout);
    }),
  ]);
}

// 可取消的 Promise
function cancellablePromise<T>(
  promise: Promise<T>
): { promise: Promise<T>; cancel: () => void } {
  let isCancelled = false;

  const wrappedPromise = new Promise<T>((resolve, reject) => {
    promise.then(
      value => {
        if (!isCancelled) resolve(value);
      },
      error => {
        if (!isCancelled) reject(error);
      }
    );
  });

  return {
    promise: wrappedPromise,
    cancel: () => {
      isCancelled = true;
    },
  };
}

// ============================================================
// 5. Promisify
// ============================================================

/**
 * 📊 将回调函数转为 Promise
 */

function promisify<T>(
  fn: (...args: [...any[], (err: any, result: T) => void]) => void
): (...args: any[]) => Promise<T> {
  return function (...args: any[]) {
    return new Promise((resolve, reject) => {
      fn(...args, (err: any, result: T) => {
        if (err) {
          reject(err);
        } else {
          resolve(result);
        }
      });
    });
  };
}

// 使用示例
const promisifyExample = `
const fs = require('fs');
const readFile = promisify(fs.readFile);

// 使用
const content = await readFile('file.txt', 'utf-8');
`;

// ============================================================
// 6. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. then 必须返回新的 Promise
 *    - 保证链式调用
 *
 * 2. 回调必须异步执行
 *    - 使用 queueMicrotask 或 setTimeout
 *
 * 3. resolvePromise 处理 thenable
 *    - 兼容各种 Promise 实现
 *
 * 4. 防止重复调用
 *    - called 标志防止多次 resolve/reject
 *
 * 5. all vs allSettled
 *    - all 一个失败全部失败
 *    - allSettled 等待全部完成
 */

// ============================================================
// 7. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: Promise.all 和 Promise.allSettled 的区别？
 * A:
 *    all：一个失败则整体失败
 *    allSettled：等待所有完成，返回每个结果
 *
 * Q2: 如何实现 Promise 的并发限制？
 * A:
 *    - 维护一个执行中的队列
 *    - 达到限制时等待
 *    - 完成一个执行下一个
 *
 * Q3: Promise 的回调为什么是异步的？
 * A:
 *    - 保证一致性（同步/异步 Promise 行为一致）
 *    - 避免 Zalgo（不可预测的执行顺序）
 *
 * Q4: async/await 是如何实现的？
 * A:
 *    - 基于 Generator + 自动执行器
 *    - 将 yield 改为 await
 *    - 返回值包装为 Promise
 */

export {
  MyPromise,
  PromisePool,
  asyncPool,
  retryPromise,
  retryWithBackoff,
  promiseWithTimeout,
  cancellablePromise,
  promisify,
  poolExample,
  promisifyExample,
};

