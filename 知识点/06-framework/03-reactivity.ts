/**
 * ============================================================
 * 📚 响应式原理
 * ============================================================
 *
 * 面试考察重点：
 * 1. Vue2 响应式（Object.defineProperty）
 * 2. Vue3 响应式（Proxy）
 * 3. 依赖收集与派发更新
 * 4. React 的"响应式"
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 什么是响应式？
 *
 * 响应式 = 数据变化 → 自动更新视图
 *
 * 核心问题：
 * 1. 如何知道数据变化了？（数据劫持）
 * 2. 数据变化后通知谁？（依赖收集）
 * 3. 如何高效更新？（异步批量更新）
 *
 * 📊 Vue vs React
 *
 * Vue：
 * - 真正的响应式
 * - 自动追踪依赖
 * - 精确更新
 *
 * React：
 * - 不是响应式，是"调度式"
 * - 手动 setState 触发更新
 * - 从根组件开始 Diff
 */

// ============================================================
// 2. Vue2 响应式（Object.defineProperty）
// ============================================================

/**
 * 📊 Vue2 响应式原理
 *
 * 1. Observer：递归遍历对象，使用 defineProperty 劫持属性
 * 2. Dep：依赖收集器，存储订阅者（Watcher）
 * 3. Watcher：订阅者，数据变化时执行更新
 *
 * 流程：
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                                                                 │
 * │   Data ──► Observer ──► defineProperty                         │
 * │                              │                                  │
 * │                              ├── get: 依赖收集（Dep.depend）    │
 * │                              │       ↓                          │
 * │                              │    Watcher 加入 Dep              │
 * │                              │                                  │
 * │                              └── set: 派发更新（Dep.notify）    │
 * │                                      ↓                          │
 * │                                   Watcher.update                │
 * │                                      ↓                          │
 * │                                   重新渲染                      │
 * └─────────────────────────────────────────────────────────────────┘
 */

// Vue2 风格的响应式实现
class Dep {
  static target: Watcher | null = null;
  private subs: Set<Watcher> = new Set();

  depend() {
    if (Dep.target) {
      this.subs.add(Dep.target);
    }
  }

  notify() {
    this.subs.forEach(watcher => watcher.update());
  }
}

class Watcher {
  private getter: () => any;
  private callback: (value: any) => void;
  private value: any;

  constructor(getter: () => any, callback: (value: any) => void) {
    this.getter = getter;
    this.callback = callback;
    this.value = this.get();
  }

  get() {
    Dep.target = this;
    const value = this.getter();
    Dep.target = null;
    return value;
  }

  update() {
    const newValue = this.getter();
    if (newValue !== this.value) {
      this.value = newValue;
      this.callback(newValue);
    }
  }
}

function defineReactive(obj: any, key: string, val: any) {
  const dep = new Dep();
  
  // 递归处理嵌套对象
  if (typeof val === 'object' && val !== null) {
    observe(val);
  }

  Object.defineProperty(obj, key, {
    enumerable: true,
    configurable: true,
    get() {
      dep.depend(); // 依赖收集
      return val;
    },
    set(newVal) {
      if (newVal === val) return;
      val = newVal;
      // 新值可能是对象，需要递归处理
      if (typeof newVal === 'object' && newVal !== null) {
        observe(newVal);
      }
      dep.notify(); // 派发更新
    },
  });
}

function observe(obj: any) {
  if (typeof obj !== 'object' || obj === null) return;
  
  Object.keys(obj).forEach(key => {
    defineReactive(obj, key, obj[key]);
  });
}

/**
 * ⚠️ Vue2 响应式的局限性
 *
 * 1. 无法检测属性的添加/删除
 *    - 需要使用 Vue.set / Vue.delete
 *
 * 2. 无法检测数组索引修改
 *    - arr[0] = 'new' 不会触发更新
 *    - 需要使用 Vue.set 或数组方法
 *
 * 3. 无法检测数组长度修改
 *    - arr.length = 0 不会触发更新
 *
 * 4. 初始化时需要递归遍历
 *    - 大对象初始化性能差
 */

// ============================================================
// 3. Vue3 响应式（Proxy）
// ============================================================

/**
 * 📊 Vue3 响应式原理
 *
 * 使用 Proxy 代替 defineProperty：
 * - 可以劫持整个对象
 * - 可以检测属性添加/删除
 * - 可以检测数组索引和长度变化
 * - 惰性处理（访问时才代理）
 */

// 存储依赖关系
const targetMap = new WeakMap<object, Map<string | symbol, Set<Function>>>();

// 当前正在执行的 effect
let activeEffect: Function | null = null;

// 依赖收集
function track(target: object, key: string | symbol) {
  if (!activeEffect) return;
  
  let depsMap = targetMap.get(target);
  if (!depsMap) {
    depsMap = new Map();
    targetMap.set(target, depsMap);
  }
  
  let deps = depsMap.get(key);
  if (!deps) {
    deps = new Set();
    depsMap.set(key, deps);
  }
  
  deps.add(activeEffect);
}

// 派发更新
function trigger(target: object, key: string | symbol) {
  const depsMap = targetMap.get(target);
  if (!depsMap) return;
  
  const deps = depsMap.get(key);
  if (deps) {
    deps.forEach(effect => effect());
  }
}

// reactive：创建响应式对象
function reactive<T extends object>(target: T): T {
  return new Proxy(target, {
    get(target, key, receiver) {
      const result = Reflect.get(target, key, receiver);
      track(target, key); // 依赖收集
      
      // 深层响应式
      if (typeof result === 'object' && result !== null) {
        return reactive(result);
      }
      return result;
    },
    
    set(target, key, value, receiver) {
      const oldValue = Reflect.get(target, key, receiver);
      const result = Reflect.set(target, key, value, receiver);
      
      if (oldValue !== value) {
        trigger(target, key); // 派发更新
      }
      return result;
    },
    
    deleteProperty(target, key) {
      const hadKey = Reflect.has(target, key);
      const result = Reflect.deleteProperty(target, key);
      
      if (hadKey && result) {
        trigger(target, key);
      }
      return result;
    },
  });
}

// ref：创建响应式基本类型
function ref<T>(value: T) {
  return {
    get value() {
      track(this, 'value');
      return value;
    },
    set value(newValue: T) {
      if (newValue !== value) {
        value = newValue;
        trigger(this, 'value');
      }
    },
  };
}

// effect：副作用函数
function effect(fn: Function) {
  const effectFn = () => {
    activeEffect = effectFn;
    fn();
    activeEffect = null;
  };
  effectFn();
  return effectFn;
}

// computed：计算属性
function computed<T>(getter: () => T) {
  let cached: T;
  let dirty = true;
  
  const effectFn = effect(() => {
    cached = getter();
    dirty = false;
  });
  
  return {
    get value() {
      if (dirty) {
        effectFn();
      }
      return cached;
    },
  };
}

// ============================================================
// 4. Vue3 响应式进阶
// ============================================================

/**
 * 📊 shallowReactive vs reactive
 *
 * reactive：深层响应式
 * shallowReactive：只有根属性是响应式的
 *
 * 使用场景：
 * - 大对象但只关心顶层变化
 * - 性能敏感场景
 */

function shallowReactive<T extends object>(target: T): T {
  return new Proxy(target, {
    get(target, key, receiver) {
      track(target, key);
      return Reflect.get(target, key, receiver);
    },
    set(target, key, value, receiver) {
      const oldValue = Reflect.get(target, key, receiver);
      const result = Reflect.set(target, key, value, receiver);
      if (oldValue !== value) {
        trigger(target, key);
      }
      return result;
    },
  });
}

/**
 * 📊 readonly vs reactive
 *
 * readonly：只读响应式，不能修改
 * 
 * 使用场景：
 * - props（组件接收的属性）
 * - 防止意外修改
 */

function readonly<T extends object>(target: T): Readonly<T> {
  return new Proxy(target, {
    get(target, key, receiver) {
      const result = Reflect.get(target, key, receiver);
      if (typeof result === 'object' && result !== null) {
        return readonly(result);
      }
      return result;
    },
    set() {
      console.warn('Cannot set on a readonly object');
      return true;
    },
    deleteProperty() {
      console.warn('Cannot delete on a readonly object');
      return true;
    },
  });
}

// ============================================================
// 5. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. 解构丢失响应式
 *    const { count } = reactive({ count: 0 });
 *    // count 是普通值，不是响应式
 *    // 解决：使用 toRefs
 *
 * 2. 直接替换响应式对象
 *    let state = reactive({ count: 0 });
 *    state = reactive({ count: 1 }); // 丢失响应式
 *    // 解决：修改属性而不是替换对象
 *
 * 3. ref 忘记 .value
 *    const count = ref(0);
 *    count = 1; // ❌ 错误
 *    count.value = 1; // ✅ 正确
 *
 * 4. 在模板中 ref 自动解包的误解
 *    - 模板中不需要 .value
 *    - JS 中需要 .value
 *
 * 5. 响应式对象作为 Map/Set 的 key
 *    - 代理对象和原对象不是同一个引用
 *    - 可能导致查找失败
 */

const reactivityPitfalls = `
// ❌ 解构丢失响应式
const state = reactive({ count: 0 });
const { count } = state; // count 不是响应式的

// ✅ 使用 toRefs
const { count } = toRefs(state); // count 是 ref

// ❌ 替换整个响应式对象
let state = reactive({ count: 0 });
state = reactive({ count: 1 }); // 新对象，模板不会更新

// ✅ 修改属性
state.count = 1;

// 或者包一层
const state = reactive({ data: { count: 0 } });
state.data = { count: 1 }; // ✅ 这样可以
`;

// ============================================================
// 6. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: Vue2 和 Vue3 响应式有什么区别？
 * A:
 *    Vue2（defineProperty）：
 *    - 需要递归遍历所有属性
 *    - 无法检测属性添加/删除
 *    - 数组需要特殊处理
 *
 *    Vue3（Proxy）：
 *    - 惰性代理，访问时才处理
 *    - 可以检测属性添加/删除
 *    - 可以监听数组变化
 *
 * Q2: 为什么 Vue3 选择 Proxy？
 * A:
 *    - 功能更强大
 *    - 性能更好（惰性处理）
 *    - 代码更简洁
 *    - 缺点：不支持 IE
 *
 * Q3: computed 和 watch 的区别？
 * A:
 *    computed：
 *    - 有返回值
 *    - 自动缓存
 *    - 同步执行
 *
 *    watch：
 *    - 无返回值（执行副作用）
 *    - 不缓存
 *    - 可以是异步
 *
 * Q4: Vue3 的 ref 和 reactive 怎么选？
 * A:
 *    ref：
 *    - 基本类型
 *    - 需要整个替换的对象
 *    - 需要 .value 访问
 *
 *    reactive：
 *    - 对象/数组
 *    - 不能整个替换
 *    - 直接访问属性
 *
 * Q5: React 有响应式吗？
 * A:
 *    - React 不是响应式，是调度式
 *    - 需要手动 setState 触发更新
 *    - 从根组件开始 Diff
 *    - 通过 memo/shouldComponentUpdate 优化
 */

// ============================================================
// 7. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：表单数据绑定
 *
 * Vue：
 * - v-model 双向绑定
 * - 响应式自动更新
 *
 * React：
 * - 受控组件
 * - 手动 onChange + setState
 */

/**
 * 🏢 场景 2：全局状态管理
 *
 * Vue3 简单方案：
 * - reactive + provide/inject
 * - 不需要额外库
 *
 * Pinia：
 * - 基于 Vue3 响应式
 * - 支持 devtools
 */

const simpleStoreExample = `
// Vue3 简单全局状态
// store.ts
import { reactive, readonly } from 'vue';

const state = reactive({
  count: 0,
  user: null,
});

export const store = {
  state: readonly(state),
  
  increment() {
    state.count++;
  },
  
  setUser(user) {
    state.user = user;
  },
};

// main.ts
app.provide('store', store);

// Component.vue
const store = inject('store');
`;

/**
 * 🏢 场景 3：性能优化
 *
 * 问题：大对象响应式初始化慢
 *
 * 解决：
 * - shallowReactive：只监听顶层
 * - markRaw：标记不需要响应式的数据
 */

export {
  // Vue2 风格
  Dep,
  Watcher,
  defineReactive,
  observe,
  
  // Vue3 风格
  reactive,
  ref,
  effect,
  computed,
  track,
  trigger,
  shallowReactive,
  readonly,
  
  // 示例
  reactivityPitfalls,
  simpleStoreExample,
};

