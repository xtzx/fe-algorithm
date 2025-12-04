/**
 * ============================================================
 * 📚 深拷贝实现
 * ============================================================
 *
 * 面试考察重点：
 * 1. 深拷贝的完整实现
 * 2. 循环引用处理
 * 3. 特殊类型处理
 * 4. 性能优化
 */

// ============================================================
// 1. 基础版本
// ============================================================

/**
 * 📊 最简单的深拷贝
 *
 * 缺点：
 * - 无法处理函数、Symbol、undefined
 * - 无法处理循环引用
 * - 无法处理特殊对象（Date、RegExp、Map、Set）
 */

function simpleDeepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

// ============================================================
// 2. 完整版本
// ============================================================

/**
 * 📊 完整的深拷贝实现
 *
 * 支持：
 * - 基本类型
 * - 数组、对象
 * - Date、RegExp、Map、Set
 * - 函数
 * - Symbol 属性
 * - 循环引用
 */

function deepClone<T>(target: T, map = new WeakMap()): T {
  // 1. 处理基本类型
  if (target === null || typeof target !== 'object') {
    return target;
  }

  // 2. 处理循环引用
  if (map.has(target)) {
    return map.get(target);
  }

  // 3. 处理特殊对象
  const constructor = (target as any).constructor;
  
  // Date
  if (target instanceof Date) {
    return new Date(target.getTime()) as T;
  }
  
  // RegExp
  if (target instanceof RegExp) {
    return new RegExp(target.source, target.flags) as T;
  }
  
  // Map
  if (target instanceof Map) {
    const result = new Map();
    map.set(target, result);
    target.forEach((value, key) => {
      result.set(deepClone(key, map), deepClone(value, map));
    });
    return result as T;
  }
  
  // Set
  if (target instanceof Set) {
    const result = new Set();
    map.set(target, result);
    target.forEach(value => {
      result.add(deepClone(value, map));
    });
    return result as T;
  }
  
  // ArrayBuffer
  if (target instanceof ArrayBuffer) {
    const result = target.slice(0);
    return result as T;
  }
  
  // TypedArray
  if (ArrayBuffer.isView(target)) {
    const result = new (constructor as any)(
      deepClone((target as any).buffer, map),
      (target as any).byteOffset,
      (target as any).length
    );
    return result as T;
  }

  // 4. 处理数组和普通对象
  const result = Array.isArray(target) ? [] : Object.create(Object.getPrototypeOf(target));
  
  // 存入 map，处理循环引用
  map.set(target, result);
  
  // 5. 复制普通属性
  for (const key in target) {
    if (Object.prototype.hasOwnProperty.call(target, key)) {
      result[key] = deepClone((target as any)[key], map);
    }
  }
  
  // 6. 复制 Symbol 属性
  const symbolKeys = Object.getOwnPropertySymbols(target);
  for (const symbolKey of symbolKeys) {
    result[symbolKey] = deepClone((target as any)[symbolKey], map);
  }
  
  return result as T;
}

// ============================================================
// 3. 处理函数
// ============================================================

/**
 * 📊 函数的拷贝
 *
 * 两种方式：
 * 1. 直接引用（通常做法）
 * 2. 真正复制（很少需要）
 */

function cloneFunction(fn: Function): Function {
  // 判断是否是箭头函数
  if (!fn.prototype) {
    return fn; // 箭头函数无法复制，直接返回
  }
  
  // 使用 new Function 复制
  const fnStr = fn.toString();
  const bodyStart = fnStr.indexOf('{') + 1;
  const bodyEnd = fnStr.lastIndexOf('}');
  const body = fnStr.substring(bodyStart, bodyEnd);
  
  const paramStart = fnStr.indexOf('(') + 1;
  const paramEnd = fnStr.indexOf(')');
  const params = fnStr.substring(paramStart, paramEnd);
  
  return new Function(params, body);
}

// ============================================================
// 4. 性能优化版本
// ============================================================

/**
 * 📊 使用循环代替递归
 *
 * 避免栈溢出
 */

function deepCloneIterative<T>(target: T): T {
  if (target === null || typeof target !== 'object') {
    return target;
  }

  const map = new WeakMap();
  const root = Array.isArray(target) ? [] : {};
  
  // 使用栈模拟递归
  const stack: Array<{
    parent: any;
    key: string | symbol | undefined;
    source: any;
  }> = [{ parent: null, key: undefined, source: target }];
  
  map.set(target, root);

  while (stack.length > 0) {
    const { parent, key, source } = stack.pop()!;
    
    let clone: any;
    
    if (map.has(source)) {
      clone = map.get(source);
    } else {
      clone = Array.isArray(source) ? [] : {};
      map.set(source, clone);
      
      // 添加子节点到栈
      const keys = [
        ...Object.keys(source),
        ...Object.getOwnPropertySymbols(source),
      ];
      
      for (const k of keys) {
        const value = source[k];
        if (value !== null && typeof value === 'object') {
          stack.push({ parent: clone, key: k, source: value });
        } else {
          clone[k] = value;
        }
      }
    }
    
    if (parent !== null && key !== undefined) {
      parent[key] = clone;
    }
  }
  
  return root as T;
}

// ============================================================
// 5. 使用 structuredClone（现代 API）
// ============================================================

/**
 * 📊 structuredClone
 *
 * 浏览器原生 API，支持：
 * - 大多数内置类型
 * - 循环引用
 * - ArrayBuffer、TypedArray
 *
 * 不支持：
 * - 函数
 * - DOM 节点
 * - 某些浏览器特定对象
 */

function modernDeepClone<T>(target: T): T {
  try {
    return structuredClone(target);
  } catch {
    // 降级到手动实现
    return deepClone(target);
  }
}

// ============================================================
// 6. 测试用例
// ============================================================

function testDeepClone() {
  // 基本测试
  const obj = {
    str: 'string',
    num: 123,
    bool: true,
    null: null,
    undefined: undefined,
    symbol: Symbol('test'),
    date: new Date(),
    regex: /test/gi,
    arr: [1, 2, [3, 4]],
    map: new Map([['key', 'value']]),
    set: new Set([1, 2, 3]),
    fn: function() { console.log('fn'); },
    arrow: () => console.log('arrow'),
    nested: {
      a: 1,
      b: { c: 2 },
    },
  };

  // 循环引用测试
  const circular: any = { a: 1 };
  circular.self = circular;
  circular.arr = [circular, 1, 2];

  const cloned = deepClone(obj);
  const clonedCircular = deepClone(circular);

  console.log('Original:', obj);
  console.log('Cloned:', cloned);
  console.log('Are they equal?', obj === cloned); // false
  console.log('Nested equal?', obj.nested === cloned.nested); // false
  console.log('Circular works:', clonedCircular.self === clonedCircular); // true
}

// ============================================================
// 7. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. 忘记处理循环引用
 *    - 导致无限递归
 *    - 使用 WeakMap 记录已克隆对象
 *
 * 2. 忘记处理特殊类型
 *    - Date、RegExp、Map、Set 等
 *    - 需要特殊构造
 *
 * 3. 忘记处理 Symbol 属性
 *    - for...in 不会遍历 Symbol
 *    - 需要 Object.getOwnPropertySymbols
 *
 * 4. 忘记处理原型链
 *    - 使用 Object.create(Object.getPrototypeOf(target))
 *
 * 5. JSON.stringify 的问题
 *    - 函数、Symbol、undefined 会丢失
 *    - 循环引用会报错
 */

// ============================================================
// 8. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: JSON.parse(JSON.stringify()) 有什么问题？
 * A:
 *    - 无法复制函数、Symbol、undefined
 *    - 循环引用会报错
 *    - Date 变成字符串
 *    - RegExp 变成空对象
 *    - NaN、Infinity 变成 null
 *
 * Q2: 为什么用 WeakMap 而不是 Map？
 * A:
 *    - WeakMap 的键是弱引用
 *    - 不会阻止垃圾回收
 *    - 克隆完成后自动释放内存
 *
 * Q3: 如何处理循环引用？
 * A:
 *    - 用 WeakMap 记录已克隆的对象
 *    - 遇到已克隆的直接返回引用
 *
 * Q4: structuredClone 和手写有什么区别？
 * A:
 *    structuredClone：
 *    - 浏览器原生，性能好
 *    - 不支持函数和某些对象
 *
 *    手写：
 *    - 可以自定义处理逻辑
 *    - 可以支持函数
 */

// ============================================================
// 9. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：状态管理中的深拷贝
 *
 * Redux/Vuex 中需要保持状态不可变
 */

const stateExample = `
// Redux reducer
function reducer(state, action) {
  switch (action.type) {
    case 'UPDATE_USER':
      return deepClone({
        ...state,
        user: action.payload,
      });
    default:
      return state;
  }
}
`;

/**
 * 🏢 场景 2：表单数据重置
 *
 * 保存初始值用于重置
 */

const formResetExample = `
const initialFormData = {
  name: '',
  email: '',
  address: { city: '', street: '' },
};

// 保存初始状态
const backup = deepClone(initialFormData);

// 重置表单
function resetForm() {
  formData = deepClone(backup);
}
`;

/**
 * 🏢 场景 3：撤销/重做功能
 *
 * 保存历史状态
 */

const undoRedoExample = `
class History {
  private states: any[] = [];
  private index = -1;

  push(state: any) {
    // 删除当前之后的状态
    this.states = this.states.slice(0, this.index + 1);
    // 深拷贝保存
    this.states.push(deepClone(state));
    this.index++;
  }

  undo() {
    if (this.index > 0) {
      this.index--;
      return deepClone(this.states[this.index]);
    }
    return null;
  }

  redo() {
    if (this.index < this.states.length - 1) {
      this.index++;
      return deepClone(this.states[this.index]);
    }
    return null;
  }
}
`;

export {
  simpleDeepClone,
  deepClone,
  cloneFunction,
  deepCloneIterative,
  modernDeepClone,
  testDeepClone,
  stateExample,
  formResetExample,
  undoRedoExample,
};

