/**
 * ============================================================
 * 📚 Phase 1: JSX 与 React 元素深度解析
 * ============================================================
 *
 * 🎯 学习目标：
 * 1. 理解 JSX 编译过程
 * 2. 掌握 React Element 数据结构
 * 3. 理解 createElement 和 jsx 的区别
 * 4. 理解 $$typeof 的安全作用
 *
 * 📁 核心源码位置：
 * - packages/react/src/ReactElement.js      # createElement、jsx
 * - packages/react/src/jsx/ReactJSXElement.js # 新 JSX 运行时
 * - packages/shared/ReactSymbols.js         # 类型标识
 *
 * ⏱️ 预计时间：3-4 小时
 * 🎯 面试权重：⭐⭐⭐
 */

// ============================================================
// Part 1: JSX 是什么？
// ============================================================

/**
 * 📊 JSX 的本质
 *
 * JSX 是语法糖！它最终会被 Babel 编译为函数调用
 *
 * 编译前（JSX）:
 * ```jsx
 * const element = (
 *   <div className="container">
 *     <h1>Hello</h1>
 *     <p>World</p>
 *   </div>
 * );
 * ```
 *
 * 编译后（旧版 - Classic Runtime）:
 * ```javascript
 * const element = React.createElement(
 *   'div',
 *   { className: 'container' },
 *   React.createElement('h1', null, 'Hello'),
 *   React.createElement('p', null, 'World')
 * );
 * ```
 *
 * 编译后（新版 - Automatic Runtime，React 17+）:
 * ```javascript
 * import { jsx as _jsx, jsxs as _jsxs } from 'react/jsx-runtime';
 *
 * const element = _jsxs('div', {
 *   className: 'container',
 *   children: [
 *     _jsx('h1', { children: 'Hello' }),
 *     _jsx('p', { children: 'World' })
 *   ]
 * });
 * ```
 */

// ============================================================
// Part 2: JSX 编译模式对比
// ============================================================

/**
 * 📊 Classic vs Automatic Runtime
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                    JSX 编译模式对比                              │
 * │                                                                 │
 * │  特性          │ Classic Runtime     │ Automatic Runtime        │
 * │  ─────────────│─────────────────────│─────────────────────────  │
 * │  引入方式      │ 必须 import React   │ 自动引入 jsx-runtime      │
 * │  函数名        │ React.createElement │ jsx / jsxs               │
 * │  children 传递 │ 作为额外参数         │ 作为 props.children      │
 * │  key 传递      │ 在 props 中         │ 作为第三参数              │
 * │  性能          │ 稍慢                │ 略快（减少参数处理）      │
 * │  支持版本      │ 所有版本            │ React 17+                │
 * │                                                                 │
 * │  Babel 配置：                                                   │
 * │  Classic:   { "runtime": "classic" }                           │
 * │  Automatic: { "runtime": "automatic" }  // 默认                 │
 * └─────────────────────────────────────────────────────────────────┘
 */

// Classic Runtime 示例
const classicExample = `
// 📁 需要手动引入 React
import React from 'react';

// JSX
const element = <div className="hello">Hello</div>;

// 编译结果
const element = React.createElement(
  'div',
  { className: 'hello' },
  'Hello'
);
`;

// Automatic Runtime 示例
const automaticExample = `
// 📁 无需手动引入 React
// Babel 会自动添加 import

// JSX
const element = <div className="hello">Hello</div>;

// 编译结果
import { jsx as _jsx } from 'react/jsx-runtime';

const element = _jsx('div', {
  className: 'hello',
  children: 'Hello'
});
`;

// ============================================================
// Part 3: React Element 数据结构（核心！）
// ============================================================

/**
 * 📁 源码位置: packages/react/src/ReactElement.js (第 148-202 行)
 *
 * ReactElement 是一个普通的 JavaScript 对象
 * 它描述了"你想在屏幕上看到什么"
 */

// React Element 的结构
interface ReactElement {
  // 🔑 类型标识符 - 用于识别这是一个 React 元素
  // 值为 Symbol.for('react.element')
  $$typeof: symbol;

  // 元素类型
  // - 字符串: 'div', 'span' (原生 DOM)
  // - 函数: function Component() {} (函数组件)
  // - 类: class Component {} (类组件)
  // - Symbol: Fragment, StrictMode (内置组件)
  type: string | Function | symbol;

  // 唯一标识，用于 Diff 算法优化
  key: string | null;

  // 引用，用于访问 DOM 节点或组件实例
  ref: any;

  // 属性对象（不包含 key、ref）
  props: {
    children?: ReactElement | ReactElement[] | string | number;
    [propName: string]: any;
  };

  // 创建这个元素的组件（内部使用）
  _owner: any;

  // 开发模式下的额外属性
  _store?: { validated: boolean };
  _self?: any;    // 调试用
  _source?: any;  // 源码位置信息
}

/**
 * 📊 实际示例
 */

const actualElementExample = `
// JSX
const element = <div className="container" key="unique">Hello</div>;

// 生成的 React Element 对象
{
  $$typeof: Symbol(react.element),
  type: 'div',
  key: 'unique',
  ref: null,
  props: {
    className: 'container',
    children: 'Hello'
  },
  _owner: null
}
`;

// ============================================================
// Part 4: createElement 源码解析
// ============================================================

/**
 * 📁 源码位置: packages/react/src/ReactElement.js (第 362-451 行)
 */

// 保留属性（不会传递给 props）
const RESERVED_PROPS = {
  key: true,      // 用于 Diff 算法
  ref: true,      // 用于获取引用
  __self: true,   // 开发模式调试
  __source: true, // 开发模式源码位置
};

/**
 * createElement 简化实现
 *
 * @param type - 元素类型 ('div' | Component)
 * @param config - 属性配置 ({ className: 'x', onClick: fn })
 * @param children - 子元素（可变参数）
 */
function createElementSimplified(
  type: string | Function,
  config: Record<string, any> | null,
  ...children: any[]
): ReactElement {
  const props: Record<string, any> = {};
  let key: string | null = null;
  let ref: any = null;

  // 1. 处理 config
  if (config != null) {
    // 提取 ref
    if (config.ref !== undefined) {
      ref = config.ref;
    }
    // 提取 key（转为字符串）
    if (config.key !== undefined) {
      key = '' + config.key;
    }
    // 复制其他属性到 props
    for (const propName in config) {
      if (
        Object.hasOwnProperty.call(config, propName) &&
        !RESERVED_PROPS[propName]
      ) {
        props[propName] = config[propName];
      }
    }
  }

  // 2. 处理 children
  if (children.length === 1) {
    props.children = children[0];
  } else if (children.length > 1) {
    props.children = children;
  }

  // 3. 处理 defaultProps
  if (type && (type as any).defaultProps) {
    const defaultProps = (type as any).defaultProps;
    for (const propName in defaultProps) {
      if (props[propName] === undefined) {
        props[propName] = defaultProps[propName];
      }
    }
  }

  // 4. 创建并返回 ReactElement
  return {
    $$typeof: Symbol.for('react.element'),
    type,
    key,
    ref,
    props,
    _owner: null, // 实际源码中是 ReactCurrentOwner.current
  };
}

// ============================================================
// Part 5: jsx/jsxs 源码解析（新版运行时）
// ============================================================

/**
 * 📁 源码位置: packages/react/src/ReactElement.js (第 210-272 行)
 *
 * jsx 和 jsxs 的区别：
 * - jsx: 单个子元素或无子元素
 * - jsxs: 多个子元素（静态，编译时确定）
 *
 * 为什么要区分？
 * - jsxs 可以跳过某些运行时检查（性能优化）
 * - 编译器知道子元素是静态的，不需要 key 检查
 */

function jsxSimplified(
  type: string | Function,
  config: Record<string, any>,
  maybeKey?: string
): ReactElement {
  const props: Record<string, any> = {};
  let key: string | null = null;
  let ref: any = null;

  // 1. key 作为第三参数传入（新版特性）
  if (maybeKey !== undefined) {
    key = '' + maybeKey;
  }

  // 2. 也支持从 config 中读取 key
  if (config.key !== undefined) {
    key = '' + config.key;
  }

  // 3. 提取 ref
  if (config.ref !== undefined) {
    ref = config.ref;
  }

  // 4. 复制属性（children 已经在 config 中）
  for (const propName in config) {
    if (
      Object.hasOwnProperty.call(config, propName) &&
      !RESERVED_PROPS[propName]
    ) {
      props[propName] = config[propName];
    }
  }

  // 5. 处理 defaultProps
  if (type && (type as any).defaultProps) {
    const defaultProps = (type as any).defaultProps;
    for (const propName in defaultProps) {
      if (props[propName] === undefined) {
        props[propName] = defaultProps[propName];
      }
    }
  }

  return {
    $$typeof: Symbol.for('react.element'),
    type,
    key,
    ref,
    props,
    _owner: null,
  };
}

/**
 * 📊 jsx vs createElement 参数对比
 */

const jsxVsCreateElement = `
// createElement 方式
React.createElement(
  'div',
  { className: 'container', key: 'unique' },  // key 在 config 中
  child1,
  child2,                                      // children 作为额外参数
  child3
);

// jsx 方式
jsx('div', {
  className: 'container',
  children: [child1, child2, child3]           // children 在 props 中
}, 'unique');                                  // key 作为第三参数
`;

// ============================================================
// Part 6: $$typeof 的安全作用（重要！）
// ============================================================

/**
 * 📁 源码位置: packages/shared/ReactSymbols.js
 *
 * 为什么用 Symbol？安全！防止 XSS 攻击
 */

const securityExplanation = `
🚨 安全问题场景：

假设有一个论坛，用户可以发布 JSON 数据，服务器返回：

// 恶意用户构造的数据
{
  type: 'div',
  props: {
    dangerouslySetInnerHTML: {
      __html: '<script>alert("XSS!")</script>'
    }
  }
}

如果 React 直接渲染这个对象，就会造成 XSS 攻击！

🛡️ $$typeof 保护机制：

React Element 必须有 $$typeof: Symbol.for('react.element')

但是！JSON.parse 无法生成 Symbol！

const json = JSON.stringify({ $$typeof: Symbol.for('react.element') });
// 结果: "{}"  Symbol 被忽略了！

const parsed = JSON.parse(json);
// 结果: {}  没有 $$typeof

React 检查到没有 $$typeof 或值不对，就不会渲染这个对象！
`;

/**
 * 📊 isValidElement 检查
 *
 * 📁 源码位置: packages/react/src/ReactElement.js (第 567-573 行)
 */

function isValidElement(object: any): boolean {
  return (
    typeof object === 'object' &&
    object !== null &&
    object.$$typeof === Symbol.for('react.element')
  );
}

// ============================================================
// Part 7: React 内置组件类型（Symbol）
// ============================================================

/**
 * 📁 源码位置: packages/shared/ReactSymbols.js
 *
 * React 使用不同的 Symbol 来标识不同类型的"元素"
 */

const ReactSymbols = {
  // 普通元素
  REACT_ELEMENT_TYPE: Symbol.for('react.element'),

  // Portal（渲染到其他 DOM 节点）
  REACT_PORTAL_TYPE: Symbol.for('react.portal'),

  // 内置组件
  REACT_FRAGMENT_TYPE: Symbol.for('react.fragment'),       // <>...</>
  REACT_STRICT_MODE_TYPE: Symbol.for('react.strict_mode'), // <StrictMode>
  REACT_PROFILER_TYPE: Symbol.for('react.profiler'),       // <Profiler>
  REACT_SUSPENSE_TYPE: Symbol.for('react.suspense'),       // <Suspense>
  REACT_SUSPENSE_LIST_TYPE: Symbol.for('react.suspense_list'),

  // Context
  REACT_PROVIDER_TYPE: Symbol.for('react.provider'),       // <Context.Provider>
  REACT_CONTEXT_TYPE: Symbol.for('react.context'),         // <Context.Consumer>

  // 高阶组件标识
  REACT_FORWARD_REF_TYPE: Symbol.for('react.forward_ref'), // forwardRef()
  REACT_MEMO_TYPE: Symbol.for('react.memo'),               // memo()
  REACT_LAZY_TYPE: Symbol.for('react.lazy'),               // lazy()
};

/**
 * 📊 不同类型元素的 type 值
 */

const elementTypeExamples = `
// 1. 原生 DOM 元素
<div />
// type: 'div'

// 2. 函数组件
function MyComponent() { return <div />; }
<MyComponent />
// type: MyComponent (函数引用)

// 3. 类组件
class MyClass extends React.Component { render() { return <div />; } }
<MyClass />
// type: MyClass (类引用)

// 4. Fragment
<>content</>
// type: Symbol.for('react.fragment')

// 5. Context.Provider
<MyContext.Provider value={1}>
// type: { $$typeof: Symbol.for('react.provider'), _context: MyContext }

// 6. forwardRef 组件
const ForwardedComponent = forwardRef((props, ref) => <div ref={ref} />);
<ForwardedComponent />
// type: { $$typeof: Symbol.for('react.forward_ref'), render: fn }

// 7. memo 组件
const MemoComponent = memo(MyComponent);
<MemoComponent />
// type: { $$typeof: Symbol.for('react.memo'), type: MyComponent }
`;

// ============================================================
// Part 8: Children 工具函数
// ============================================================

/**
 * 📁 源码位置: packages/react/src/ReactChildren.js
 *
 * React.Children 提供了处理 props.children 的工具函数
 */

const childrenAPI = `
// props.children 可能是：
// - 单个元素: <Parent><Child /></Parent>
// - 数组: <Parent>{[<A/>, <B/>]}</Parent>
// - 字符串: <Parent>text</Parent>
// - 数字: <Parent>{123}</Parent>
// - null/undefined: <Parent>{null}</Parent>

// React.Children 工具函数
React.Children.map(children, fn);     // 遍历并转换
React.Children.forEach(children, fn); // 只遍历
React.Children.count(children);       // 统计数量
React.Children.only(children);        // 确保只有一个子元素
React.Children.toArray(children);     // 转为扁平数组

// 示例
function MyComponent({ children }) {
  // 为每个子元素添加 key
  return React.Children.map(children, (child, index) => {
    return React.cloneElement(child, { key: index });
  });
}
`;

// ============================================================
// Part 9: cloneElement 解析
// ============================================================

/**
 * 📁 源码位置: packages/react/src/ReactElement.js (第 486-558 行)
 *
 * cloneElement 用于克隆元素并覆盖部分属性
 */

function cloneElementSimplified(
  element: ReactElement,
  config?: Record<string, any>,
  ...children: any[]
): ReactElement {
  // 1. 复制原有 props
  const props = { ...element.props };

  // 2. 保留原有 key 和 ref
  let key = element.key;
  let ref = element.ref;

  // 3. 如果 config 中有新值，则覆盖
  if (config != null) {
    if (config.ref !== undefined) {
      ref = config.ref;
    }
    if (config.key !== undefined) {
      key = '' + config.key;
    }
    // 覆盖其他属性
    for (const propName in config) {
      if (
        Object.hasOwnProperty.call(config, propName) &&
        !RESERVED_PROPS[propName]
      ) {
        props[propName] = config[propName];
      }
    }
  }

  // 4. 处理 children
  if (children.length === 1) {
    props.children = children[0];
  } else if (children.length > 1) {
    props.children = children;
  }

  // 5. 创建新元素
  return {
    $$typeof: Symbol.for('react.element'),
    type: element.type,
    key,
    ref,
    props,
    _owner: element._owner,
  };
}

// 使用示例
const cloneElementUsage = `
// 场景：给子元素注入额外的 props
function RadioGroup({ children, selectedValue, onChange }) {
  return (
    <div>
      {React.Children.map(children, child => {
        // 克隆子元素，注入 checked 和 onChange
        return React.cloneElement(child, {
          checked: child.props.value === selectedValue,
          onChange: () => onChange(child.props.value)
        });
      })}
    </div>
  );
}

// 使用
<RadioGroup selectedValue="a" onChange={handleChange}>
  <Radio value="a">Option A</Radio>
  <Radio value="b">Option B</Radio>
</RadioGroup>
`;

// ============================================================
// Part 10: 面试题
// ============================================================

const interviewQuestions = `
💡 Q1: JSX 是什么？它会被编译成什么？
A: JSX 是 JavaScript 的语法扩展，允许在 JS 中写类似 HTML 的代码。
   它会被 Babel 编译成 React.createElement() 或 jsx() 函数调用。
   最终返回一个描述 UI 的普通 JavaScript 对象（React Element）。

💡 Q2: React Element 和 Component 有什么区别？
A: - Element 是一个普通对象，描述你想在屏幕上看到什么
   - Component 是一个函数或类，接收 props，返回 Element
   - Element 是 Component 的"输出"
   - Element 是不可变的，每次渲染都会创建新的

💡 Q3: 为什么需要 $$typeof？
A: 防止 XSS 攻击。因为 Symbol 无法通过 JSON.parse 创建，
   所以即使攻击者构造了恶意的 JSON 对象，也无法被 React 渲染。
   React 检查 $$typeof === Symbol.for('react.element')。

💡 Q4: key 为什么不能在 props 中访问？
A: key 是 React 内部使用的特殊属性，用于 Diff 算法。
   它不会传递到组件的 props 中。
   如果需要相同的值，应该用另一个 prop 名传递。

💡 Q5: createElement 和 jsx 有什么区别？
A: 1. jsx 是 React 17 引入的新运行时
   2. jsx 的 children 在 props 对象中，createElement 作为额外参数
   3. jsx 的 key 作为第三参数，createElement 在 config 中
   4. jsx 无需手动 import React
   5. jsx 性能略好（减少参数处理）

💡 Q6: React.Children.map 和普通 map 有什么区别？
A: 1. React.Children.map 能处理 null/undefined
   2. 能正确处理单个子元素（不是数组）
   3. 自动扁平化嵌套数组
   4. 自动添加正确的 key 前缀

💡 Q7: Fragment 和 div 有什么区别？
A: 1. Fragment 不会创建额外的 DOM 节点
   2. Fragment 可以使用短语法 <></>
   3. 需要 key 时必须用 <Fragment key={...}>
   4. 性能略好（减少 DOM 层级）
`;

// ============================================================
// Part 11: 实践练习
// ============================================================

/**
 * 练习 1：手写简化版 createElement
 */
function myCreateElement(
  type: any,
  config: any,
  ...children: any[]
) {
  // 你的实现
  const props: Record<string, any> = {};
  let key = null;
  let ref = null;

  if (config != null) {
    if (config.key !== undefined) key = '' + config.key;
    if (config.ref !== undefined) ref = config.ref;

    for (const prop in config) {
      if (
        Object.hasOwnProperty.call(config, prop) &&
        prop !== 'key' &&
        prop !== 'ref'
      ) {
        props[prop] = config[prop];
      }
    }
  }

  if (children.length === 1) {
    props.children = children[0];
  } else if (children.length > 1) {
    props.children = children;
  }

  return {
    $$typeof: Symbol.for('react.element'),
    type,
    key,
    ref,
    props,
  };
}

/**
 * 练习 2：实现 isValidElement
 */
function myIsValidElement(object: any): boolean {
  return (
    typeof object === 'object' &&
    object !== null &&
    object.$$typeof === Symbol.for('react.element')
  );
}

/**
 * 练习 3：实现 Children.count
 */
function myChildrenCount(children: any): number {
  let count = 0;
  
  function countChild(child: any) {
    if (child == null || typeof child === 'boolean') {
      return;
    }
    if (Array.isArray(child)) {
      child.forEach(countChild);
    } else {
      count++;
    }
  }
  
  countChild(children);
  return count;
}

// ============================================================
// 学习检查清单
// ============================================================

/**
 * ✅ Phase 1 学习检查
 *
 * JSX 编译：
 * - [ ] 理解 JSX 是语法糖，会被编译为函数调用
 * - [ ] 理解 Classic 和 Automatic 两种运行时的区别
 * - [ ] 能说出 jsx 和 createElement 的参数差异
 *
 * React Element：
 * - [ ] 能画出 React Element 的数据结构
 * - [ ] 理解 $$typeof 的安全作用
 * - [ ] 理解 key 和 ref 是保留属性
 *
 * API 理解：
 * - [ ] 能手写简化版 createElement
 * - [ ] 理解 React.Children 的作用
 * - [ ] 理解 cloneElement 的使用场景
 *
 * 源码位置：
 * - [ ] 能找到 createElement 源码
 * - [ ] 能找到 ReactSymbols 定义
 */

export {
  createElementSimplified,
  jsxSimplified,
  isValidElement,
  cloneElementSimplified,
  myCreateElement,
  myIsValidElement,
  myChildrenCount,
  RESERVED_PROPS,
  ReactSymbols,
  classicExample,
  automaticExample,
  actualElementExample,
  jsxVsCreateElement,
  securityExplanation,
  elementTypeExamples,
  childrenAPI,
  cloneElementUsage,
  interviewQuestions,
};
