/**
 * ============================================================
 * 📚 Phase 1: JSX 与 React 元素
 * ============================================================
 *
 * 🎯 学习目标：
 * 1. 理解 JSX 的本质
 * 2. 掌握 React.createElement 的实现
 * 3. 理解 ReactElement 数据结构
 *
 * 📁 源码位置：
 * - packages/react/src/ReactElement.js
 * - packages/react/src/jsx/ReactJSXElement.js
 *
 * ⏱️ 预计时间：3 小时
 */

// ============================================================
// 1. JSX 的本质
// ============================================================

/**
 * 📊 JSX 编译过程
 *
 * JSX 代码：
 * ```jsx
 * <div className="container">
 *   <h1>Hello</h1>
 *   <p>World</p>
 * </div>
 * ```
 *
 * 编译后（React 17+ 使用新 JSX 转换）：
 * ```js
 * import { jsx as _jsx, jsxs as _jsxs } from 'react/jsx-runtime';
 *
 * _jsxs('div', {
 *   className: 'container',
 *   children: [
 *     _jsx('h1', { children: 'Hello' }),
 *     _jsx('p', { children: 'World' })
 *   ]
 * });
 * ```
 *
 * 旧版编译（React 17 之前）：
 * ```js
 * React.createElement('div', { className: 'container' },
 *   React.createElement('h1', null, 'Hello'),
 *   React.createElement('p', null, 'World')
 * );
 * ```
 */

// ============================================================
// 2. ReactElement 数据结构
// ============================================================

/**
 * 📊 ReactElement 结构
 *
 * 源码位置：packages/react/src/ReactElement.js
 *
 * ```js
 * const element = {
 *   $$typeof: REACT_ELEMENT_TYPE,  // 元素类型标识
 *   type: 'div',                   // 元素类型（string/function/class）
 *   key: null,                     // diff 用的 key
 *   ref: null,                     // ref 引用
 *   props: {                       // 属性
 *     className: 'container',
 *     children: [...]
 *   },
 *   _owner: null,                  // 所属的 Fiber 节点
 * };
 * ```
 */

// 简化版 createElement 实现
function createElement(
  type: string | Function,
  config: Record<string, any> | null,
  ...children: any[]
) {
  const props: Record<string, any> = {};
  let key = null;
  let ref = null;

  // 1. 处理 config
  if (config != null) {
    // 提取 key
    if (config.key !== undefined) {
      key = '' + config.key;
    }
    // 提取 ref
    if (config.ref !== undefined) {
      ref = config.ref;
    }
    // 复制其他属性到 props
    for (const propName in config) {
      if (
        Object.prototype.hasOwnProperty.call(config, propName) &&
        propName !== 'key' &&
        propName !== 'ref'
      ) {
        props[propName] = config[propName];
      }
    }
  }

  // 2. 处理 children
  const childrenLength = children.length;
  if (childrenLength === 1) {
    props.children = children[0];
  } else if (childrenLength > 1) {
    props.children = children;
  }

  // 3. 创建 ReactElement
  return {
    $$typeof: Symbol.for('react.element'),
    type,
    key,
    ref,
    props,
    _owner: null,
  };
}

// ============================================================
// 3. $$typeof 的作用
// ============================================================

/**
 * 📊 $$typeof 安全机制
 *
 * 问题：XSS 攻击可能注入恶意对象
 *
 * ```js
 * // 恶意代码可能构造这样的对象
 * const malicious = {
 *   type: 'script',
 *   props: { dangerouslySetInnerHTML: { __html: '...' } }
 * };
 * ```
 *
 * 解决：使用 Symbol 作为标识
 *
 * ```js
 * const REACT_ELEMENT_TYPE = Symbol.for('react.element');
 * ```
 *
 * 因为 JSON 不支持 Symbol，所以：
 * - 服务端返回的 JSON 无法伪造 $$typeof
 * - React 只渲染带有正确 $$typeof 的对象
 */

// ============================================================
// 4. 组件类型判断
// ============================================================

/**
 * 📊 type 的不同类型
 *
 * 1. 原生标签：type = 'div' | 'span' | ...
 * 2. 函数组件：type = Function
 * 3. 类组件：type = Class (有 prototype.isReactComponent)
 * 4. Fragment：type = Symbol.for('react.fragment')
 * 5. Portal：type = Symbol.for('react.portal')
 * 6. Context：type = Symbol.for('react.context')
 * 7. Memo：type = Symbol.for('react.memo')
 * 8. Lazy：type = Symbol.for('react.lazy')
 */

// 判断是否为类组件
function isClassComponent(type: any): boolean {
  return (
    typeof type === 'function' &&
    type.prototype &&
    type.prototype.isReactComponent
  );
}

// ============================================================
// 5. 💡 面试题
// ============================================================

/**
 * 💡 Q1: JSX 的本质是什么？
 *
 * A: JSX 是 React.createElement 的语法糖。
 *    Babel 会将 JSX 编译为 createElement 调用，
 *    createElement 返回一个 ReactElement 对象（虚拟 DOM）。
 *
 * 💡 Q2: $$typeof 有什么作用？
 *
 * A: 防止 XSS 攻击。
 *    $$typeof 使用 Symbol 类型，JSON 无法序列化 Symbol，
 *    所以服务端返回的恶意数据无法伪造成 ReactElement。
 *
 * 💡 Q3: React 如何区分函数组件和类组件？
 *
 * A: 检查 type.prototype.isReactComponent
 *    类组件继承自 React.Component，有这个属性
 *    函数组件没有
 *
 * 💡 Q4: key 和 ref 为什么不在 props 中？
 *
 * A: key 用于 Diff 算法，ref 用于获取实例引用
 *    它们是 React 内部使用的特殊属性
 *    不应该被组件访问，所以单独提取出来
 */

// ============================================================
// 6. 🏢 实际开发应用
// ============================================================

/**
 * 🏢 应用 1：动态创建元素
 *
 * 理解 createElement 后，可以动态创建组件
 */
const DynamicComponent = ({ component, ...props }: any) => {
  return createElement(component, props);
};

/**
 * 🏢 应用 2：理解 children
 *
 * props.children 可能是：
 * - undefined（无子元素）
 * - 单个元素
 * - 数组
 */
const Container = ({ children }: { children?: React.ReactNode }) => {
  // React.Children.map 统一处理各种情况
  // 源码位置：packages/react/src/ReactChildren.js
  return createElement('div', null, children);
};

/**
 * 🏢 应用 3：自定义 JSX 运行时
 *
 * 了解 createElement 后，可以实现自定义渲染器
 * 比如渲染到 Canvas、Native 等
 */

// ============================================================
// 7. 📖 源码阅读指南
// ============================================================

/**
 * 📖 阅读顺序：
 *
 * 1. packages/react/src/ReactElement.js
 *    - createElement 函数
 *    - isValidElement 函数
 *
 * 2. packages/react/src/jsx/ReactJSXElement.js
 *    - jsx 函数（新版 JSX 转换）
 *    - jsxs 函数（多子元素）
 *
 * 3. packages/shared/ReactSymbols.js
 *    - 各种 Symbol 定义
 *
 * 4. packages/react/src/ReactChildren.js
 *    - Children.map/forEach/count/only/toArray
 */

// ============================================================
// 8. ✅ 学习检查
// ============================================================

/**
 * ✅ 完成以下任务：
 *
 * - [ ] 理解 JSX 编译过程
 * - [ ] 理解 ReactElement 结构
 * - [ ] 理解 $$typeof 的安全作用
 * - [ ] 能手写简化版 createElement
 * - [ ] 阅读源码：ReactElement.js
 */

export { createElement, isClassComponent, DynamicComponent, Container };

