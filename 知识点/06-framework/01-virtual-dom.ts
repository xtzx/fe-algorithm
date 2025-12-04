/**
 * ============================================================
 * 📚 虚拟 DOM 原理
 * ============================================================
 *
 * 面试考察重点：
 * 1. 虚拟 DOM 是什么？为什么需要？
 * 2. 虚拟 DOM 的实现原理
 * 3. 虚拟 DOM 的优缺点
 * 4. 与直接操作 DOM 的对比
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 什么是虚拟 DOM？
 *
 * 虚拟 DOM（Virtual DOM）是用 JavaScript 对象描述真实 DOM 结构的技术。
 *
 * 真实 DOM：
 * <div class="container">
 *   <span>Hello</span>
 * </div>
 *
 * 虚拟 DOM：
 * {
 *   type: 'div',
 *   props: { className: 'container' },
 *   children: [
 *     { type: 'span', props: {}, children: ['Hello'] }
 *   ]
 * }
 *
 * 📊 为什么需要虚拟 DOM？
 *
 * 1. 性能优化
 *    - 批量更新，减少 DOM 操作
 *    - Diff 算法找出最小变更
 *
 * 2. 跨平台
 *    - 不直接依赖浏览器 DOM API
 *    - 可以渲染到 Native、Canvas 等
 *
 * 3. 声明式编程
 *    - 描述"是什么"而不是"怎么做"
 *    - 简化开发心智负担
 */

// ============================================================
// 2. 虚拟 DOM 实现
// ============================================================

/**
 * 📊 VNode 结构定义
 */
interface VNode {
  type: string | Function;  // 标签名或组件
  props: Record<string, any>;
  children: (VNode | string)[];
  key?: string | number;
  el?: HTMLElement | Text;  // 对应的真实 DOM
}

// 创建虚拟 DOM 节点
function createElement(
  type: string | Function,
  props: Record<string, any> | null,
  ...children: (VNode | string)[]
): VNode {
  return {
    type,
    props: props || {},
    children: children.flat(),
    key: props?.key,
  };
}

// JSX 编译后的样子
const jsxExample = `
// JSX 写法
const element = (
  <div className="container">
    <span>Hello</span>
  </div>
);

// 编译后（React 17 之前）
const element = React.createElement(
  'div',
  { className: 'container' },
  React.createElement('span', null, 'Hello')
);

// 编译后（React 17+ jsx-runtime）
import { jsx as _jsx } from 'react/jsx-runtime';
const element = _jsx('div', {
  className: 'container',
  children: _jsx('span', { children: 'Hello' })
});
`;

// ============================================================
// 3. 渲染虚拟 DOM 到真实 DOM
// ============================================================

/**
 * 📊 mount：首次渲染
 */
function mount(vnode: VNode | string, container: HTMLElement) {
  // 文本节点
  if (typeof vnode === 'string') {
    const textNode = document.createTextNode(vnode);
    container.appendChild(textNode);
    return textNode;
  }

  // 函数组件
  if (typeof vnode.type === 'function') {
    const componentVNode = (vnode.type as Function)(vnode.props);
    return mount(componentVNode, container);
  }

  // 元素节点
  const el = document.createElement(vnode.type as string);
  vnode.el = el;

  // 设置属性
  for (const [key, value] of Object.entries(vnode.props)) {
    if (key === 'key') continue;
    if (key.startsWith('on')) {
      // 事件绑定
      const eventName = key.slice(2).toLowerCase();
      el.addEventListener(eventName, value);
    } else if (key === 'className') {
      el.className = value;
    } else if (key === 'style' && typeof value === 'object') {
      Object.assign(el.style, value);
    } else {
      el.setAttribute(key, value);
    }
  }

  // 递归渲染子节点
  for (const child of vnode.children) {
    mount(child, el);
  }

  container.appendChild(el);
  return el;
}

/**
 * 📊 unmount：卸载节点
 */
function unmount(vnode: VNode) {
  if (vnode.el) {
    vnode.el.parentNode?.removeChild(vnode.el);
  }
}

// ============================================================
// 4. 更新虚拟 DOM（简化版 Diff）
// ============================================================

/**
 * 📊 patch：更新节点
 */
function patch(oldVNode: VNode, newVNode: VNode) {
  // 类型不同，直接替换
  if (oldVNode.type !== newVNode.type) {
    const parent = oldVNode.el?.parentNode;
    if (parent) {
      unmount(oldVNode);
      mount(newVNode, parent as HTMLElement);
    }
    return;
  }

  // 复用 DOM 元素
  const el = (newVNode.el = oldVNode.el!);

  // 更新属性
  patchProps(el as HTMLElement, oldVNode.props, newVNode.props);

  // 更新子节点
  patchChildren(el as HTMLElement, oldVNode.children, newVNode.children);
}

/**
 * 📊 patchProps：更新属性
 */
function patchProps(
  el: HTMLElement,
  oldProps: Record<string, any>,
  newProps: Record<string, any>
) {
  // 删除旧属性
  for (const key of Object.keys(oldProps)) {
    if (!(key in newProps)) {
      if (key.startsWith('on')) {
        el.removeEventListener(key.slice(2).toLowerCase(), oldProps[key]);
      } else {
        el.removeAttribute(key);
      }
    }
  }

  // 更新/新增属性
  for (const [key, value] of Object.entries(newProps)) {
    if (oldProps[key] !== value) {
      if (key.startsWith('on')) {
        el.removeEventListener(key.slice(2).toLowerCase(), oldProps[key]);
        el.addEventListener(key.slice(2).toLowerCase(), value);
      } else if (key === 'className') {
        el.className = value;
      } else if (key === 'style' && typeof value === 'object') {
        Object.assign(el.style, value);
      } else {
        el.setAttribute(key, value);
      }
    }
  }
}

/**
 * 📊 patchChildren：更新子节点（简化版）
 */
function patchChildren(
  el: HTMLElement,
  oldChildren: (VNode | string)[],
  newChildren: (VNode | string)[]
) {
  const commonLength = Math.min(oldChildren.length, newChildren.length);

  // 更新公共部分
  for (let i = 0; i < commonLength; i++) {
    const oldChild = oldChildren[i];
    const newChild = newChildren[i];

    if (typeof oldChild === 'string' || typeof newChild === 'string') {
      if (oldChild !== newChild) {
        el.childNodes[i].textContent = String(newChild);
      }
    } else {
      patch(oldChild, newChild);
    }
  }

  // 删除多余节点
  if (oldChildren.length > newChildren.length) {
    for (let i = commonLength; i < oldChildren.length; i++) {
      const child = oldChildren[i];
      if (typeof child !== 'string') {
        unmount(child);
      } else {
        el.childNodes[commonLength]?.remove();
      }
    }
  }

  // 新增节点
  if (newChildren.length > oldChildren.length) {
    for (let i = commonLength; i < newChildren.length; i++) {
      mount(newChildren[i], el);
    }
  }
}

// ============================================================
// 5. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见误解
 *
 * 1. "虚拟 DOM 比直接操作 DOM 快"
 *    - ❌ 错误！虚拟 DOM 本身有额外开销
 *    - ✅ 虚拟 DOM 的优势是：
 *      · 减少不必要的 DOM 操作
 *      · 批量更新
 *      · 跨平台能力
 *      · 声明式编程体验
 *
 * 2. "虚拟 DOM 就是内存中的 DOM"
 *    - ❌ 虚拟 DOM 是 JS 对象，不是 DOM 节点
 *    - ✅ 它只是描述 DOM 结构的数据
 *
 * 3. "React 就是虚拟 DOM"
 *    - ❌ 虚拟 DOM 只是 React 的一部分
 *    - ✅ React 还包括组件模型、Hooks、Fiber 等
 *
 * 4. key 的作用
 *    - ❌ key 不只是消除警告
 *    - ✅ key 用于帮助 Diff 算法识别节点
 *    - ❌ 不要用 index 作为 key（列表变化时会出问题）
 */

// ============================================================
// 6. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 虚拟 DOM 一定比直接操作 DOM 快吗？
 * A: 不一定。
 *    - 单次简单操作，直接 DOM 更快
 *    - 复杂场景、批量更新，虚拟 DOM 更有优势
 *    - 虚拟 DOM 的主要价值是声明式 + 跨平台
 *
 * Q2: React 和 Vue 的虚拟 DOM 有什么区别？
 * A:
 *    React：
 *    - 每次渲染生成新的虚拟 DOM 树
 *    - 依赖 shouldComponentUpdate / memo 优化
 *
 *    Vue：
 *    - 编译时标记静态节点
 *    - 响应式系统精确追踪变化
 *    - 更新粒度更细
 *
 * Q3: 为什么需要 key？
 * A:
 *    - 帮助 Diff 算法识别哪些节点变化了
 *    - 没有 key 时只能按顺序对比
 *    - 有 key 时可以复用 DOM 节点
 *
 * Q4: 为什么不推荐用 index 作为 key？
 * A:
 *    - 列表顺序变化时，index 会变化
 *    - 导致错误的节点复用
 *    - 可能导致状态错乱和性能问题
 *
 * Q5: 虚拟 DOM 如何实现跨平台？
 * A:
 *    - 虚拟 DOM 只是 JS 对象
 *    - 渲染器（Renderer）负责将虚拟 DOM 转换为目标平台
 *    - React Native、ReactART、react-three-fiber 等
 */

// ============================================================
// 7. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：列表渲染性能问题
 *
 * 问题：
 * - 使用 index 作为 key
 * - 列表头部插入/删除时性能差
 *
 * 原因：
 * - index 变化导致所有节点都被认为变化了
 * - 无法复用 DOM 节点
 *
 * 解决：
 * - 使用稳定的唯一标识作为 key（如 id）
 */

/**
 * 🏢 场景 2：大列表渲染卡顿
 *
 * 问题：
 * - 渲染 10000 个节点
 * - 虚拟 DOM 创建和 Diff 耗时
 *
 * 解决：
 * - 虚拟滚动
 * - 分页加载
 * - 使用 windowing 库（react-window）
 */

/**
 * 🏢 场景 3：频繁更新性能问题
 *
 * 问题：
 * - 每次状态变化都生成完整的虚拟 DOM 树
 * - 即使只有小部分变化
 *
 * 解决：
 * React：
 * - React.memo 避免不必要渲染
 * - useMemo 缓存计算结果
 *
 * Vue：
 * - 响应式系统自动追踪依赖
 * - 编译时优化
 */

// ============================================================
// 8. 完整示例：Mini Virtual DOM
// ============================================================

class MiniReact {
  private container: HTMLElement;
  private currentVNode: VNode | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  render(vnode: VNode) {
    if (this.currentVNode) {
      patch(this.currentVNode, vnode);
    } else {
      mount(vnode, this.container);
    }
    this.currentVNode = vnode;
  }
}

// 使用示例
const miniReactUsage = `
const app = new MiniReact(document.getElementById('root')!);

// 首次渲染
app.render(
  createElement('div', { className: 'app' },
    createElement('h1', null, 'Hello'),
    createElement('button', { onClick: () => console.log('clicked') }, 'Click')
  )
);

// 更新
app.render(
  createElement('div', { className: 'app' },
    createElement('h1', null, 'Hello World'),
    createElement('button', { onClick: () => console.log('clicked') }, 'Click')
  )
);
`;

export {
  createElement,
  mount,
  unmount,
  patch,
  patchProps,
  patchChildren,
  MiniReact,
  jsxExample,
  miniReactUsage,
};

export type { VNode };

