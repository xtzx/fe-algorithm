/**
 * ============================================================
 * 📚 Phase 8: 事件系统
 * ============================================================
 *
 * 🎯 学习目标：
 * 1. 理解 React 事件与原生事件的区别
 * 2. 理解事件委托机制
 * 3. 理解合成事件
 * 4. 理解事件优先级
 *
 * 📁 源码位置：
 * - packages/react-dom/src/events/
 *
 * ⏱️ 预计时间：4 小时
 * 🔥 面试权重：⭐⭐⭐
 */

// ============================================================
// 1. React 事件系统概述
// ============================================================

/**
 * 📊 React 事件 vs 原生事件
 *
 * ```
 * 原生事件：
 * <button onclick="handleClick()">Click</button>
 *
 * React 事件：
 * <button onClick={handleClick}>Click</button>
 * ```
 *
 * 主要区别：
 *
 * | 特性 | 原生事件 | React 事件 |
 * |------|---------|-----------|
 * | 命名 | onclick | onClick |
 * | 绑定位置 | 元素本身 | 根容器（委托）|
 * | 事件对象 | Event | SyntheticEvent |
 * | 阻止默认 | return false | e.preventDefault() |
 */

/**
 * 📊 事件委托机制
 *
 * React 17+ 将事件委托到根容器：
 *
 * ```
 *                    Root Container
 *                    ┌─────────────────┐
 *                    │   所有事件都     │
 *                    │   绑定在这里     │
 *                    └────────┬────────┘
 *                             │
 *          ┌──────────────────┼──────────────────┐
 *          │                  │                  │
 *     ┌────▼────┐       ┌─────▼─────┐      ┌────▼────┐
 *     │   App   │       │  事件冒泡  │      │  事件捕获│
 *     └────┬────┘       │  时处理   │      │  时处理 │
 *          │            └───────────┘      └─────────┘
 *     ┌────▼────┐
 *     │ Button  │  ◄─── 点击这里
 *     └─────────┘
 * ```
 *
 * React 16 及之前：委托到 document
 * React 17+：委托到根容器（createRoot 挂载的元素）
 */

// ============================================================
// 2. 合成事件
// ============================================================

/**
 * 📊 SyntheticEvent（合成事件）
 *
 * React 封装了原生事件，提供跨浏览器一致的接口
 *
 * ```js
 * interface SyntheticEvent {
 *   // 原生事件对象
 *   nativeEvent: Event;
 *
 *   // 事件目标
 *   target: EventTarget;
 *   currentTarget: EventTarget;
 *
 *   // 事件类型
 *   type: string;
 *
 *   // 方法
 *   preventDefault(): void;
 *   stopPropagation(): void;
 *   persist(): void;  // React 17 后不再需要
 *
 *   // 其他属性
 *   bubbles: boolean;
 *   cancelable: boolean;
 *   timeStamp: number;
 * }
 * ```
 */

// 简化版合成事件
class SyntheticEvent {
  nativeEvent: Event;
  target: EventTarget | null;
  currentTarget: EventTarget | null;
  type: string;
  bubbles: boolean;
  cancelable: boolean;
  defaultPrevented: boolean = false;
  _isPropagationStopped: boolean = false;

  constructor(nativeEvent: Event) {
    this.nativeEvent = nativeEvent;
    this.target = nativeEvent.target;
    this.currentTarget = nativeEvent.currentTarget;
    this.type = nativeEvent.type;
    this.bubbles = nativeEvent.bubbles;
    this.cancelable = nativeEvent.cancelable;
  }

  preventDefault() {
    this.defaultPrevented = true;
    this.nativeEvent.preventDefault();
  }

  stopPropagation() {
    this._isPropagationStopped = true;
    this.nativeEvent.stopPropagation();
  }

  isPropagationStopped(): boolean {
    return this._isPropagationStopped;
  }
}

// ============================================================
// 3. 事件注册与触发
// ============================================================

/**
 * 📊 事件注册流程
 *
 * 1. createRoot 时，在根容器上注册所有事件
 * 2. 不是每个组件单独绑定
 *
 * ```js
 * // 简化的注册逻辑
 * function createRoot(container) {
 *   // 注册所有支持的事件
 *   allNativeEvents.forEach(eventName => {
 *     // 捕获阶段
 *     container.addEventListener(eventName, dispatchEvent, true);
 *     // 冒泡阶段
 *     container.addEventListener(eventName, dispatchEvent, false);
 *   });
 * }
 * ```
 */

/**
 * 📊 事件触发流程
 *
 * ```
 * 用户点击 Button
 *        │
 *        ▼
 * 原生事件冒泡到根容器
 *        │
 *        ▼
 * ┌─────────────────────────────┐
 * │     dispatchEvent           │
 * │                             │
 * │  1. 获取事件目标的 Fiber    │
 * │  2. 收集沿途的事件处理函数   │
 * │  3. 创建合成事件对象         │
 * │  4. 按顺序执行处理函数       │
 * └─────────────────────────────┘
 * ```
 */

// 简化版事件分发
function dispatchEvent(
  domEventName: string,
  eventSystemFlags: number,
  targetContainer: EventTarget,
  nativeEvent: Event
) {
  // 1. 获取事件目标
  const nativeEventTarget = nativeEvent.target;

  // 2. 获取目标的 Fiber 节点
  const targetFiber = getClosestInstanceFromNode(nativeEventTarget as Node);

  // 3. 收集事件处理函数
  const listeners = collectListeners(targetFiber, domEventName);

  // 4. 创建合成事件
  const syntheticEvent = new SyntheticEvent(nativeEvent);

  // 5. 执行处理函数
  for (const listener of listeners) {
    listener.call(undefined, syntheticEvent);

    // 检查是否停止传播
    if (syntheticEvent.isPropagationStopped()) {
      break;
    }
  }
}

function getClosestInstanceFromNode(node: Node): any {
  // 从 DOM 节点获取 Fiber
  // 实际通过 node[internalInstanceKey] 获取
  return null;
}

function collectListeners(fiber: any, eventName: string): Function[] {
  // 沿着 Fiber 树向上收集事件处理函数
  const listeners: Function[] = [];
  let current = fiber;

  while (current !== null) {
    const props = current.memoizedProps;
    if (props) {
      // onClick -> onClick
      const handler = props[eventName];
      if (handler) {
        listeners.push(handler);
      }
    }
    current = current.return;
  }

  return listeners;
}

// ============================================================
// 4. 事件优先级
// ============================================================

/**
 * 📊 事件优先级
 *
 * 不同事件有不同优先级，影响更新的调度
 *
 * ```
 * 离散事件（DiscreteEvent）- 最高优先级
 * - click, keydown, input
 * - 需要立即响应
 *
 * 连续事件（ContinuousEvent）- 较低优先级
 * - scroll, drag, mousemove
 * - 可以合并处理
 *
 * 默认事件（DefaultEvent）- 正常优先级
 * - 其他事件
 * ```
 */

const EventPriorities = {
  DiscreteEventPriority: 1,    // 离散事件
  ContinuousEventPriority: 4,  // 连续事件
  DefaultEventPriority: 16,    // 默认
  IdleEventPriority: 536870912, // 空闲
};

// 根据事件名获取优先级
function getEventPriority(domEventName: string): number {
  switch (domEventName) {
    case 'click':
    case 'keydown':
    case 'keyup':
    case 'input':
    case 'change':
      return EventPriorities.DiscreteEventPriority;

    case 'scroll':
    case 'drag':
    case 'mousemove':
    case 'touchmove':
      return EventPriorities.ContinuousEventPriority;

    default:
      return EventPriorities.DefaultEventPriority;
  }
}

// ============================================================
// 5. 💡 面试题
// ============================================================

/**
 * 💡 Q1: React 事件和原生事件的区别？
 *
 * A:
 *    1. 命名：onClick vs onclick
 *    2. 绑定：委托到根容器 vs 绑定到元素
 *    3. 事件对象：SyntheticEvent vs Event
 *    4. 执行顺序：原生先于 React
 *
 * 💡 Q2: React 为什么要用事件委托？
 *
 * A:
 *    1. 减少事件监听器数量，节省内存
 *    2. 动态添加的元素自动有事件处理
 *    3. 统一管理，便于实现优先级调度
 *
 * 💡 Q3: React 17 事件系统有什么变化？
 *
 * A:
 *    1. 事件委托从 document 改到根容器
 *    2. 支持多个 React 版本共存
 *    3. 事件池被移除（不需要 e.persist()）
 *
 * 💡 Q4: React 事件和原生事件的执行顺序？
 *
 * A:
 *    1. 原生捕获阶段
 *    2. 目标元素的原生事件
 *    3. 原生冒泡阶段
 *    4. React 事件（在根容器处理）
 *
 *    注意：React 事件在冒泡到根容器后才执行
 */

// ============================================================
// 6. 🏢 实际开发应用
// ============================================================

/**
 * 🏢 应用 1：阻止事件冒泡
 *
 * ```jsx
 * function Modal({ onClose }) {
 *   return (
 *     <div className="overlay" onClick={onClose}>
 *       <div className="modal" onClick={e => e.stopPropagation()}>
 *         内容
 *       </div>
 *     </div>
 *   );
 * }
 * ```
 */

/**
 * 🏢 应用 2：混合使用原生事件
 *
 * ```jsx
 * function Component() {
 *   const ref = useRef();
 *
 *   useEffect(() => {
 *     const el = ref.current;
 *     const handler = (e) => {
 *       console.log('Native event');
 *     };
 *
 *     // 原生事件先执行
 *     el.addEventListener('click', handler);
 *     return () => el.removeEventListener('click', handler);
 *   }, []);
 *
 *   return (
 *     <div ref={ref} onClick={() => console.log('React event')}>
 *       Click
 *     </div>
 *   );
 *   // 输出顺序：Native event -> React event
 * }
 * ```
 */

/**
 * 🏢 应用 3：事件代理优化
 *
 * 理解事件委托后，就知道不需要给每个列表项绑定事件
 *
 * ```jsx
 * // 不推荐
 * {items.map(item => (
 *   <li onClick={() => handleClick(item.id)}>{item.name}</li>
 * ))}
 *
 * // 推荐（利用事件委托）
 * <ul onClick={e => {
 *   const id = e.target.dataset.id;
 *   if (id) handleClick(id);
 * }}>
 *   {items.map(item => (
 *     <li data-id={item.id}>{item.name}</li>
 *   ))}
 * </ul>
 * ```
 */

// ============================================================
// 7. 📖 源码阅读指南
// ============================================================

/**
 * 📖 阅读顺序：
 *
 * 1. packages/react-dom/src/events/DOMPluginEventSystem.js
 *    - listenToAllSupportedEvents（注册事件）
 *    - dispatchEvent（分发事件）
 *
 * 2. packages/react-dom/src/events/SyntheticEvent.js
 *    - SyntheticEvent 类定义
 *
 * 3. packages/react-dom/src/events/ReactDOMEventListener.js
 *    - createEventListenerWrapper
 *    - dispatchEvent 入口
 *
 * 4. packages/react-dom/src/events/getEventPriority.js
 *    - 事件优先级定义
 */

// ============================================================
// 8. ✅ 学习检查
// ============================================================

/**
 * ✅ 完成以下任务：
 *
 * - [ ] 理解 React 事件与原生事件的区别
 * - [ ] 理解事件委托机制
 * - [ ] 理解合成事件
 * - [ ] 理解事件优先级
 * - [ ] 能解释事件执行顺序
 */

export {
  SyntheticEvent,
  EventPriorities,
  dispatchEvent,
  getEventPriority,
};

