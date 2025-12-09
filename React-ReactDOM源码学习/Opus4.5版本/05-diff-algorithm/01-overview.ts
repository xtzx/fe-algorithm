/**
 * ============================================================
 * 📚 Phase 5: Diff 算法 - Part 1: 概述与核心思想
 * ============================================================
 *
 * 🎯 学习目标：
 * 1. 理解 Diff 算法的设计思想
 * 2. 掌握 React Diff 的三个限制
 * 3. 理解 key 的作用
 *
 * 📁 核心源码位置：
 * - packages/react-reconciler/src/ReactChildFiber.new.js
 *
 * ⏱️ 预计时间：2-3 小时
 * 🎯 面试权重：⭐⭐⭐⭐⭐
 */

// ============================================================
// Part 1: 为什么需要 Diff 算法
// ============================================================

/**
 * 📊 传统 Diff 的问题
 *
 * 如果要完整比较两棵树的差异，时间复杂度是 O(n³)
 * - 找到两棵树对应节点需要 O(n²)
 * - 编辑操作需要 O(n)
 * - 总计：O(n³)
 *
 * 对于 1000 个节点的树，需要 10 亿次比较！
 */

const traditionalDiffProblem = `
📊 传统 Diff 算法复杂度

假设有 1000 个节点:
- 传统 Diff: O(n³) = 10^9 次操作 → 几秒钟
- React Diff: O(n) = 10^3 次操作 → 几毫秒

React 如何做到 O(n)？
通过三个策略，牺牲一些通用性换取性能！
`;

// ============================================================
// Part 2: React Diff 的三个策略（限制）
// ============================================================

/**
 * 📊 策略1: 同层比较（tree diff）
 *
 * React 只比较同一层级的节点，不跨层级比较
 */

const treeDiffStrategy = `
📊 策略1: 同层比较

假设 DOM 结构从 A 变成 B：

A:          B:
  1           1
 / \\         / \\
2   3       2   4
   / \\           \\
  4   5           3
                 / \\
                5   6

传统 Diff: 尝试找到最优移动路径（如移动节点 3）
React Diff: 只在同层比较
  - 层级1: 1 vs 1 ✓
  - 层级2: [2,3] vs [2,4] → 删除 3，新增 4
  - 层级3: 4 被删除，3 的子树 [4,5] 全部删除

React 的策略：
  如果节点跨层级移动 → 删除旧节点 + 创建新节点
  不会尝试复用跨层级的节点

为什么这样设计？
  - 跨层级移动在实际开发中很少见
  - 这种策略大大简化了算法
`;

/**
 * 📊 策略2: 类型比较（component diff）
 *
 * 不同类型的元素产生不同的树
 */

const componentDiffStrategy = `
📊 策略2: 类型比较

规则：
  - 类型不同 → 直接替换整个子树
  - 类型相同 → 继续比较属性和子节点

示例1: 标签类型变化
  <div>              <span>
    <Counter />  →     <Counter />
  </div>             </span>

  结果：
  - 销毁 <div> 和其子树（包括 Counter 实例）
  - 创建 <span> 和新的 Counter 实例
  - Counter 的 state 会丢失！

示例2: 组件类型变化
  <Counter />  →  <Timer />

  结果：
  - 销毁 Counter 实例
  - 创建 Timer 实例
  - 即使它们渲染相似的 DOM，也不会复用

为什么这样设计？
  - 不同类型的组件通常生成不同的 DOM 结构
  - 深度比较不同类型组件的代价太大
`;

/**
 * 📊 策略3: key 标识（element diff）
 *
 * 开发者可以通过 key 提示哪些元素是稳定的
 */

const elementDiffStrategy = `
📊 策略3: key 标识

没有 key 时的比较（按索引）:
  旧: [A, B, C]
  新: [B, C, A]

  比较过程:
    index 0: A vs B → 更新 A 为 B
    index 1: B vs C → 更新 B 为 C
    index 2: C vs A → 更新 C 为 A

  结果：3 次更新操作！

有 key 时的比较:
  旧: [A(key=a), B(key=b), C(key=c)]
  新: [B(key=b), C(key=c), A(key=a)]

  比较过程:
    通过 key 找到对应关系:
    - B 还在 → 保持/移动
    - C 还在 → 保持/移动
    - A 还在 → 保持/移动

  结果：只需要移动 A，不需要更新内容！

⚠️ key 的注意事项:
  1. key 应该稳定、唯一、可预测
  2. 不要用 index 作为 key（除非列表不会重排）
  3. 不要用随机数作为 key（每次渲染都变）
`;

// ============================================================
// Part 3: Diff 发生的位置
// ============================================================

/**
 * 📁 源码位置: packages/react-reconciler/src/ReactChildFiber.new.js
 *
 * reconcileChildFibers 是 Diff 的入口
 */

const diffEntryPoint = `
📊 Diff 入口函数

// beginWork 中调用
function reconcileChildren(current, workInProgress, nextChildren, renderLanes) {
  if (current === null) {
    // 首次渲染，不需要 Diff
    workInProgress.child = mountChildFibers(
      workInProgress,
      null,
      nextChildren,
      renderLanes,
    );
  } else {
    // 更新渲染，需要 Diff
    workInProgress.child = reconcileChildFibers(
      workInProgress,
      current.child,     // 旧的子 Fiber
      nextChildren,      // 新的子元素
      renderLanes,
    );
  }
}

// reconcileChildFibers 内部根据 newChild 类型分发
function reconcileChildFibers(returnFiber, currentFirstChild, newChild, lanes) {
  // 处理 Fragment
  if (typeof newChild === 'object' && newChild !== null) {
    switch (newChild.$$typeof) {
      case REACT_ELEMENT_TYPE:
        return reconcileSingleElement(...);  // 单个元素
      case REACT_PORTAL_TYPE:
        return reconcileSinglePortal(...);
    }

    if (isArray(newChild)) {
      return reconcileChildrenArray(...);    // 多个子元素（核心！）
    }
  }

  if (typeof newChild === 'string' || typeof newChild === 'number') {
    return reconcileSingleTextNode(...);     // 文本节点
  }

  return deleteRemainingChildren(...);       // 其他情况删除所有
}
`;

// ============================================================
// Part 4: 核心数据结构
// ============================================================

/**
 * 📊 Diff 相关的核心数据结构
 */

// Fiber 节点中与 Diff 相关的属性
interface FiberDiffProps {
  /**
   * 唯一标识，用于快速查找
   * 来源: React Element 的 key 属性
   */
  key: string | null;

  /**
   * 在兄弟节点中的索引位置
   * 用于判断节点是否需要移动
   */
  index: number;

  /**
   * 元素类型（用于类型比较）
   * - 字符串: 'div', 'span'
   * - 函数/类: Component
   * - Symbol: Fragment
   */
  type: any;

  /**
   * 元素类型（包括 key）
   * 用于判断是否可以复用
   */
  elementType: any;

  /**
   * 副作用标记
   * - Placement: 需要插入
   * - ChildDeletion: 需要删除子节点
   */
  flags: number;

  /**
   * 要删除的子 Fiber 数组
   */
  deletions: Array<Fiber> | null;

  /**
   * 指向另一棵树的对应节点
   * 如果存在，说明可以复用
   */
  alternate: Fiber | null;

  /**
   * 链表结构
   */
  child: Fiber | null;    // 第一个子节点
  sibling: Fiber | null;  // 下一个兄弟节点
  return: Fiber | null;   // 父节点
}

// Diff 算法中使用的 Map（用于快速查找）
type ExistingChildrenMap = Map<string | number, Fiber>;

// 副作用标记
const Placement = 0b00000000000000000000000010;      // 需要插入
const ChildDeletion = 0b00000000000000000000010000;  // 需要删除子节点

export {
  traditionalDiffProblem,
  treeDiffStrategy,
  componentDiffStrategy,
  elementDiffStrategy,
  diffEntryPoint,
  Placement,
  ChildDeletion,
};

