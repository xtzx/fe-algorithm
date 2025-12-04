/**
 * ============================================================
 * 📚 Diff 算法
 * ============================================================
 *
 * 面试考察重点：
 * 1. Diff 算法的核心思想
 * 2. React 和 Vue 的 Diff 区别
 * 3. key 的作用
 * 4. 双端对比算法
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 什么是 Diff 算法？
 *
 * Diff 算法用于比较新旧虚拟 DOM 树，找出最小变更集合。
 *
 * 📊 传统 Diff vs 前端框架 Diff
 *
 * 传统 Diff（完全对比）：
 * - 时间复杂度 O(n³)
 * - 不实用
 *
 * 前端框架 Diff（启发式算法）：
 * - 时间复杂度 O(n)
 * - 三个假设：
 *   1. 不同类型的元素会产生不同的树
 *   2. 开发者可以通过 key 暗示哪些子元素在不同渲染中保持稳定
 *   3. 只做同层比较，不跨层级
 */

/**
 * 📊 Diff 策略
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │                         三层 Diff 策略                              │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │                                                                     │
 * │   Tree Diff          Component Diff        Element Diff            │
 * │   ──────────         ──────────────        ────────────            │
 * │   同层比较            组件类型比较          同类型元素比较          │
 * │   不跨层级            类型不同直接替换      更新属性和子节点        │
 * │                      类型相同递归 Diff                              │
 * │                                                                     │
 * └─────────────────────────────────────────────────────────────────────┘
 */

// ============================================================
// 2. React Diff 算法
// ============================================================

/**
 * 📊 React Diff 特点（单向遍历）
 *
 * React 使用从左到右的单向遍历：
 *
 * 旧：A B C D E
 * 新：A C D B E
 *
 * 过程：
 * 1. A 位置不变，复用
 * 2. C 需要移动（原位置 2 < lastIndex 1，需要移动）
 * 3. D 需要移动
 * 4. B 需要移动
 * 5. E 位置不变，复用
 *
 * 特点：
 * - 只会把节点往右移动
 * - 从左到右遍历新节点
 * - 记录已处理节点的最大索引 lastIndex
 * - 如果旧节点索引 < lastIndex，需要移动
 */

interface VNode {
  type: string;
  key?: string | number;
  props: Record<string, any>;
  children: (VNode | string)[];
  el?: HTMLElement;
}

// React 风格的 Diff（简化版）
function reactDiff(
  parentEl: HTMLElement,
  oldChildren: VNode[],
  newChildren: VNode[]
) {
  const oldKeyMap = new Map<string | number, { node: VNode; index: number }>();
  
  // 建立旧节点的 key -> index 映射
  oldChildren.forEach((child, index) => {
    if (child.key != null) {
      oldKeyMap.set(child.key, { node: child, index });
    }
  });

  let lastIndex = 0;

  newChildren.forEach((newChild, newIndex) => {
    const key = newChild.key;
    const old = key != null ? oldKeyMap.get(key) : null;

    if (old) {
      // 找到可复用节点
      patch(old.node, newChild);
      
      if (old.index < lastIndex) {
        // 需要移动：将节点移动到前一个新节点之后
        const prevNode = newChildren[newIndex - 1];
        if (prevNode) {
          const anchor = prevNode.el?.nextSibling;
          parentEl.insertBefore(newChild.el!, anchor || null);
        }
      } else {
        // 不需要移动，更新 lastIndex
        lastIndex = old.index;
      }
      
      oldKeyMap.delete(key!);
    } else {
      // 没找到，创建新节点
      const prevNode = newChildren[newIndex - 1];
      const anchor = prevNode ? prevNode.el?.nextSibling : parentEl.firstChild;
      mount(newChild, parentEl, anchor as HTMLElement);
    }
  });

  // 删除剩余的旧节点
  oldKeyMap.forEach(({ node }) => {
    unmount(node);
  });
}

// 辅助函数
function patch(oldNode: VNode, newNode: VNode) {
  newNode.el = oldNode.el;
  // 更新属性和子节点...
}

function mount(node: VNode, parent: HTMLElement, anchor?: HTMLElement) {
  const el = document.createElement(node.type);
  node.el = el;
  parent.insertBefore(el, anchor || null);
}

function unmount(node: VNode) {
  node.el?.remove();
}

// ============================================================
// 3. Vue 双端 Diff 算法
// ============================================================

/**
 * 📊 Vue2 双端 Diff
 *
 * 使用四个指针，从两端向中间收缩：
 *
 * oldStartIdx → ... ← oldEndIdx
 * newStartIdx → ... ← newEndIdx
 *
 * 比较顺序：
 * 1. oldStart vs newStart（头头比较）
 * 2. oldEnd vs newEnd（尾尾比较）
 * 3. oldStart vs newEnd（头尾比较）
 * 4. oldEnd vs newStart（尾头比较）
 * 5. 都不匹配，遍历查找
 *
 * 优势：
 * - 能处理更多的常见移动场景
 * - 减少不必要的移动操作
 */

function vue2Diff(
  parentEl: HTMLElement,
  oldChildren: VNode[],
  newChildren: VNode[]
) {
  let oldStartIdx = 0;
  let oldEndIdx = oldChildren.length - 1;
  let newStartIdx = 0;
  let newEndIdx = newChildren.length - 1;

  let oldStartNode = oldChildren[oldStartIdx];
  let oldEndNode = oldChildren[oldEndIdx];
  let newStartNode = newChildren[newStartIdx];
  let newEndNode = newChildren[newEndIdx];

  // 建立 key -> index 映射（用于第 5 种情况）
  const oldKeyMap = new Map<string | number, number>();
  oldChildren.forEach((child, index) => {
    if (child.key != null) {
      oldKeyMap.set(child.key, index);
    }
  });

  while (oldStartIdx <= oldEndIdx && newStartIdx <= newEndIdx) {
    // 跳过已处理的节点
    if (!oldStartNode) {
      oldStartNode = oldChildren[++oldStartIdx];
    } else if (!oldEndNode) {
      oldEndNode = oldChildren[--oldEndIdx];
    }
    // 1. 头头比较
    else if (isSameNode(oldStartNode, newStartNode)) {
      patch(oldStartNode, newStartNode);
      oldStartNode = oldChildren[++oldStartIdx];
      newStartNode = newChildren[++newStartIdx];
    }
    // 2. 尾尾比较
    else if (isSameNode(oldEndNode, newEndNode)) {
      patch(oldEndNode, newEndNode);
      oldEndNode = oldChildren[--oldEndIdx];
      newEndNode = newChildren[--newEndIdx];
    }
    // 3. 头尾比较（旧头 vs 新尾）
    else if (isSameNode(oldStartNode, newEndNode)) {
      patch(oldStartNode, newEndNode);
      // 移动到最后
      parentEl.insertBefore(oldStartNode.el!, oldEndNode.el!.nextSibling);
      oldStartNode = oldChildren[++oldStartIdx];
      newEndNode = newChildren[--newEndIdx];
    }
    // 4. 尾头比较（旧尾 vs 新头）
    else if (isSameNode(oldEndNode, newStartNode)) {
      patch(oldEndNode, newStartNode);
      // 移动到最前
      parentEl.insertBefore(oldEndNode.el!, oldStartNode.el!);
      oldEndNode = oldChildren[--oldEndIdx];
      newStartNode = newChildren[++newStartIdx];
    }
    // 5. 都不匹配，查找
    else {
      const idxInOld = newStartNode.key != null
        ? oldKeyMap.get(newStartNode.key)
        : findIdxInOld(newStartNode, oldChildren, oldStartIdx, oldEndIdx);

      if (idxInOld === undefined) {
        // 新节点，创建
        mount(newStartNode, parentEl, oldStartNode.el);
      } else {
        // 找到可复用节点
        const nodeToMove = oldChildren[idxInOld];
        patch(nodeToMove, newStartNode);
        parentEl.insertBefore(nodeToMove.el!, oldStartNode.el!);
        // 标记已处理
        (oldChildren as any)[idxInOld] = undefined;
      }
      newStartNode = newChildren[++newStartIdx];
    }
  }

  // 处理剩余节点
  if (oldStartIdx > oldEndIdx) {
    // 新节点还有剩余，添加
    const anchor = newChildren[newEndIdx + 1]?.el || null;
    for (let i = newStartIdx; i <= newEndIdx; i++) {
      mount(newChildren[i], parentEl, anchor as HTMLElement);
    }
  } else if (newStartIdx > newEndIdx) {
    // 旧节点还有剩余，删除
    for (let i = oldStartIdx; i <= oldEndIdx; i++) {
      if (oldChildren[i]) {
        unmount(oldChildren[i]);
      }
    }
  }
}

function isSameNode(a: VNode, b: VNode): boolean {
  return a.type === b.type && a.key === b.key;
}

function findIdxInOld(
  node: VNode,
  oldChildren: VNode[],
  start: number,
  end: number
): number | undefined {
  for (let i = start; i <= end; i++) {
    if (oldChildren[i] && isSameNode(oldChildren[i], node)) {
      return i;
    }
  }
  return undefined;
}

// ============================================================
// 4. Vue3 快速 Diff 算法
// ============================================================

/**
 * 📊 Vue3 快速 Diff
 *
 * 结合了双端对比和最长递增子序列：
 *
 * 1. 预处理：处理头部和尾部相同的节点
 * 2. 中间部分使用最长递增子序列（LIS）减少移动次数
 *
 * 优势：
 * - 预处理减少比较范围
 * - LIS 保证最少移动次数
 */

/**
 * 📊 最长递增子序列（LIS）
 *
 * 给定数组 [3, 1, 4, 1, 5, 9, 2, 6]
 * LIS = [1, 4, 5, 9] 或 [1, 4, 5, 6]
 *
 * 在 Diff 中的作用：
 * - 找出不需要移动的节点序列
 * - 只移动不在 LIS 中的节点
 */

// 最长递增子序列算法
function getSequence(arr: number[]): number[] {
  const p = arr.slice(); // 存储前驱索引
  const result = [0];    // 存储 LIS 的索引
  let i, j, u, v, c;
  const len = arr.length;

  for (i = 0; i < len; i++) {
    const arrI = arr[i];
    if (arrI !== 0) {
      j = result[result.length - 1];
      if (arr[j] < arrI) {
        // 当前值大于 result 最后一个，直接 push
        p[i] = j;
        result.push(i);
        continue;
      }
      // 二分查找找到第一个大于 arrI 的位置
      u = 0;
      v = result.length - 1;
      while (u < v) {
        c = (u + v) >> 1;
        if (arr[result[c]] < arrI) {
          u = c + 1;
        } else {
          v = c;
        }
      }
      if (arrI < arr[result[u]]) {
        if (u > 0) {
          p[i] = result[u - 1];
        }
        result[u] = i;
      }
    }
  }

  // 回溯构建最终结果
  u = result.length;
  v = result[u - 1];
  while (u-- > 0) {
    result[u] = v;
    v = p[v];
  }

  return result;
}

// ============================================================
// 5. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. 用 index 作为 key
 *    - ❌ 列表变化时 index 会变
 *    - ❌ 导致错误的节点复用
 *    - ✅ 用稳定的唯一标识（id）
 *
 * 2. 用随机数作为 key
 *    - ❌ 每次渲染都变化
 *    - ❌ 完全失去复用能力
 *    - ✅ key 应该是稳定的
 *
 * 3. 兄弟节点 key 重复
 *    - ❌ Diff 算法依赖 key 唯一性
 *    - ❌ 可能导致渲染错误
 *
 * 4. 忽略 key 的重要性
 *    - ❌ 认为 key 只是消除警告
 *    - ✅ key 直接影响性能和正确性
 */

// key 问题示例
const keyProblemExample = `
// ❌ 错误：用 index 作为 key
{items.map((item, index) => (
  <Item key={index} data={item} />
))}

// 问题场景：删除第一项
// 旧：[A, B, C] -> key: [0, 1, 2]
// 新：[B, C]    -> key: [0, 1]
// Diff 认为：0 变成 B，1 变成 C，删除 2
// 实际上：应该删除 A，复用 B 和 C

// ✅ 正确：用唯一 id 作为 key
{items.map(item => (
  <Item key={item.id} data={item} />
))}
`;

// ============================================================
// 6. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 为什么 Diff 算法是 O(n) 而不是 O(n³)？
 * A:
 *    - 传统算法需要比较所有节点对，然后找最小编辑距离
 *    - 前端框架基于三个假设简化：
 *      · 只做同层比较
 *      · 不同类型直接替换
 *      · key 标识稳定节点
 *    - 只需要遍历一次新节点
 *
 * Q2: React 和 Vue 的 Diff 有什么区别？
 * A:
 *    React：
 *    - 单向从左到右遍历
 *    - 只会把节点往右移
 *    - 头部插入性能差
 *
 *    Vue2：
 *    - 双端对比
 *    - 四个指针从两端向中间收缩
 *    - 能处理更多移动场景
 *
 *    Vue3：
 *    - 快速 Diff
 *    - 预处理 + 最长递增子序列
 *    - 移动次数最少
 *
 * Q3: 为什么 Vue3 要用最长递增子序列？
 * A:
 *    - LIS 中的节点不需要移动
 *    - 只移动不在 LIS 中的节点
 *    - 保证移动次数最少
 *
 * Q4: Diff 算法的时间复杂度是多少？
 * A:
 *    - 遍历：O(n)
 *    - Vue3 的 LIS：O(n log n)
 *    - 总体：O(n) 或 O(n log n)
 */

// ============================================================
// 7. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：列表头部插入性能差（React）
 *
 * 问题：
 * - 在列表头部插入元素
 * - React 的单向 Diff 会移动所有元素
 *
 * 旧：[B, C, D]
 * 新：[A, B, C, D]
 *
 * React 处理：
 * - B 位置 0，lastIndex = 0
 * - C 位置 1，lastIndex = 1
 * - D 位置 2，lastIndex = 2
 * - 所有节点都被认为需要移动！
 *
 * 解决：
 * - 使用唯一稳定的 key
 * - 考虑使用 Vue（双端 Diff 处理更好）
 */

/**
 * 🏢 场景 2：列表项状态丢失
 *
 * 问题：
 * - 用 index 作为 key
 * - 删除某项后，其他项的状态错乱
 *
 * 原因：
 * - index 变化导致 key 变化
 * - Diff 错误复用节点
 * - 状态和 DOM 对应错误
 *
 * 解决：
 * - 使用稳定的 id 作为 key
 */

/**
 * 🏢 场景 3：大列表 Diff 性能问题
 *
 * 问题：
 * - 1000+ 节点的列表
 * - Diff 耗时明显
 *
 * 解决：
 * - 虚拟滚动（只 Diff 可见部分）
 * - 分页加载
 * - 使用 key 确保正确复用
 */

export {
  reactDiff,
  vue2Diff,
  getSequence,
  isSameNode,
  keyProblemExample,
};

export type { VNode };

