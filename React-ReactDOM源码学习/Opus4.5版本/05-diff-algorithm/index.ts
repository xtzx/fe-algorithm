/**
 * ============================================================
 * 📚 Phase 5: Diff 算法（核心重点）
 * ============================================================
 *
 * 🎯 学习目标：
 * 1. 理解 React Diff 的三个限制
 * 2. 掌握单节点 Diff
 * 3. 掌握多节点 Diff
 * 4. 理解 key 的作用
 *
 * 📁 源码位置：
 * - packages/react-reconciler/src/ReactChildFiber.js
 *
 * ⏱️ 预计时间：6 小时
 * 🔥 面试权重：⭐⭐⭐⭐⭐（必考）
 */

// ============================================================
// 1. Diff 算法概述
// ============================================================

/**
 * 📊 为什么需要 Diff？
 *
 * 完全对比两棵树的复杂度是 O(n³)
 * 1000 个节点需要 10 亿次操作
 *
 * React 通过三个限制将复杂度降为 O(n)：
 *
 * 1. 只比较同层节点
 *    不会跨层级移动节点
 *
 * 2. 不同类型的节点产生不同的树
 *    div 变成 span，直接删除重建
 *
 * 3. 通过 key 标识哪些节点可以复用
 *    key 相同才尝试复用
 */

/**
 * 📊 Diff 流程
 *
 * ```
 *                  Diff 入口
 *                      │
 *          ┌───────────┴───────────┐
 *          │                       │
 *      单节点 Diff            多节点 Diff
 *   (newChild 不是数组)     (newChild 是数组)
 *          │                       │
 *          │                       │
 *  ┌───────┴───────┐      ┌───────┴───────┐
 *  │               │      │               │
 * 复用/创建       删除    第一轮遍历    第二轮遍历
 *                         (处理更新)    (处理移动)
 * ```
 */

// ============================================================
// 2. 单节点 Diff
// ============================================================

/**
 * 📊 单节点 Diff 流程
 *
 * ```
 * 新节点是单个节点时：
 *
 *         ┌─────────────┐
 *         │ 遍历旧子节点 │
 *         └──────┬──────┘
 *                │
 *         ┌──────▼──────┐
 *     ┌───│  key 相同？  │───┐
 *     │   └─────────────┘   │
 *     │是                   │否
 *     │                     │
 * ┌───▼───────┐      ┌──────▼──────┐
 * │ type 相同？ │      │  标记删除   │
 * └─────┬─────┘      │  继续遍历   │
 *       │            └─────────────┘
 *   ┌───┴───┐
 *   │是     │否
 *   │       │
 * ┌─▼─┐  ┌──▼───┐
 * │复用│  │删除旧│
 * │节点│  │创建新│
 * └───┘  └──────┘
 * ```
 */

interface SimpleFiber {
  key: string | null;
  type: any;
  child: SimpleFiber | null;
  sibling: SimpleFiber | null;
  return: SimpleFiber | null;
  alternate: SimpleFiber | null;
  flags: number;
}

// 简化版单节点 Diff
function reconcileSingleElement(
  returnFiber: SimpleFiber,
  currentFirstChild: SimpleFiber | null,
  element: { type: any; key: string | null; props: any }
): SimpleFiber {
  const key = element.key;
  let child = currentFirstChild;

  // 遍历旧子节点
  while (child !== null) {
    if (child.key === key) {
      // key 相同
      if (child.type === element.type) {
        // type 也相同，可以复用
        // 删除其他兄弟节点
        deleteRemainingChildren(returnFiber, child.sibling);
        // 复用当前节点
        const existing = useFiber(child, element.props);
        existing.return = returnFiber;
        return existing;
      } else {
        // type 不同，删除所有旧节点
        deleteRemainingChildren(returnFiber, child);
        break;
      }
    } else {
      // key 不同，删除当前节点
      deleteChild(returnFiber, child);
    }
    child = child.sibling;
  }

  // 创建新节点
  const created = createFiberFromElement(element);
  created.return = returnFiber;
  return created;
}

// ============================================================
// 3. 多节点 Diff
// ============================================================

/**
 * 📊 多节点 Diff 的两轮遍历
 *
 * 第一轮：处理更新（key 和 type 都相同）
 * 第二轮：处理新增、删除、移动
 *
 * 设计原因：更新是最常见的操作，优先处理
 */

/**
 * 📊 第一轮遍历
 *
 * ```
 * 旧: A → B → C → D
 * 新: A → B → E → F
 *
 * 第一轮：从左到右遍历
 *
 * i=0: A vs A → key 相同，type 相同 → 复用
 * i=1: B vs B → key 相同，type 相同 → 复用
 * i=2: C vs E → key 不同 → 跳出第一轮
 *
 * 结果：复用 A、B
 * ```
 */

/**
 * 📊 第二轮遍历
 *
 * 情况 1：新节点遍历完，旧节点还有
 * ```
 * 旧: A → B → C → D
 * 新: A → B
 *
 * → 删除 C、D
 * ```
 *
 * 情况 2：旧节点遍历完，新节点还有
 * ```
 * 旧: A → B
 * 新: A → B → C → D
 *
 * → 新建 C、D
 * ```
 *
 * 情况 3：都没遍历完（移动）
 * ```
 * 旧: A → B → C → D
 * 新: A → C → D → B
 *
 * → 使用 Map 优化查找
 * → 通过 lastPlacedIndex 判断是否需要移动
 * ```
 */

// 简化版多节点 Diff
function reconcileChildrenArray(
  returnFiber: SimpleFiber,
  currentFirstChild: SimpleFiber | null,
  newChildren: any[]
): SimpleFiber | null {
  let resultingFirstChild: SimpleFiber | null = null;
  let previousNewFiber: SimpleFiber | null = null;

  let oldFiber = currentFirstChild;
  let newIdx = 0;
  let lastPlacedIndex = 0;

  // ========== 第一轮遍历 ==========
  for (; oldFiber !== null && newIdx < newChildren.length; newIdx++) {
    const newChild = newChildren[newIdx];

    if (oldFiber.key !== newChild.key) {
      // key 不同，跳出第一轮
      break;
    }

    const newFiber = updateElement(returnFiber, oldFiber, newChild);
    if (newFiber === null) break;

    // 判断是否需要移动
    lastPlacedIndex = placeChild(newFiber, lastPlacedIndex, newIdx);

    // 构建链表
    if (previousNewFiber === null) {
      resultingFirstChild = newFiber;
    } else {
      previousNewFiber.sibling = newFiber;
    }
    previousNewFiber = newFiber;

    oldFiber = oldFiber.sibling;
  }

  // ========== 第二轮遍历 ==========

  // 情况 1：新节点遍历完
  if (newIdx === newChildren.length) {
    deleteRemainingChildren(returnFiber, oldFiber);
    return resultingFirstChild;
  }

  // 情况 2：旧节点遍历完
  if (oldFiber === null) {
    for (; newIdx < newChildren.length; newIdx++) {
      const newFiber = createChild(returnFiber, newChildren[newIdx]);
      if (newFiber === null) continue;

      lastPlacedIndex = placeChild(newFiber, lastPlacedIndex, newIdx);

      if (previousNewFiber === null) {
        resultingFirstChild = newFiber;
      } else {
        previousNewFiber.sibling = newFiber;
      }
      previousNewFiber = newFiber;
    }
    return resultingFirstChild;
  }

  // 情况 3：都没遍历完（移动）
  // 将剩余旧节点放入 Map
  const existingChildren = mapRemainingChildren(oldFiber);

  for (; newIdx < newChildren.length; newIdx++) {
    const newFiber = updateFromMap(
      existingChildren,
      returnFiber,
      newIdx,
      newChildren[newIdx]
    );

    if (newFiber !== null) {
      // 从 Map 中删除已使用的节点
      if (newFiber.alternate !== null) {
        existingChildren.delete(
          newFiber.key === null ? newIdx : newFiber.key
        );
      }

      lastPlacedIndex = placeChild(newFiber, lastPlacedIndex, newIdx);

      if (previousNewFiber === null) {
        resultingFirstChild = newFiber;
      } else {
        previousNewFiber.sibling = newFiber;
      }
      previousNewFiber = newFiber;
    }
  }

  // 删除未使用的旧节点
  existingChildren.forEach(child => deleteChild(returnFiber, child));

  return resultingFirstChild;
}

// ============================================================
// 4. 移动判断 - lastPlacedIndex
// ============================================================

/**
 * 📊 移动判断算法
 *
 * 关键变量：lastPlacedIndex（最后一个不需要移动的节点索引）
 *
 * 规则：
 * - 如果 oldIndex >= lastPlacedIndex，不需要移动
 * - 如果 oldIndex < lastPlacedIndex，需要移动
 *
 * 示例：
 * ```
 * 旧: A(0) → B(1) → C(2) → D(3)
 * 新: A → C → D → B
 *
 * 遍历新节点：
 * A: oldIndex=0, lastPlacedIndex=0 → 0>=0 不移动，更新 lastPlacedIndex=0
 * C: oldIndex=2, lastPlacedIndex=0 → 2>=0 不移动，更新 lastPlacedIndex=2
 * D: oldIndex=3, lastPlacedIndex=2 → 3>=2 不移动，更新 lastPlacedIndex=3
 * B: oldIndex=1, lastPlacedIndex=3 → 1<3  需要移动！
 *
 * 结果：只需要移动 B
 * ```
 */

function placeChild(
  newFiber: SimpleFiber,
  lastPlacedIndex: number,
  newIndex: number
): number {
  newFiber.index = newIndex;

  const current = newFiber.alternate;
  if (current !== null) {
    const oldIndex = current.index;
    if (oldIndex < lastPlacedIndex) {
      // 需要移动
      newFiber.flags |= 2; // Placement
      return lastPlacedIndex;
    } else {
      // 不需要移动
      return oldIndex;
    }
  } else {
    // 新节点
    newFiber.flags |= 2; // Placement
    return lastPlacedIndex;
  }
}

// ============================================================
// 5. 💡 面试题
// ============================================================

/**
 * 💡 Q1: React 的 Diff 算法是怎么工作的？
 *
 * A: React Diff 有三个限制：
 *    1. 只比较同层节点
 *    2. 不同类型直接替换
 *    3. 通过 key 标识复用
 *
 *    流程：
 *    - 单节点：遍历旧节点，找 key 和 type 都相同的复用
 *    - 多节点：两轮遍历，先处理更新，再处理移动
 *
 * 💡 Q2: key 的作用是什么？
 *
 * A: key 用于 Diff 时标识节点身份：
 *    - 帮助 React 找到对应的旧节点
 *    - 避免不必要的删除和创建
 *    - 保持组件状态
 *
 *    注意：
 *    - key 必须稳定、唯一
 *    - 不要用 index 作为 key（除非列表不变）
 *
 * 💡 Q3: 为什么不推荐用 index 作为 key？
 *
 * A: 当列表顺序变化时：
 *    - index 会变化，导致 key 不稳定
 *    - React 会错误复用节点
 *    - 可能导致状态错乱
 *
 *    示例：
 *    ```
 *    删除第一项后：
 *    旧: A(key=0) → B(key=1) → C(key=2)
 *    新: B(key=0) → C(key=1)
 *
 *    React 会复用 key=0 的节点（A→B）
 *    实际上 A 被删除了，但状态被保留给了 B
 *    ```
 *
 * 💡 Q4: 什么是 lastPlacedIndex？
 *
 * A: 用于判断节点是否需要移动的变量。
 *    - 记录最后一个不需要移动的旧节点索引
 *    - 如果当前旧节点索引 < lastPlacedIndex，需要移动
 *    - 这是 React 特有的右移优化
 */

// ============================================================
// 6. 辅助函数（简化版）
// ============================================================

function deleteRemainingChildren(
  returnFiber: SimpleFiber,
  currentFirstChild: SimpleFiber | null
) {
  let childToDelete = currentFirstChild;
  while (childToDelete !== null) {
    deleteChild(returnFiber, childToDelete);
    childToDelete = childToDelete.sibling;
  }
}

function deleteChild(returnFiber: SimpleFiber, childToDelete: SimpleFiber) {
  // 标记删除
  childToDelete.flags |= 8; // Deletion
  console.log('Delete child:', childToDelete.key);
}

function useFiber(fiber: SimpleFiber, pendingProps: any): SimpleFiber {
  // 复用 Fiber
  const clone = { ...fiber };
  clone.sibling = null;
  return clone;
}

function createFiberFromElement(element: any): SimpleFiber {
  return {
    key: element.key,
    type: element.type,
    child: null,
    sibling: null,
    return: null,
    alternate: null,
    flags: 0,
  };
}

function updateElement(
  returnFiber: SimpleFiber,
  oldFiber: SimpleFiber,
  newChild: any
): SimpleFiber | null {
  if (oldFiber.type === newChild.type) {
    return useFiber(oldFiber, newChild.props);
  }
  return createFiberFromElement(newChild);
}

function createChild(returnFiber: SimpleFiber, newChild: any): SimpleFiber | null {
  return createFiberFromElement(newChild);
}

function mapRemainingChildren(
  currentFirstChild: SimpleFiber | null
): Map<string | number, SimpleFiber> {
  const existingChildren: Map<string | number, SimpleFiber> = new Map();
  let existingChild = currentFirstChild;
  while (existingChild !== null) {
    if (existingChild.key !== null) {
      existingChildren.set(existingChild.key, existingChild);
    } else {
      existingChildren.set(existingChild.index, existingChild);
    }
    existingChild = existingChild.sibling;
  }
  return existingChildren;
}

function updateFromMap(
  existingChildren: Map<string | number, SimpleFiber>,
  returnFiber: SimpleFiber,
  newIdx: number,
  newChild: any
): SimpleFiber | null {
  const matchedFiber = existingChildren.get(
    newChild.key === null ? newIdx : newChild.key
  );
  if (matchedFiber !== undefined) {
    return updateElement(returnFiber, matchedFiber, newChild);
  }
  return createFiberFromElement(newChild);
}

// ============================================================
// 7. 📖 源码阅读指南
// ============================================================

/**
 * 📖 阅读顺序：
 *
 * 1. packages/react-reconciler/src/ReactChildFiber.js
 *    - reconcileChildFibers（入口）
 *    - reconcileSingleElement（单节点）
 *    - reconcileChildrenArray（多节点）
 *    - placeChild（移动判断）
 *    - mapRemainingChildren（构建 Map）
 */

// ============================================================
// 8. ✅ 学习检查
// ============================================================

/**
 * ✅ 完成以下任务：
 *
 * - [ ] 理解 Diff 的三个限制
 * - [ ] 理解单节点 Diff 流程
 * - [ ] 理解多节点 Diff 的两轮遍历
 * - [ ] 理解 lastPlacedIndex 的作用
 * - [ ] 能解释 key 的重要性
 * - [ ] 阅读源码：ReactChildFiber.js
 */

export {
  reconcileSingleElement,
  reconcileChildrenArray,
  placeChild,
};

