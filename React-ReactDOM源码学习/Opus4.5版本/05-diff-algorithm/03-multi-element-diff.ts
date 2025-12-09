/**
 * ============================================================
 * 📚 Phase 5: Diff 算法 - Part 3: 多节点 Diff（核心！）
 * ============================================================
 *
 * 📁 源码位置:
 * - ReactChildFiber.new.js 第 736-901 行: reconcileChildrenArray
 *
 * 多节点 Diff 是面试重点！React 使用两轮遍历来处理
 */

// ============================================================
// Part 1: 多节点 Diff 概述
// ============================================================

/**
 * 📊 多节点更新的场景分类
 */

const multiNodeScenarios = `
📊 多节点更新的四种场景

1. 节点更新（最常见）
   旧: [A, B, C]
   新: [A', B', C']  // 只是 props 变了

2. 节点新增
   旧: [A, B]
   新: [A, B, C, D]

3. 节点删除
   旧: [A, B, C, D]
   新: [A, B]

4. 节点移动
   旧: [A, B, C]
   新: [C, A, B]

React 的设计假设:
  场景 1（更新）在实际开发中最常见
  所以 React 的算法针对场景 1 进行了优化
`;

/**
 * 📊 两轮遍历策略
 */

const twoRoundStrategy = `
📊 React 多节点 Diff 的两轮遍历

┌─────────────────────────────────────────────────────────────────────────┐
│                          第一轮遍历                                     │
│                     （处理更新的节点）                                   │
│                                                                         │
│   从左到右同时遍历新旧数组                                               │
│   比较 key 是否相同                                                     │
│                                                                         │
│   ├── key 相同 → 可能复用，继续比较 type                                │
│   │            └── type 相同 → 复用，继续下一个                         │
│   │            └── type 不同 → 标记删除旧，创建新，继续下一个            │
│   │                                                                     │
│   └── key 不同 → 停止第一轮遍历，进入第二轮                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          第二轮遍历                                     │
│                   （处理非更新的情况）                                   │
│                                                                         │
│   判断第一轮遍历的结束状态：                                             │
│                                                                         │
│   情况1: newIdx === newChildren.length                                  │
│         新数组遍历完了 → 删除剩余旧节点                                  │
│                                                                         │
│   情况2: oldFiber === null                                              │
│         旧数组遍历完了 → 新增剩余新节点                                  │
│                                                                         │
│   情况3: 都没遍历完                                                     │
│         有节点移动 → 使用 Map 优化查找                                  │
│         将剩余旧节点放入 Map<key, Fiber>                                │
│         遍历剩余新节点，从 Map 中查找可复用的                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 2: 第一轮遍历详解
// ============================================================

/**
 * 📁 源码位置: ReactChildFiber.new.js 第 777-820 行
 */

const firstRoundCode = `
📊 第一轮遍历源码分析

// 初始化变量
let resultingFirstChild: Fiber | null = null;  // 新 Fiber 链表头
let previousNewFiber: Fiber | null = null;     // 上一个新 Fiber
let oldFiber = currentFirstChild;              // 当前旧 Fiber
let lastPlacedIndex = 0;                       // 最后一个不需要移动的节点位置
let newIdx = 0;                                // 新数组遍历索引
let nextOldFiber = null;                       // 下一个旧 Fiber

// 第一轮遍历
for (; oldFiber !== null && newIdx < newChildren.length; newIdx++) {

  // 处理旧节点索引大于新节点索引的情况
  if (oldFiber.index > newIdx) {
    nextOldFiber = oldFiber;
    oldFiber = null;
  } else {
    nextOldFiber = oldFiber.sibling;
  }

  // ⭐ 尝试复用：比较 key
  const newFiber = updateSlot(
    returnFiber,
    oldFiber,
    newChildren[newIdx],
    lanes,
  );

  // key 不同，updateSlot 返回 null，跳出第一轮遍历
  if (newFiber === null) {
    if (oldFiber === null) {
      oldFiber = nextOldFiber;
    }
    break;  // ⭐ 关键：key 不同就跳出
  }

  // 处理复用情况...
  lastPlacedIndex = placeChild(newFiber, lastPlacedIndex, newIdx);

  // 构建新 Fiber 链表
  if (previousNewFiber === null) {
    resultingFirstChild = newFiber;
  } else {
    previousNewFiber.sibling = newFiber;
  }
  previousNewFiber = newFiber;
  oldFiber = nextOldFiber;
}
`;

/**
 * 📊 updateSlot 函数
 */

const updateSlotExplanation = `
📊 updateSlot 函数解析

function updateSlot(returnFiber, oldFiber, newChild, lanes) {
  const key = oldFiber !== null ? oldFiber.key : null;

  // 新节点是文本
  if (typeof newChild === 'string' || typeof newChild === 'number') {
    // 旧节点有 key，说明不是文本，不能复用
    if (key !== null) {
      return null;  // key 不匹配，返回 null
    }
    // 旧节点也是文本，尝试复用
    return updateTextNode(returnFiber, oldFiber, '' + newChild, lanes);
  }

  // 新节点是对象（ReactElement）
  if (typeof newChild === 'object' && newChild !== null) {
    if (newChild.key === key) {
      // key 相同，尝试复用
      return updateElement(returnFiber, oldFiber, newChild, lanes);
    } else {
      // key 不同，返回 null
      return null;
    }
  }

  return null;
}

关键点：
- updateSlot 只比较 key
- key 不同返回 null，触发跳出第一轮遍历
- key 相同才继续比较 type
`;

// ============================================================
// Part 3: 第二轮遍历详解
// ============================================================

/**
 * 📁 源码位置: ReactChildFiber.new.js 第 822-900 行
 */

const secondRoundCode = `
📊 第二轮遍历源码分析

// 情况1: 新数组遍历完了
if (newIdx === newChildren.length) {
  // 删除剩余旧节点
  deleteRemainingChildren(returnFiber, oldFiber);
  return resultingFirstChild;
}

// 情况2: 旧数组遍历完了
if (oldFiber === null) {
  // 新增剩余新节点
  for (; newIdx < newChildren.length; newIdx++) {
    const newFiber = createChild(returnFiber, newChildren[newIdx], lanes);
    if (newFiber === null) continue;

    // 标记为需要插入
    lastPlacedIndex = placeChild(newFiber, lastPlacedIndex, newIdx);

    // 构建链表
    if (previousNewFiber === null) {
      resultingFirstChild = newFiber;
    } else {
      previousNewFiber.sibling = newFiber;
    }
    previousNewFiber = newFiber;
  }
  return resultingFirstChild;
}

// 情况3: 都没遍历完（有移动）
// 将剩余旧节点放入 Map
const existingChildren = mapRemainingChildren(returnFiber, oldFiber);

// 遍历剩余新节点
for (; newIdx < newChildren.length; newIdx++) {
  // 从 Map 中查找可复用的
  const newFiber = updateFromMap(
    existingChildren,
    returnFiber,
    newIdx,
    newChildren[newIdx],
    lanes,
  );

  if (newFiber !== null) {
    if (newFiber.alternate !== null) {
      // 复用了，从 Map 中删除
      existingChildren.delete(newFiber.key ?? newIdx);
    }
    lastPlacedIndex = placeChild(newFiber, lastPlacedIndex, newIdx);
    // 构建链表...
  }
}

// 删除 Map 中剩余的（未被复用的）
existingChildren.forEach(child => deleteChild(returnFiber, child));
`;

/**
 * 📊 mapRemainingChildren 函数
 */

const mapRemainingChildrenExplanation = `
📊 mapRemainingChildren 函数

function mapRemainingChildren(returnFiber, currentFirstChild) {
  const existingChildren = new Map();

  let existingChild = currentFirstChild;
  while (existingChild !== null) {
    if (existingChild.key !== null) {
      // 有 key，用 key 作为键
      existingChildren.set(existingChild.key, existingChild);
    } else {
      // 没有 key，用 index 作为键
      existingChildren.set(existingChild.index, existingChild);
    }
    existingChild = existingChild.sibling;
  }

  return existingChildren;
}

Map 结构示例:
旧节点: [A(key=a), B(key=b), C(key=c)]

Map:
{
  'a' => FiberA,
  'b' => FiberB,
  'c' => FiberC
}
`;

// ============================================================
// Part 4: placeChild - 判断是否需要移动
// ============================================================

/**
 * 📁 源码位置: ReactChildFiber.new.js 第 329-357 行
 *
 * 这是理解移动判断的关键！
 */

const placeChildExplanation = `
📊 placeChild 函数 - 移动判断的核心

function placeChild(newFiber, lastPlacedIndex, newIndex) {
  newFiber.index = newIndex;  // 更新索引

  const current = newFiber.alternate;

  if (current !== null) {
    // 复用的节点，判断是否需要移动
    const oldIndex = current.index;  // 旧位置

    if (oldIndex < lastPlacedIndex) {
      // ⭐ 旧位置 < 最后放置位置 → 需要移动
      newFiber.flags |= Placement;
      return lastPlacedIndex;  // lastPlacedIndex 不变
    } else {
      // 不需要移动
      return oldIndex;  // 更新 lastPlacedIndex
    }
  } else {
    // 新创建的节点，需要插入
    newFiber.flags |= Placement;
    return lastPlacedIndex;
  }
}

核心逻辑：
- lastPlacedIndex 记录「最后一个不需要移动的节点」在旧列表中的位置
- 如果当前节点的旧位置 < lastPlacedIndex，说明它相对位置向右移动了
- 向右移动需要标记 Placement
`;

// ============================================================
// Part 5: 真实案例详解
// ============================================================

/**
 * 📊 案例1: 节点更新（最常见）
 */

const case1_update = `
📊 案例1: 节点更新

旧: [A(key=a), B(key=b), C(key=c)]
新: [A'(key=a), B'(key=b), C'(key=c)]

第一轮遍历:
┌─────────────────────────────────────────────────────────────┐
│ newIdx │ oldFiber │ 操作                                    │
├────────┼──────────┼─────────────────────────────────────────┤
│   0    │    A     │ key='a'='a' ✓, 复用 A，更新为 A'        │
│   1    │    B     │ key='b'='b' ✓, 复用 B，更新为 B'        │
│   2    │    C     │ key='c'='c' ✓, 复用 C，更新为 C'        │
└────────┴──────────┴─────────────────────────────────────────┘

结果: newIdx === 3，新数组遍历完毕
进入情况1: 旧数组也遍历完了，直接返回

最终: 3 个节点都复用，只更新 props
DOM 操作: 0 次移动，只更新属性
`;

/**
 * 📊 案例2: 节点删除
 */

const case2_delete = `
📊 案例2: 节点删除

旧: [A(key=a), B(key=b), C(key=c), D(key=d)]
新: [A(key=a), C(key=c)]

第一轮遍历:
┌─────────────────────────────────────────────────────────────┐
│ newIdx │ oldFiber │ 操作                                    │
├────────┼──────────┼─────────────────────────────────────────┤
│   0    │    A     │ key='a'='a' ✓, 复用 A                   │
│   1    │    B     │ key='b'!='c' ✗, 跳出循环！              │
└────────┴──────────┴─────────────────────────────────────────┘

第一轮结束: newIdx=1, oldFiber=B

进入情况3: 都没遍历完
- 剩余旧节点: [B, C, D] → 放入 Map
- Map: { 'b'=>B, 'c'=>C, 'd'=>D }

第二轮遍历:
┌─────────────────────────────────────────────────────────────┐
│ newIdx │ 新节点 │ 从 Map 查找                                │
├────────┼────────┼────────────────────────────────────────────┤
│   1    │   C    │ Map.get('c') = C, 复用 C, Map 删除 'c'     │
└────────┴────────┴────────────────────────────────────────────┘

遍历结束，Map 剩余: { 'b'=>B, 'd'=>D }
删除 B 和 D

最终: 复用 A、C，删除 B、D
DOM 操作: 删除 2 个节点
`;

/**
 * 📊 案例3: 节点新增
 */

const case3_insert = `
📊 案例3: 节点新增

旧: [A(key=a), B(key=b)]
新: [A(key=a), B(key=b), C(key=c), D(key=d)]

第一轮遍历:
┌─────────────────────────────────────────────────────────────┐
│ newIdx │ oldFiber │ 操作                                    │
├────────┼──────────┼─────────────────────────────────────────┤
│   0    │    A     │ key='a'='a' ✓, 复用 A                   │
│   1    │    B     │ key='b'='b' ✓, 复用 B                   │
│   2    │   null   │ oldFiber 为空，跳出循环                  │
└────────┴──────────┴─────────────────────────────────────────┘

第一轮结束: newIdx=2, oldFiber=null

进入情况2: 旧数组遍历完了
- 新增剩余新节点

┌─────────────────────────────────────────────────────────────┐
│ newIdx │ 操作                                               │
├────────┼────────────────────────────────────────────────────┤
│   2    │ createChild(C), 标记 Placement                     │
│   3    │ createChild(D), 标记 Placement                     │
└────────┴────────────────────────────────────────────────────┘

最终: 复用 A、B，新增 C、D
DOM 操作: 插入 2 个新节点
`;

/**
 * 📊 案例4: 节点移动（复杂！）
 */

const case4_move = `
📊 案例4: 节点移动

旧: [A(key=a), B(key=b), C(key=c), D(key=d)]
      index: 0       1       2       3

新: [A(key=a), C(key=c), D(key=d), B(key=b)]
      index: 0       1       2       3

第一轮遍历:
┌─────────────────────────────────────────────────────────────┐
│ newIdx │ oldFiber │ 操作                                    │
├────────┼──────────┼─────────────────────────────────────────┤
│   0    │    A     │ key='a'='a' ✓, 复用 A                   │
│   1    │    B     │ key='b'!='c' ✗, 跳出循环！              │
└────────┴──────────┴─────────────────────────────────────────┘

第一轮结束: newIdx=1, oldFiber=B, lastPlacedIndex=0

进入情况3: 都没遍历完
- 剩余旧节点: [B, C, D] → 放入 Map
- Map: { 'b'=>B(index=1), 'c'=>C(index=2), 'd'=>D(index=3) }

第二轮遍历 + placeChild 判断:
┌───────────────────────────────────────────────────────────────────────────┐
│ newIdx │ 新节点 │ Map查找 │ oldIndex │ lastPlacedIndex │ 移动？          │
├────────┼────────┼─────────┼──────────┼─────────────────┼──────────────────┤
│   1    │   C    │ C(2)    │    2     │ 0 → 2           │ 2>0, 不移动     │
│   2    │   D    │ D(3)    │    3     │ 2 → 3           │ 3>2, 不移动     │
│   3    │   B    │ B(1)    │    1     │ 3               │ 1<3, 需要移动！ │
└────────┴────────┴─────────┴──────────┴─────────────────┴──────────────────┘

分析:
- C: oldIndex(2) >= lastPlacedIndex(0), 不移动, lastPlacedIndex=2
- D: oldIndex(3) >= lastPlacedIndex(2), 不移动, lastPlacedIndex=3
- B: oldIndex(1) < lastPlacedIndex(3), 需要移动！

最终: A、C、D 保持不动，B 移动到最后
DOM 操作: 只移动 B 一个节点
`;

/**
 * 📊 案例5: 移动优化分析
 */

const case5_moveOptimization = `
📊 案例5: 为什么只移动 B？

旧: [A, B, C, D]  (索引: 0, 1, 2, 3)
新: [A, C, D, B]

直觉方案:
  移动 C 到位置 1
  移动 D 到位置 2
  → 2 次移动

React 方案:
  A、C、D 保持相对顺序不变
  只移动 B 到末尾
  → 1 次移动！

原理:
  lastPlacedIndex 记录「基准位置」
  - A 在位置 0，lastPlacedIndex = 0
  - C 原位置 2 > 0，不需要移动，lastPlacedIndex = 2
  - D 原位置 3 > 2，不需要移动，lastPlacedIndex = 3
  - B 原位置 1 < 3，需要移动

  相当于：找到一个「递增子序列」，其他元素移动
`;

// ============================================================
// Part 6: 源码简化实现
// ============================================================

interface Fiber {
  key: string | null;
  index: number;
  flags: number;
  alternate: Fiber | null;
  sibling: Fiber | null;
  return: Fiber | null;
}

const Placement = 0b00000000000000000000000010;

// 简化版 reconcileChildrenArray
function reconcileChildrenArraySimplified(
  returnFiber: Fiber,
  currentFirstChild: Fiber | null,
  newChildren: any[],
  lanes: number
): Fiber | null {
  let resultingFirstChild: Fiber | null = null;
  let previousNewFiber: Fiber | null = null;
  let oldFiber = currentFirstChild;
  let lastPlacedIndex = 0;
  let newIdx = 0;
  let nextOldFiber: Fiber | null = null;

  // ========== 第一轮遍历 ==========
  for (; oldFiber !== null && newIdx < newChildren.length; newIdx++) {
    if (oldFiber.index > newIdx) {
      nextOldFiber = oldFiber;
      oldFiber = null;
    } else {
      nextOldFiber = oldFiber.sibling;
    }

    // 尝试复用（比较 key）
    const newFiber = updateSlot(returnFiber, oldFiber, newChildren[newIdx], lanes);

    if (newFiber === null) {
      // key 不同，跳出第一轮
      if (oldFiber === null) {
        oldFiber = nextOldFiber;
      }
      break;
    }

    // 处理移动
    lastPlacedIndex = placeChild(newFiber, lastPlacedIndex, newIdx);

    // 构建链表
    if (previousNewFiber === null) {
      resultingFirstChild = newFiber;
    } else {
      previousNewFiber.sibling = newFiber;
    }
    previousNewFiber = newFiber;
    oldFiber = nextOldFiber;
  }

  // ========== 情况1: 新数组遍历完 ==========
  if (newIdx === newChildren.length) {
    deleteRemainingChildren(returnFiber, oldFiber);
    return resultingFirstChild;
  }

  // ========== 情况2: 旧数组遍历完 ==========
  if (oldFiber === null) {
    for (; newIdx < newChildren.length; newIdx++) {
      const newFiber = createChild(returnFiber, newChildren[newIdx], lanes);
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

  // ========== 情况3: 都没遍历完（有移动） ==========
  const existingChildren = mapRemainingChildren(returnFiber, oldFiber);

  for (; newIdx < newChildren.length; newIdx++) {
    const newFiber = updateFromMap(
      existingChildren,
      returnFiber,
      newIdx,
      newChildren[newIdx],
      lanes
    );

    if (newFiber !== null) {
      if (newFiber.alternate !== null) {
        existingChildren.delete(newFiber.key ?? newIdx);
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

  // 删除未复用的
  existingChildren.forEach(child => deleteChild(returnFiber, child));

  return resultingFirstChild;
}

// placeChild 简化实现
function placeChild(
  newFiber: Fiber,
  lastPlacedIndex: number,
  newIndex: number
): number {
  newFiber.index = newIndex;

  const current = newFiber.alternate;
  if (current !== null) {
    const oldIndex = current.index;
    if (oldIndex < lastPlacedIndex) {
      // 需要移动
      newFiber.flags |= Placement;
      return lastPlacedIndex;
    } else {
      // 不需要移动
      return oldIndex;
    }
  } else {
    // 新节点
    newFiber.flags |= Placement;
    return lastPlacedIndex;
  }
}

// 辅助函数声明
declare function updateSlot(returnFiber: Fiber, oldFiber: Fiber | null, newChild: any, lanes: number): Fiber | null;
declare function createChild(returnFiber: Fiber, newChild: any, lanes: number): Fiber | null;
declare function updateFromMap(existingChildren: Map<string | number, Fiber>, returnFiber: Fiber, newIdx: number, newChild: any, lanes: number): Fiber | null;
declare function mapRemainingChildren(returnFiber: Fiber, currentFirstChild: Fiber): Map<string | number, Fiber>;
declare function deleteRemainingChildren(returnFiber: Fiber, currentFirstChild: Fiber | null): void;
declare function deleteChild(returnFiber: Fiber, child: Fiber): void;

export {
  multiNodeScenarios,
  twoRoundStrategy,
  firstRoundCode,
  updateSlotExplanation,
  secondRoundCode,
  mapRemainingChildrenExplanation,
  placeChildExplanation,
  case1_update,
  case2_delete,
  case3_insert,
  case4_move,
  case5_moveOptimization,
  reconcileChildrenArraySimplified,
  placeChild,
};

