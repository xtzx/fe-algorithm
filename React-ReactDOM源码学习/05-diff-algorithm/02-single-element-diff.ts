/**
 * ============================================================
 * 📚 Phase 5: Diff 算法 - Part 2: 单节点 Diff
 * ============================================================
 *
 * 📁 源码位置:
 * - ReactChildFiber.new.js 第 1129-1205 行: reconcileSingleElement
 * - ReactChildFiber.new.js 第 1207-1249 行: reconcileSingleTextNode
 *
 * 单节点 Diff 指新的子元素只有一个（不是数组）
 */

// ============================================================
// Part 1: 单元素 Diff 流程
// ============================================================

/**
 * 📊 reconcileSingleElement 流程图
 *
 * 当新的 children 是单个 React Element 时调用
 */

const singleElementDiffFlow = `
📊 单元素 Diff 流程

reconcileSingleElement(returnFiber, currentFirstChild, element, lanes)
    │
    ├── 获取新元素的 key
    │   const key = element.key;
    │
    ├── 遍历旧的子 Fiber 链表
    │   let child = currentFirstChild;
    │   while (child !== null) {
    │       │
    │       ├── key 相同？
    │       │   │
    │       │   ├── YES → type 也相同？
    │       │   │         │
    │       │   │         ├── YES → ⭐ 复用！
    │       │   │         │         deleteRemainingChildren() // 删除其他兄弟
    │       │   │         │         return useFiber(child, props)
    │       │   │         │
    │       │   │         └── NO → 删除当前及所有兄弟，跳出循环
    │       │   │                  deleteRemainingChildren(child)
    │       │   │                  break
    │       │   │
    │       │   └── NO → 只删除当前，继续遍历
    │       │            deleteChild(child)
    │       │
    │       └── child = child.sibling (继续遍历)
    │
    └── 没找到可复用的 → 创建新 Fiber
        createFiberFromElement(element)
`;

// ============================================================
// Part 2: 真实案例分析
// ============================================================

/**
 * 📊 案例1: key 和 type 都相同 → 复用
 */

const case1_sameKeyAndType = `
📊 案例1: key 和 type 都相同

// 更新前
<div key="a" className="old">Hello</div>

// 更新后
<div key="a" className="new">World</div>

旧 Fiber 链表: [div(key=a)]
新 Element: div(key=a)

Diff 过程:
1. key 相同 ('a' === 'a') ✓
2. type 相同 ('div' === 'div') ✓
3. 复用 Fiber，更新 props

结果:
- 不创建新 DOM
- 只更新 className 和 children
- 标记 Update flag
`;

/**
 * 📊 案例2: key 相同但 type 不同 → 不复用
 */

const case2_sameKeyDiffType = `
📊 案例2: key 相同但 type 不同

// 更新前
<div key="a">Hello</div>

// 更新后
<span key="a">Hello</span>

旧 Fiber 链表: [div(key=a)]
新 Element: span(key=a)

Diff 过程:
1. key 相同 ('a' === 'a') ✓
2. type 不同 ('div' !== 'span') ✗
3. 删除当前及所有兄弟节点
4. 创建新的 span Fiber

结果:
- 删除 div DOM
- 创建 span DOM
- 即使 key 相同，type 不同也不会复用！
`;

/**
 * 📊 案例3: key 不同 → 继续寻找
 */

const case3_diffKey = `
📊 案例3: key 不同

// 更新前
<div key="a">A</div>
<div key="b">B</div>
<div key="c">C</div>

// 更新后（单元素）
<div key="b">B new</div>

旧 Fiber 链表: [div(a) → div(b) → div(c)]
新 Element: div(key=b)

Diff 过程:
1. 第一个节点 key='a' !== 'b' → deleteChild(a)
2. 第二个节点 key='b' === 'b' ✓
   type='div' === 'div' ✓ → 复用！
3. deleteRemainingChildren() → 删除 c

结果:
- 删除 a 和 c
- 复用 b，更新 props
`;

/**
 * 📊 案例4: 没有匹配的 key
 */

const case4_noMatch = `
📊 案例4: 没有匹配的 key

// 更新前
<div key="a">A</div>
<div key="b">B</div>

// 更新后
<div key="c">C</div>

旧 Fiber 链表: [div(a) → div(b)]
新 Element: div(key=c)

Diff 过程:
1. key='a' !== 'c' → deleteChild(a)
2. key='b' !== 'c' → deleteChild(b)
3. 遍历完毕，没找到可复用的
4. createFiberFromElement() → 创建新 Fiber

结果:
- 删除 a 和 b
- 创建新的 c
`;

// ============================================================
// Part 3: 源码简化实现
// ============================================================

/**
 * 📁 源码位置: ReactChildFiber.new.js 第 1129-1205 行
 */

interface Fiber {
  key: string | null;
  tag: number;
  elementType: any;
  type: any;
  stateNode: any;
  return: Fiber | null;
  sibling: Fiber | null;
  child: Fiber | null;
  index: number;
  ref: any;
  flags: number;
  deletions: Fiber[] | null;
  alternate: Fiber | null;
}

interface ReactElement {
  $$typeof: symbol;
  type: any;
  key: string | null;
  ref: any;
  props: any;
}

type Lanes = number;
const Fragment = 7;
const REACT_FRAGMENT_TYPE = Symbol.for('react.fragment');
const Placement = 0b00000000000000000000000010;
const ChildDeletion = 0b00000000000000000000010000;

// 简化版 reconcileSingleElement
function reconcileSingleElement(
  returnFiber: Fiber,
  currentFirstChild: Fiber | null,
  element: ReactElement,
  lanes: Lanes
): Fiber {
  const key = element.key;
  let child = currentFirstChild;

  // 遍历旧的子 Fiber
  while (child !== null) {
    // 1. 比较 key
    if (child.key === key) {
      const elementType = element.type;

      // 2. 比较 type
      if (elementType === REACT_FRAGMENT_TYPE) {
        // Fragment 特殊处理
        if (child.tag === Fragment) {
          deleteRemainingChildren(returnFiber, child.sibling);
          const existing = useFiber(child, element.props.children);
          existing.return = returnFiber;
          return existing;
        }
      } else {
        // 普通元素
        if (child.elementType === elementType) {
          // ⭐ key 和 type 都相同，可以复用！
          deleteRemainingChildren(returnFiber, child.sibling);
          const existing = useFiber(child, element.props);
          existing.ref = element.ref;
          existing.return = returnFiber;
          return existing;
        }
      }

      // key 相同但 type 不同，删除所有旧节点
      deleteRemainingChildren(returnFiber, child);
      break;
    } else {
      // key 不同，只删除当前节点，继续遍历
      deleteChild(returnFiber, child);
    }

    child = child.sibling;
  }

  // 没找到可复用的，创建新 Fiber
  if (element.type === REACT_FRAGMENT_TYPE) {
    const created = createFiberFromFragment(
      element.props.children,
      returnFiber.mode,
      lanes,
      element.key
    );
    created.return = returnFiber;
    return created;
  } else {
    const created = createFiberFromElement(element, returnFiber.mode, lanes);
    created.ref = element.ref;
    created.return = returnFiber;
    return created;
  }
}

// 复用 Fiber
function useFiber(fiber: Fiber, pendingProps: any): Fiber {
  // 基于旧 Fiber 创建 workInProgress
  const clone = createWorkInProgress(fiber, pendingProps);
  clone.index = 0;
  clone.sibling = null;
  return clone;
}

// 删除单个子节点
function deleteChild(returnFiber: Fiber, childToDelete: Fiber): void {
  // 标记父节点有子节点需要删除
  const deletions = returnFiber.deletions;
  if (deletions === null) {
    returnFiber.deletions = [childToDelete];
    returnFiber.flags |= ChildDeletion;
  } else {
    deletions.push(childToDelete);
  }
}

// 删除剩余所有子节点
function deleteRemainingChildren(
  returnFiber: Fiber,
  currentFirstChild: Fiber | null
): void {
  let childToDelete = currentFirstChild;
  while (childToDelete !== null) {
    deleteChild(returnFiber, childToDelete);
    childToDelete = childToDelete.sibling;
  }
}

// 辅助函数声明
declare function createWorkInProgress(fiber: Fiber, pendingProps: any): Fiber;
declare function createFiberFromElement(element: ReactElement, mode: number, lanes: Lanes): Fiber;
declare function createFiberFromFragment(elements: any, mode: number, lanes: Lanes, key: string | null): Fiber;

// ============================================================
// Part 4: 单文本节点 Diff
// ============================================================

/**
 * 📁 源码位置: ReactChildFiber.new.js 第 1207-1249 行
 *
 * 当新的 children 是字符串或数字时
 */

const singleTextDiff = `
📊 单文本节点 Diff

// 更新前
<div>
  <span>A</span>
</div>

// 更新后
<div>
  Hello World
</div>

旧 Fiber 链表: [span]
新内容: "Hello World"（文本）

Diff 过程:
1. 检查第一个旧节点是否是文本节点
   child.tag === HostText?

2. 如果是文本节点 → 复用，只更新内容
3. 如果不是文本节点 → 删除所有旧节点，创建新文本节点

源码简化:
function reconcileSingleTextNode(returnFiber, currentFirstChild, textContent) {
  if (currentFirstChild !== null && currentFirstChild.tag === HostText) {
    // 复用文本节点
    deleteRemainingChildren(returnFiber, currentFirstChild.sibling);
    const existing = useFiber(currentFirstChild, textContent);
    existing.return = returnFiber;
    return existing;
  }

  // 删除旧节点，创建新文本节点
  deleteRemainingChildren(returnFiber, currentFirstChild);
  const created = createFiberFromText(textContent, returnFiber.mode, lanes);
  created.return = returnFiber;
  return created;
}
`;

// ============================================================
// Part 5: 面试题
// ============================================================

const interviewQuestions = `
💡 Q1: 单节点 Diff 的判断顺序是什么？
A: 先比较 key，再比较 type
   1. key 不同 → 删除当前节点，继续遍历兄弟
   2. key 相同，type 不同 → 删除当前及所有兄弟
   3. key 相同，type 相同 → 复用

💡 Q2: 为什么 key 相同但 type 不同时要删除所有兄弟？
A: 因为 key 是唯一标识，如果 key 相同说明就是这个元素。
   type 不同说明元素已经改变，不可能在其他兄弟中找到匹配的。
   所以直接删除所有剩余节点。

💡 Q3: 为什么 key 不同只删除当前节点？
A: 因为 key 不同说明这不是我们要找的元素。
   但是我们要找的元素可能在后面的兄弟节点中。
   所以只删除当前，继续遍历。
`;

export {
  singleElementDiffFlow,
  case1_sameKeyAndType,
  case2_sameKeyDiffType,
  case3_diffKey,
  case4_noMatch,
  singleTextDiff,
  interviewQuestions,
  reconcileSingleElement,
  useFiber,
  deleteChild,
  deleteRemainingChildren,
};

