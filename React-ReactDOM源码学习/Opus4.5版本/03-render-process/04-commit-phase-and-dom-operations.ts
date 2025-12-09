/**
 * ============================================================
 * 📚 Phase 3: 渲染流程 - Part 4: Commit 阶段与 DOM 操作
 * ============================================================
 *
 * 📁 核心源码位置:
 * - packages/react-reconciler/src/ReactFiberWorkLoop.new.js
 * - packages/react-reconciler/src/ReactFiberCommitWork.new.js
 * - packages/react-dom/src/client/ReactDOMHostConfig.js
 *
 * ⏱️ 预计时间：2-3 小时
 * 🎯 面试权重：⭐⭐⭐⭐⭐
 */

// ============================================================
// Part 1: Commit 阶段概览
// ============================================================

/**
 * 📊 Commit 阶段的三个子阶段
 */

const commitOverview = `
📊 Commit 阶段概览

源码位置: packages/react-reconciler/src/ReactFiberWorkLoop.new.js
═══════════════════════════════════════════════════════════════════════════════

Commit 阶段是渲染的最后一步，负责将计算结果应用到真实 DOM。

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Commit 阶段                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Render 阶段完成后                                                          │
│         │                                                                   │
│         │  root.finishedWork = workInProgress (新的 Fiber 树)               │
│         ▼                                                                   │
│   ┌─────────────────┐                                                       │
│   │   commitRoot    │  Commit 阶段入口                                      │
│   └────────┬────────┘                                                       │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                      commitRootImpl                              │      │
│   │                                                                  │      │
│   │   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────┐  │      │
│   │   │ Before Mutation │ → │    Mutation     │ → │   Layout    │  │      │
│   │   │    阶段         │   │    阶段         │   │    阶段     │  │      │
│   │   └─────────────────┘   └─────────────────┘   └─────────────┘  │      │
│   │         │                     │                     │          │      │
│   │         ▼                     ▼                     ▼          │      │
│   │   getSnapshot...        执行 DOM 操作         生命周期/Hooks   │      │
│   │                         插入/更新/删除        didMount/Update  │      │
│   │                                               useLayoutEffect  │      │
│   │                                                                  │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│            │                                                                │
│            │  之后异步调度                                                   │
│            ▼                                                                │
│   ┌─────────────────┐                                                       │
│   │ Passive Effects │  useEffect 的执行                                     │
│   │  (异步执行)     │                                                       │
│   └─────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


三个子阶段的职责:
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────┬─────────────────────────────────────────────────────────┐
│ 阶段            │ 职责                                                    │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ Before Mutation │ DOM 操作前                                              │
│                 │ • getSnapshotBeforeUpdate 生命周期                      │
│                 │ • 读取 DOM 状态（如滚动位置）                           │
│                 │ • 调度 useEffect（仅调度，不执行）                      │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ Mutation        │ DOM 变更                                                │
│                 │ • 执行真实的 DOM 操作                                   │
│                 │ • 插入新节点（appendChild）                             │
│                 │ • 更新属性（updateProperties）                          │
│                 │ • 删除节点（removeChild）                               │
│                 │ • 文本更新                                              │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ Layout          │ DOM 操作后                                              │
│                 │ • componentDidMount / componentDidUpdate                │
│                 │ • useLayoutEffect 的 create 函数                        │
│                 │ • 更新 ref                                              │
│                 │ • 此时可以读取新的 DOM 布局                             │
└─────────────────┴─────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 2: commitRoot 入口
// ============================================================

/**
 * 📊 commitRoot 函数
 */

const commitRootDetail = `
📊 commitRoot - Commit 阶段入口

源码位置: packages/react-reconciler/src/ReactFiberWorkLoop.new.js (Line 1963)
═══════════════════════════════════════════════════════════════════════════════

function commitRoot(
  root: FiberRoot,
  recoverableErrors: null | Array<CapturedValue<mixed>>,
  transitions: Array<Transition> | null,
) {
  // 保存和重置优先级
  const previousUpdateLanePriority = getCurrentUpdatePriority();
  const prevTransition = ReactCurrentBatchConfig.transition;

  try {
    ReactCurrentBatchConfig.transition = null;
    // ⭐ 以最高优先级执行 commit
    setCurrentUpdatePriority(DiscreteEventPriority);
    commitRootImpl(
      root,
      recoverableErrors,
      transitions,
      previousUpdateLanePriority,
    );
  } finally {
    ReactCurrentBatchConfig.transition = prevTransition;
    setCurrentUpdatePriority(previousUpdateLanePriority);
  }
}


commitRootImpl 核心逻辑:
═══════════════════════════════════════════════════════════════════════════════

function commitRootImpl(
  root: FiberRoot,
  recoverableErrors: null | Array<CapturedValue<mixed>>,
  transitions: Array<Transition> | null,
  renderPriorityLevel: EventPriority,
) {
  // ─────────────────────────────────────────────────
  // Step 1: 准备阶段
  // ─────────────────────────────────────────────────

  const finishedWork = root.finishedWork;
  const lanes = root.finishedLanes;

  if (finishedWork === null) {
    return null;
  }

  // 清理
  root.finishedWork = null;
  root.finishedLanes = NoLanes;
  root.callbackNode = null;
  root.callbackPriority = NoLane;

  // 计算剩余的工作
  const remainingLanes = mergeLanes(
    finishedWork.lanes,
    finishedWork.childLanes,
  );
  markRootFinished(root, remainingLanes);

  // ─────────────────────────────────────────────────
  // Step 2: 调度 Passive Effects (useEffect)
  // ─────────────────────────────────────────────────

  if (
    (finishedWork.subtreeFlags & PassiveMask) !== NoFlags ||
    (finishedWork.flags & PassiveMask) !== NoFlags
  ) {
    if (!rootDoesHavePassiveEffects) {
      rootDoesHavePassiveEffects = true;
      pendingPassiveEffectsLanes = lanes;
      // ⭐ 使用 Scheduler 异步调度 useEffect
      scheduleCallback(NormalSchedulerPriority, () => {
        flushPassiveEffects();
        return null;
      });
    }
  }

  // ─────────────────────────────────────────────────
  // Step 3: 检查是否有副作用需要处理
  // ─────────────────────────────────────────────────

  const subtreeHasEffects =
    (finishedWork.subtreeFlags &
      (BeforeMutationMask | MutationMask | LayoutMask | PassiveMask)) !==
    NoFlags;
  const rootHasEffect =
    (finishedWork.flags &
      (BeforeMutationMask | MutationMask | LayoutMask | PassiveMask)) !==
    NoFlags;

  if (subtreeHasEffects || rootHasEffect) {
    // ─────────────────────────────────────────────────
    // Step 4: Before Mutation 阶段
    // ─────────────────────────────────────────────────

    const shouldFireAfterActiveInstanceBlur = commitBeforeMutationEffects(
      root,
      finishedWork,
    );

    // ─────────────────────────────────────────────────
    // Step 5: Mutation 阶段（执行 DOM 操作）
    // ─────────────────────────────────────────────────

    commitMutationEffects(root, finishedWork, lanes);

    // ⭐⭐⭐ 关键：在 mutation 后、layout 前切换 Fiber 树
    root.current = finishedWork;

    // ─────────────────────────────────────────────────
    // Step 6: Layout 阶段
    // ─────────────────────────────────────────────────

    commitLayoutEffects(finishedWork, root, lanes);

    // 请求浏览器绘制
    requestPaint();
  } else {
    // 没有副作用，直接切换树
    root.current = finishedWork;
  }

  // ─────────────────────────────────────────────────
  // Step 7: 清理和安排下一次更新
  // ─────────────────────────────────────────────────

  rootDoesHavePassiveEffects = false;
  root.finishedWork = null;
  root.finishedLanes = NoLanes;

  ensureRootIsScheduled(root, now());

  return null;
}
`;

// ============================================================
// Part 3: Before Mutation 阶段
// ============================================================

/**
 * 📊 Before Mutation 阶段
 */

const beforeMutationPhase = `
📊 Before Mutation 阶段

源码位置: packages/react-reconciler/src/ReactFiberCommitWork.new.js
═══════════════════════════════════════════════════════════════════════════════

这是 DOM 操作前的阶段，主要用于：
• 获取 DOM 状态快照（getSnapshotBeforeUpdate）
• 调度 useEffect（仅标记，不执行）


export function commitBeforeMutationEffects(
  root: FiberRoot,
  firstChild: Fiber,
) {
  // 为 getSnapshotBeforeUpdate 做准备
  focusedInstanceHandle = prepareForCommit(root.containerInfo);

  nextEffect = firstChild;
  commitBeforeMutationEffects_begin();

  return shouldFireAfterActiveInstanceBlur;
}

function commitBeforeMutationEffects_begin() {
  while (nextEffect !== null) {
    const fiber = nextEffect;
    const child = fiber.child;

    // 递归处理子树
    if (
      (fiber.subtreeFlags & BeforeMutationMask) !== NoFlags &&
      child !== null
    ) {
      child.return = fiber;
      nextEffect = child;
    } else {
      commitBeforeMutationEffects_complete();
    }
  }
}

function commitBeforeMutationEffects_complete() {
  while (nextEffect !== null) {
    const fiber = nextEffect;

    // ⭐ 处理当前 Fiber 的 Before Mutation 副作用
    commitBeforeMutationEffectsOnFiber(fiber);

    const sibling = fiber.sibling;
    if (sibling !== null) {
      sibling.return = fiber.return;
      nextEffect = sibling;
      return;
    }

    nextEffect = fiber.return;
  }
}


commitBeforeMutationEffectsOnFiber 核心逻辑:
═══════════════════════════════════════════════════════════════════════════════

function commitBeforeMutationEffectsOnFiber(finishedWork: Fiber) {
  const current = finishedWork.alternate;
  const flags = finishedWork.flags;

  // 处理 Snapshot flag（用于 getSnapshotBeforeUpdate）
  if ((flags & Snapshot) !== NoFlags) {
    switch (finishedWork.tag) {
      case ClassComponent: {
        if (current !== null) {
          const prevProps = current.memoizedProps;
          const prevState = current.memoizedState;
          const instance = finishedWork.stateNode;

          // ⭐ 调用 getSnapshotBeforeUpdate
          const snapshot = instance.getSnapshotBeforeUpdate(
            finishedWork.elementType === finishedWork.type
              ? prevProps
              : resolveDefaultProps(finishedWork.type, prevProps),
            prevState,
          );

          // 保存快照，供 componentDidUpdate 使用
          instance.__reactInternalSnapshotBeforeUpdate = snapshot;
        }
        break;
      }
      case HostRoot: {
        // 清空根容器的子节点（为首次渲染准备）
        if (supportsMutation) {
          const container = finishedWork.stateNode.containerInfo;
          clearContainer(container);
        }
        break;
      }
    }
  }
}
`;

// ============================================================
// Part 4: Mutation 阶段
// ============================================================

/**
 * 📊 Mutation 阶段 - 执行 DOM 操作
 */

const mutationPhase = `
📊 Mutation 阶段 - 执行 DOM 操作

源码位置: packages/react-reconciler/src/ReactFiberCommitWork.new.js
═══════════════════════════════════════════════════════════════════════════════

这是真正执行 DOM 操作的阶段！

export function commitMutationEffects(
  root: FiberRoot,
  finishedWork: Fiber,
  committedLanes: Lanes,
) {
  inProgressLanes = committedLanes;
  inProgressRoot = root;

  commitMutationEffectsOnFiber(finishedWork, root, committedLanes);

  inProgressLanes = null;
  inProgressRoot = null;
}

function recursivelyTraverseMutationEffects(
  root: FiberRoot,
  parentFiber: Fiber,
  lanes: Lanes,
) {
  // ⭐ 先处理 deletions
  const deletions = parentFiber.deletions;
  if (deletions !== null) {
    for (let i = 0; i < deletions.length; i++) {
      const childToDelete = deletions[i];
      // 删除节点
      commitDeletionEffects(root, parentFiber, childToDelete);
    }
  }

  // 递归处理子树
  if (parentFiber.subtreeFlags & MutationMask) {
    let child = parentFiber.child;
    while (child !== null) {
      commitMutationEffectsOnFiber(child, root, lanes);
      child = child.sibling;
    }
  }
}


commitMutationEffectsOnFiber 核心逻辑:
═══════════════════════════════════════════════════════════════════════════════

function commitMutationEffectsOnFiber(
  finishedWork: Fiber,
  root: FiberRoot,
  lanes: Lanes,
) {
  const current = finishedWork.alternate;
  const flags = finishedWork.flags;

  switch (finishedWork.tag) {
    case HostComponent: {
      // 先递归处理子树
      recursivelyTraverseMutationEffects(root, finishedWork, lanes);

      // 处理当前节点的 DOM 操作
      commitReconciliationEffects(finishedWork);

      // ⭐ 更新 DOM 属性
      if (flags & Update) {
        const instance = finishedWork.stateNode;
        if (instance != null) {
          const newProps = finishedWork.memoizedProps;
          const oldProps = current !== null ? current.memoizedProps : newProps;
          const type = finishedWork.type;
          const updatePayload = finishedWork.updateQueue;
          finishedWork.updateQueue = null;

          if (updatePayload !== null) {
            // 应用属性变更到 DOM
            commitUpdate(instance, updatePayload, type, oldProps, newProps, ...);
          }
        }
      }
      break;
    }

    case HostText: {
      recursivelyTraverseMutationEffects(root, finishedWork, lanes);
      commitReconciliationEffects(finishedWork);

      // ⭐ 更新文本内容
      if (flags & Update) {
        const textInstance = finishedWork.stateNode;
        const newText = finishedWork.memoizedProps;
        commitTextUpdate(textInstance, oldText, newText);
      }
      break;
    }

    case FunctionComponent:
    case ForwardRef:
    case MemoComponent:
    case SimpleMemoComponent: {
      recursivelyTraverseMutationEffects(root, finishedWork, lanes);
      commitReconciliationEffects(finishedWork);

      // ⭐ 执行 useLayoutEffect / useInsertionEffect 的 destroy
      if (flags & Update) {
        commitHookEffectListUnmount(
          HookInsertion | HookHasEffect,
          finishedWork,
          finishedWork.return,
        );
        commitHookEffectListMount(HookInsertion | HookHasEffect, finishedWork);
      }
      break;
    }
    // ... 其他类型
  }
}


commitReconciliationEffects - 处理 Placement:
═══════════════════════════════════════════════════════════════════════════════

function commitReconciliationEffects(finishedWork: Fiber) {
  const flags = finishedWork.flags;

  // ⭐ Placement：插入新节点
  if (flags & Placement) {
    commitPlacement(finishedWork);
    // 清除 Placement 标记
    finishedWork.flags &= ~Placement;
  }
}

function commitPlacement(finishedWork: Fiber): void {
  // 找到最近的 Host 父节点
  const parentFiber = getHostParentFiber(finishedWork);

  switch (parentFiber.tag) {
    case HostComponent: {
      const parent = parentFiber.stateNode;
      const before = getHostSibling(finishedWork);

      // ⭐ 插入 DOM
      insertOrAppendPlacementNode(finishedWork, before, parent);
      break;
    }
    case HostRoot: {
      const parent = parentFiber.stateNode.containerInfo;
      const before = getHostSibling(finishedWork);
      insertOrAppendPlacementNodeIntoContainer(finishedWork, before, parent);
      break;
    }
  }
}


DOM 操作时序图:
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   commitMutationEffects 开始                                                │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────────────────┐                                               │
│   │ 1. 处理 deletions       │  删除标记的节点                               │
│   │    commitDeletionEffects│  • 调用 componentWillUnmount                  │
│   │                         │  • 解绑 ref                                   │
│   │                         │  • removeChild                                │
│   └──────────┬──────────────┘                                               │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────────┐                                               │
│   │ 2. 递归处理子树          │                                              │
│   └──────────┬──────────────┘                                               │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────────┐                                               │
│   │ 3. 处理 Placement       │  插入新节点                                   │
│   │    commitPlacement      │  • appendChild / insertBefore                 │
│   └──────────┬──────────────┘                                               │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────────┐                                               │
│   │ 4. 处理 Update          │  更新现有节点                                 │
│   │    commitUpdate         │  • 更新属性（className, style...）            │
│   │    commitTextUpdate     │  • 更新文本                                   │
│   └─────────────────────────┘                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 5: Layout 阶段
// ============================================================

/**
 * 📊 Layout 阶段
 */

const layoutPhase = `
📊 Layout 阶段 - DOM 操作后

源码位置: packages/react-reconciler/src/ReactFiberCommitWork.new.js
═══════════════════════════════════════════════════════════════════════════════

此时 DOM 已经更新完成，可以安全地读取新的 DOM 布局。

export function commitLayoutEffects(
  finishedWork: Fiber,
  root: FiberRoot,
  committedLanes: Lanes,
): void {
  inProgressLanes = committedLanes;
  inProgressRoot = root;

  const current = finishedWork.alternate;
  commitLayoutEffectOnFiber(root, current, finishedWork, committedLanes);

  inProgressLanes = null;
  inProgressRoot = null;
}

function commitLayoutEffectOnFiber(
  finishedRoot: FiberRoot,
  current: Fiber | null,
  finishedWork: Fiber,
  committedLanes: Lanes,
): void {
  const flags = finishedWork.flags;

  switch (finishedWork.tag) {
    case FunctionComponent:
    case ForwardRef:
    case SimpleMemoComponent: {
      // 递归处理子树
      recursivelyTraverseLayoutEffects(finishedRoot, finishedWork, committedLanes);

      // ⭐ 执行 useLayoutEffect 的 create 函数
      if (flags & Update) {
        commitHookEffectListMount(HookLayout | HookHasEffect, finishedWork);
      }
      break;
    }

    case ClassComponent: {
      recursivelyTraverseLayoutEffects(finishedRoot, finishedWork, committedLanes);

      if (flags & Update) {
        const instance = finishedWork.stateNode;

        if (current === null) {
          // ⭐ 首次渲染：componentDidMount
          instance.componentDidMount();
        } else {
          // ⭐ 更新：componentDidUpdate
          const prevProps = finishedWork.elementType === finishedWork.type
            ? current.memoizedProps
            : resolveDefaultProps(finishedWork.type, current.memoizedProps);
          const prevState = current.memoizedState;

          instance.componentDidUpdate(
            prevProps,
            prevState,
            instance.__reactInternalSnapshotBeforeUpdate,  // snapshot
          );
        }
      }
      break;
    }

    case HostComponent: {
      recursivelyTraverseLayoutEffects(finishedRoot, finishedWork, committedLanes);

      // ⭐ 处理 ref
      if (flags & Ref) {
        commitAttachRef(finishedWork);
      }
      break;
    }

    // ... 其他类型
  }
}


commitAttachRef - 更新 ref:
═══════════════════════════════════════════════════════════════════════════════

function commitAttachRef(finishedWork: Fiber) {
  const ref = finishedWork.ref;
  if (ref !== null) {
    const instance = finishedWork.stateNode;

    if (typeof ref === 'function') {
      // ⭐ 函数 ref
      ref(instance);
    } else {
      // ⭐ createRef 或 useRef
      ref.current = instance;
    }
  }
}


Layout 阶段时序图:
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   commitLayoutEffects 开始                                                  │
│         │                                                                   │
│         ▼                                                                   │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │ 深度优先遍历 Fiber 树                                              │    │
│   │                                                                    │    │
│   │  FunctionComponent:                                                │    │
│   │    • 执行 useLayoutEffect(() => { ... })                          │    │
│   │    • 此时可以访问更新后的 DOM                                      │    │
│   │                                                                    │    │
│   │  ClassComponent:                                                   │    │
│   │    • 首次：componentDidMount()                                     │    │
│   │    • 更新：componentDidUpdate(prevProps, prevState, snapshot)     │    │
│   │                                                                    │    │
│   │  HostComponent:                                                    │    │
│   │    • 更新 ref（函数 ref 或 ref.current）                          │    │
│   │                                                                    │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 6: Passive Effects（useEffect）
// ============================================================

/**
 * 📊 Passive Effects - useEffect 的执行
 */

const passiveEffects = `
📊 Passive Effects - useEffect 的异步执行

源码位置: packages/react-reconciler/src/ReactFiberWorkLoop.new.js
═══════════════════════════════════════════════════════════════════════════════

useEffect 与 useLayoutEffect 的执行时机不同:
• useLayoutEffect：在 Layout 阶段同步执行
• useEffect：在 commit 完成后异步执行


调度时机:
─────────────────────────────────────────────────────────────────

// 在 commitRootImpl 中
if (
  (finishedWork.subtreeFlags & PassiveMask) !== NoFlags ||
  (finishedWork.flags & PassiveMask) !== NoFlags
) {
  if (!rootDoesHavePassiveEffects) {
    rootDoesHavePassiveEffects = true;

    // ⭐ 使用 Scheduler 异步调度
    scheduleCallback(NormalSchedulerPriority, () => {
      flushPassiveEffects();
      return null;
    });
  }
}


执行时机:
─────────────────────────────────────────────────────────────────

export function flushPassiveEffects(): boolean {
  if (pendingPassiveEffectsLanes !== NoLanes) {
    // ...
    return flushPassiveEffectsImpl();
  }
  return false;
}

function flushPassiveEffectsImpl() {
  // ⭐ 1. 执行所有 effect 的 destroy（上一次的清理）
  commitPassiveUnmountEffects(root.current);

  // ⭐ 2. 执行所有 effect 的 create（本次的副作用）
  commitPassiveMountEffects(root, root.current, lanes, transitions);

  return true;
}


时序图:
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   commitRoot                                                                │
│      │                                                                      │
│      ├── Before Mutation ─────────────────────────────────────────────────  │
│      │                                                                      │
│      ├── Mutation (DOM 操作) ─────────────────────────────────────────────  │
│      │                                                                      │
│      ├── root.current = finishedWork ⭐ 切换 Fiber 树 ────────────────────  │
│      │                                                                      │
│      ├── Layout (useLayoutEffect, componentDidMount) ─────────────────────  │
│      │                                                                      │
│      └── 调度 flushPassiveEffects (异步) ─────────────────────────────────  │
│                │                                                            │
│                │                                                            │
│   ═════════════│═══════════════ commit 结束，控制权交还浏览器 ═════════════ │
│                │                                                            │
│                │  浏览器绘制...                                              │
│                │                                                            │
│                ▼                                                            │
│      ┌─────────────────────────────────────────────────────────┐           │
│      │           flushPassiveEffects                            │           │
│      │                                                          │           │
│      │   1. commitPassiveUnmountEffects                        │           │
│      │      • 执行上一次 useEffect 的 destroy                   │           │
│      │                                                          │           │
│      │   2. commitPassiveMountEffects                          │           │
│      │      • 执行本次 useEffect 的 create                      │           │
│      │                                                          │           │
│      └─────────────────────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


useLayoutEffect vs useEffect 对比:
═══════════════════════════════════════════════════════════════════════════════

┌────────────────────┬──────────────────────┬───────────────────────────────┐
│ 特性               │ useLayoutEffect      │ useEffect                     │
├────────────────────┼──────────────────────┼───────────────────────────────┤
│ 执行时机           │ Layout 阶段（同步）  │ commit 后（异步）             │
│ 执行环境           │ 浏览器绘制前         │ 浏览器绘制后                  │
│ 阻塞绘制           │ ✅ 是               │ ❌ 否                         │
│ 用途               │ 需要同步读取/修改 DOM│ 数据获取、订阅、日志等        │
│ 性能影响           │ 可能导致视觉卡顿     │ 不阻塞视觉更新                │
└────────────────────┴──────────────────────┴───────────────────────────────┘
`;

// ============================================================
// Part 7: 真实案例
// ============================================================

/**
 * 📊 真实案例分析
 */

const realCases = `
📊 真实案例：从渲染到 DOM 更新

案例 A：初次渲染 <App />
═══════════════════════════════════════════════════════════════════════════════

function App() {
  useEffect(() => {
    console.log('useEffect');
  }, []);

  useLayoutEffect(() => {
    console.log('useLayoutEffect');
  }, []);

  return <div>Hello</div>;
}

createRoot(container).render(<App />);


完整流程:
─────────────────────────────────────────────────────────────────

1. createRoot(container)
   │
   └─▶ 创建 FiberRoot + HostRootFiber

2. root.render(<App />)
   │
   └─▶ updateContainer → scheduleUpdateOnFiber → ensureRootIsScheduled

3. Render 阶段
   │
   ├─▶ beginWork(HostRoot) → 创建 App Fiber
   ├─▶ beginWork(App) → 执行 App 函数，处理 Hooks，返回 <div>
   ├─▶ beginWork(div) → null（没有子节点）
   ├─▶ completeWork(div) → 创建 DOM: document.createElement('div')
   ├─▶ completeWork(App) → 冒泡 flags
   └─▶ completeWork(HostRoot) → 完成

4. Commit 阶段
   │
   ├─▶ Before Mutation: （这个例子没有 getSnapshotBeforeUpdate）
   │
   ├─▶ Mutation:
   │   └─▶ commitPlacement → appendChild(div) 到 container
   │
   ├─▶ root.current = finishedWork ⭐ 切换 Fiber 树
   │
   ├─▶ Layout:
   │   └─▶ console.log('useLayoutEffect') ⭐
   │
   └─▶ 调度 flushPassiveEffects

5. 浏览器绘制（用户看到 "Hello"）

6. flushPassiveEffects
   │
   └─▶ console.log('useEffect') ⭐


控制台输出顺序:
─────────────────────────────────────────────────────────────────
useLayoutEffect
useEffect


案例 B：setState 更新
═══════════════════════════════════════════════════════════════════════════════

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <button onClick={() => setCount(count + 1)}>
      Count: {count}
    </button>
  );
}


点击按钮后的流程:
─────────────────────────────────────────────────────────────────

1. 用户点击按钮
   │
   └─▶ 原生 click 事件 → React 事件系统

2. 执行 onClick 回调
   │
   └─▶ setCount(1)
       │
       └─▶ dispatchSetState → scheduleUpdateOnFiber

3. Render 阶段
   │
   ├─▶ beginWork(HostRoot) → bailout（没有更新）
   ├─▶ beginWork(Counter) → 执行 Counter 函数
   │   │                    useState 返回 [1, setCount]
   │   │                    返回新的 <button>...</button>
   │   └─▶ reconcileChildren → Diff 算法比较
   │       │
   │       └─▶ 发现 text 变化：0 → 1
   │           标记 Update flag
   │
   ├─▶ beginWork(button) → ...
   ├─▶ completeWork(text) → 标记需要更新
   ├─▶ completeWork(button) → 冒泡 flags
   └─▶ completeWork(HostRoot)

4. Commit 阶段
   │
   ├─▶ Before Mutation: （无）
   │
   ├─▶ Mutation:
   │   └─▶ commitTextUpdate → textNode.nodeValue = 'Count: 1'
   │
   └─▶ Layout: （无 useLayoutEffect）

5. 浏览器绘制（用户看到 "Count: 1"）
`;

// ============================================================
// Part 8: 面试要点
// ============================================================

const interviewPoints = `
💡 Part 4 面试要点

Q1: Commit 阶段分为哪几个子阶段？
A: 三个子阶段：
   1. Before Mutation：DOM 操作前，getSnapshotBeforeUpdate
   2. Mutation：执行 DOM 操作（插入、更新、删除）
   3. Layout：DOM 操作后，componentDidMount/Update、useLayoutEffect

Q2: root.current 在什么时候切换？
A: 在 Mutation 阶段之后、Layout 阶段之前切换。
   这样保证：
   - Mutation 阶段：current 仍指向旧树（可以获取旧 DOM 状态）
   - Layout 阶段：current 指向新树（生命周期能访问新状态）

Q3: useEffect 和 useLayoutEffect 的执行时机有什么区别？
A: - useLayoutEffect：在 Layout 阶段同步执行，阻塞浏览器绘制
   - useEffect：在 commit 完成后异步执行，不阻塞绘制
   useLayoutEffect 适合需要同步读取/修改 DOM 的场景，
   useEffect 适合副作用（数据获取、订阅等）。

Q4: 为什么 useEffect 要异步执行？
A: 1. 不阻塞浏览器绘制，提升用户体验
   2. 大多数副作用不需要同步执行
   3. 多个 effect 可以批量执行，提升性能
   4. 允许浏览器在 effect 执行前完成布局和绘制

Q5: Mutation 阶段的 DOM 操作顺序是什么？
A: 1. 先处理 deletions（删除节点）
   2. 递归处理子树
   3. 处理 Placement（插入新节点）
   4. 处理 Update（更新属性）

Q6: 为什么要先处理 deletions？
A: 1. 确保被删除的节点不再被引用
   2. 触发 componentWillUnmount 和 useEffect 的 destroy
   3. 解绑 ref
   4. 释放资源，防止内存泄漏

Q7: DOM 是在 Render 阶段还是 Commit 阶段创建的？
A: DOM 元素在 Render 阶段的 completeWork 中创建（document.createElement），
   但此时只是创建和组装 DOM 树。
   真正插入到页面（appendChild/insertBefore）是在 Commit 阶段的 Mutation 子阶段。
`;

export {
  commitOverview,
  commitRootDetail,
  beforeMutationPhase,
  mutationPhase,
  layoutPhase,
  passiveEffects,
  realCases,
  interviewPoints,
};

