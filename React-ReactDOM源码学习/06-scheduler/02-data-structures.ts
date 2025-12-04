/**
 * ============================================================
 * 📚 Phase 6: Scheduler 调度机制 - Part 2: 关键数据结构
 * ============================================================
 *
 * 📁 核心源码位置:
 * - packages/scheduler/src/forks/Scheduler.js (Task 结构)
 * - packages/scheduler/src/SchedulerMinHeap.js (最小堆)
 * - packages/scheduler/src/SchedulerPriorities.js (优先级)
 * - packages/react-reconciler/src/ReactFiberLane.new.js (Lane)
 */

// ============================================================
// Part 1: Task（任务）数据结构
// ============================================================

/**
 * 📊 Task 结构定义
 *
 * 📁 源码位置: packages/scheduler/src/forks/Scheduler.js 第 345-355 行
 */

interface Task {
  /**
   * 任务唯一标识
   * 自增 ID，用于在相同 sortIndex 时决定执行顺序
   * 先创建的任务 ID 小，优先执行
   */
  id: number;

  /**
   * 任务回调函数
   * React 传入的通常是 performConcurrentWorkOnRoot
   * 如果返回函数，说明任务未完成，会继续调度
   * 如果返回 null/undefined，说明任务完成
   */
  callback: ((didTimeout: boolean) => any) | null;

  /**
   * 优先级
   * 1: Immediate, 2: UserBlocking, 3: Normal, 4: Low, 5: Idle
   */
  priorityLevel: number;

  /**
   * 任务开始时间
   * 如果有 delay，startTime = currentTime + delay
   * 否则 startTime = currentTime
   */
  startTime: number;

  /**
   * 过期时间
   * expirationTime = startTime + timeout
   * 不同优先级有不同的 timeout
   */
  expirationTime: number;

  /**
   * 排序索引
   * - 在 taskQueue 中: sortIndex = expirationTime
   * - 在 timerQueue 中: sortIndex = startTime
   * 用于最小堆排序
   */
  sortIndex: number;

  /**
   * DEV 模式下的标记
   * 标识任务是否已入队
   */
  isQueued?: boolean;
}

/**
 * 📊 Task 字段详解表
 */

const taskFieldsTable = `
┌────────────────┬──────────────────┬──────────────────────────────────────────────┐
│ 字段名          │ 类型             │ 含义与作用                                    │
├────────────────┼──────────────────┼──────────────────────────────────────────────┤
│ id             │ number           │ 自增唯一标识，决定同优先级任务的执行顺序        │
│ callback       │ Function | null  │ 任务执行函数，为 null 表示任务被取消            │
│ priorityLevel  │ 1-5              │ 优先级，决定 timeout 和执行顺序                │
│ startTime      │ number           │ 任务开始时间（ms），用于延迟任务                │
│ expirationTime │ number           │ 过期时间，过期后任务会被强制执行                │
│ sortIndex      │ number           │ 堆排序依据，taskQueue 用过期时间，timerQueue 用开始时间 │
└────────────────┴──────────────────┴──────────────────────────────────────────────┘

典型使用场景:

写入:
  - unstable_scheduleCallback(): 创建 Task 并入队
  - advanceTimers(): 修改 sortIndex，从 timerQueue 移动到 taskQueue

读取:
  - workLoop(): 读取 callback 并执行，检查 expirationTime
  - peek(): 获取堆顶任务
`;

/**
 * 📊 Task 创建过程
 */

const taskCreation = `
📊 Task 创建过程

📁 源码: packages/scheduler/src/forks/Scheduler.js 第 308-388 行

function unstable_scheduleCallback(priorityLevel, callback, options) {
  var currentTime = getCurrentTime();
  
  // 1. 计算开始时间
  var startTime;
  if (options && options.delay > 0) {
    startTime = currentTime + options.delay;
  } else {
    startTime = currentTime;
  }
  
  // 2. 根据优先级计算超时时间
  var timeout;
  switch (priorityLevel) {
    case ImmediatePriority:
      timeout = -1;           // 立即过期
      break;
    case UserBlockingPriority:
      timeout = 250;          // 250ms 后过期
      break;
    case IdlePriority:
      timeout = 1073741823;   // 几乎不过期
      break;
    case LowPriority:
      timeout = 10000;        // 10s 后过期
      break;
    case NormalPriority:
    default:
      timeout = 5000;         // 5s 后过期
  }
  
  // 3. 计算过期时间
  var expirationTime = startTime + timeout;
  
  // 4. 创建任务对象
  var newTask = {
    id: taskIdCounter++,
    callback,
    priorityLevel,
    startTime,
    expirationTime,
    sortIndex: -1,   // 稍后设置
  };
  
  // 5. 入队
  if (startTime > currentTime) {
    // 延迟任务 → timerQueue
    newTask.sortIndex = startTime;
    push(timerQueue, newTask);
  } else {
    // 立即任务 → taskQueue
    newTask.sortIndex = expirationTime;
    push(taskQueue, newTask);
  }
  
  return newTask;
}
`;

// ============================================================
// Part 2: 最小堆（Min Heap）数据结构
// ============================================================

/**
 * 📊 最小堆实现
 *
 * 📁 源码位置: packages/scheduler/src/SchedulerMinHeap.js
 *
 * 用于高效获取最高优先级（最小 sortIndex）的任务
 */

const minHeapStructure = `
📊 最小堆结构

特点:
- 数组存储
- 父节点 < 子节点
- 堆顶是最小元素
- 插入/删除时间复杂度: O(log n)
- 查询最小值时间复杂度: O(1)

数组索引关系:
- 父节点: parentIndex = (index - 1) >>> 1
- 左子节点: leftIndex = (index + 1) * 2 - 1
- 右子节点: rightIndex = leftIndex + 1

示例（按 sortIndex 排序）:
                    [Task(sort=5)]            索引 0
                    /              \\
          [Task(sort=8)]      [Task(sort=10)]  索引 1, 2
          /           \\
  [Task(sort=12)]  [Task(sort=15)]            索引 3, 4

数组表示: [5, 8, 10, 12, 15]
`;

/**
 * 📊 最小堆核心操作
 */

// 堆节点类型
interface HeapNode {
  id: number;
  sortIndex: number;
}

type Heap = HeapNode[];

// push: 插入节点并上浮
function push(heap: Heap, node: HeapNode): void {
  const index = heap.length;
  heap.push(node);
  siftUp(heap, node, index);
}

// peek: 查看堆顶
function peek(heap: Heap): HeapNode | null {
  return heap.length === 0 ? null : heap[0];
}

// pop: 弹出堆顶并下沉
function pop(heap: Heap): HeapNode | null {
  if (heap.length === 0) {
    return null;
  }
  const first = heap[0];
  const last = heap.pop()!;
  if (last !== first) {
    heap[0] = last;
    siftDown(heap, last, 0);
  }
  return first;
}

// siftUp: 上浮操作
function siftUp(heap: Heap, node: HeapNode, i: number) {
  let index = i;
  while (index > 0) {
    const parentIndex = (index - 1) >>> 1;  // 位运算除以 2
    const parent = heap[parentIndex];
    if (compare(parent, node) > 0) {
      // 父节点更大，交换
      heap[parentIndex] = node;
      heap[index] = parent;
      index = parentIndex;
    } else {
      // 父节点更小，停止
      return;
    }
  }
}

// siftDown: 下沉操作
function siftDown(heap: Heap, node: HeapNode, i: number) {
  let index = i;
  const length = heap.length;
  const halfLength = length >>> 1;
  while (index < halfLength) {
    const leftIndex = (index + 1) * 2 - 1;
    const left = heap[leftIndex];
    const rightIndex = leftIndex + 1;
    const right = heap[rightIndex];

    if (compare(left, node) < 0) {
      if (rightIndex < length && compare(right, left) < 0) {
        heap[index] = right;
        heap[rightIndex] = node;
        index = rightIndex;
      } else {
        heap[index] = left;
        heap[leftIndex] = node;
        index = leftIndex;
      }
    } else if (rightIndex < length && compare(right, node) < 0) {
      heap[index] = right;
      heap[rightIndex] = node;
      index = rightIndex;
    } else {
      return;
    }
  }
}

// compare: 比较函数
function compare(a: HeapNode, b: HeapNode) {
  // 先比较 sortIndex，再比较 id
  const diff = a.sortIndex - b.sortIndex;
  return diff !== 0 ? diff : a.id - b.id;
}

/**
 * 📊 最小堆操作示例
 */

const minHeapExample = `
📊 最小堆操作示例

初始状态: taskQueue = []

1. push(Task{id:1, sortIndex:100})
   堆: [100]
   
2. push(Task{id:2, sortIndex:50})
   堆: [50, 100]  (50 上浮到堆顶)
   
3. push(Task{id:3, sortIndex:80})
   堆: [50, 100, 80]
   
4. push(Task{id:4, sortIndex:30})
   堆: [30, 50, 80, 100]  (30 上浮到堆顶)

5. pop() 
   返回 Task{id:4, sortIndex:30}
   堆: [50, 100, 80]  (100 下沉)
   
6. peek()
   返回 Task{id:2, sortIndex:50}
   堆不变
`;

// ============================================================
// Part 3: Lane（车道）优先级模型
// ============================================================

/**
 * 📊 Lane 模型
 *
 * 📁 源码位置: packages/react-reconciler/src/ReactFiberLane.new.js
 *
 * Lane 是 React 内部的优先级模型，需要转换为 Scheduler 优先级
 */

const laneModel = `
📊 Lane 优先级模型

Lane 使用 31 位二进制数表示，每个位代表一个"车道"
数值越小优先级越高

┌────────────────────────────────────────────────────────────────────────────────┐
│ Lane 名称                │ 二进制值                          │ 十进制 │ 优先级  │
├──────────────────────────┼───────────────────────────────────┼────────┼─────────┤
│ SyncLane                 │ 0b0000000000000000000000000000001 │ 1      │ 最高    │
│ InputContinuousLane      │ 0b0000000000000000000000000000100 │ 4      │         │
│ DefaultLane              │ 0b0000000000000000000000000010000 │ 16     │         │
│ TransitionLane1          │ 0b0000000000000000000000001000000 │ 64     │         │
│ ...                      │ ...                               │        │         │
│ TransitionLane16         │ 0b0000000001000000000000000000000 │        │         │
│ RetryLanes               │ 0b0000111110000000000000000000000 │        │         │
│ IdleLane                 │ 0b0100000000000000000000000000000 │        │         │
│ OffscreenLane            │ 0b1000000000000000000000000000000 │        │ 最低    │
└──────────────────────────┴───────────────────────────────────┴────────┴─────────┘

为什么用位运算?
1. 可以同时表示多个 Lane（批量更新）
2. 位运算效率高
3. 方便合并和检查：lanes |= lane, lanes & lane
`;

/**
 * 📊 Lane 到 Scheduler 优先级的映射
 *
 * 📁 源码: packages/react-reconciler/src/ReactFiberWorkLoop.new.js 第 798-820 行
 */

const laneToSchedulerPriority = `
📊 Lane 到 Scheduler 优先级映射

┌─────────────────────────┬────────────────────────┬──────────────────────────┐
│ EventPriority           │ Lane                   │ Scheduler Priority       │
├─────────────────────────┼────────────────────────┼──────────────────────────┤
│ DiscreteEventPriority   │ SyncLane               │ ImmediatePriority (1)    │
│ ContinuousEventPriority │ InputContinuousLane    │ UserBlockingPriority (2) │
│ DefaultEventPriority    │ DefaultLane            │ NormalPriority (3)       │
│ IdleEventPriority       │ IdleLane               │ IdlePriority (5)         │
└─────────────────────────┴────────────────────────┴──────────────────────────┘

映射代码:
switch (lanesToEventPriority(nextLanes)) {
  case DiscreteEventPriority:
    schedulerPriorityLevel = ImmediateSchedulerPriority;
    break;
  case ContinuousEventPriority:
    schedulerPriorityLevel = UserBlockingSchedulerPriority;
    break;
  case DefaultEventPriority:
    schedulerPriorityLevel = NormalSchedulerPriority;
    break;
  case IdleEventPriority:
    schedulerPriorityLevel = IdleSchedulerPriority;
    break;
  default:
    schedulerPriorityLevel = NormalSchedulerPriority;
}

scheduleCallback(schedulerPriorityLevel, performConcurrentWorkOnRoot.bindnull, root));
`;

// ============================================================
// Part 4: FiberRoot 中的调度相关字段
// ============================================================

/**
 * 📊 FiberRoot 中与 Scheduler 相关的字段
 */

interface FiberRootSchedulerFields {
  /**
   * 当前 Scheduler 回调节点
   * 用于取消之前的调度
   */
  callbackNode: Task | null;

  /**
   * 当前回调的优先级
   * 用于判断是否需要重新调度
   */
  callbackPriority: number;

  /**
   * 待处理的 Lanes
   * 存储所有待处理的更新的优先级
   */
  pendingLanes: number;

  /**
   * 已过期的 Lanes
   * 存储需要同步执行的过期更新
   */
  expiredLanes: number;

  /**
   * 被挂起的 Lanes
   * 存储因 Suspense 等原因暂停的更新
   */
  suspendedLanes: number;

  /**
   * 被 ping 的 Lanes
   * 存储 Suspense resolve 后需要恢复的更新
   */
  pingedLanes: number;

  /**
   * 事件时间数组
   * 记录每个 Lane 对应的事件时间
   */
  eventTimes: number[];

  /**
   * 过期时间数组
   * 记录每个 Lane 对应的过期时间
   */
  expirationTimes: number[];
}

const fiberRootSchedulerFieldsTable = `
┌─────────────────┬────────────────┬───────────────────────────────────────────────────┐
│ 字段名           │ 类型           │ 作用                                              │
├─────────────────┼────────────────┼───────────────────────────────────────────────────┤
│ callbackNode    │ Task | null    │ 当前调度任务，用于取消/复用                         │
│ callbackPriority│ Lane           │ 当前回调优先级，相同优先级不需要重新调度             │
│ pendingLanes    │ Lanes          │ 待处理的更新优先级集合                             │
│ expiredLanes    │ Lanes          │ 已过期需要同步执行的更新                           │
│ suspendedLanes  │ Lanes          │ 因 Suspense 暂停的更新                            │
│ pingedLanes     │ Lanes          │ Suspense resolve 后需要恢复的更新                  │
│ eventTimes      │ number[]       │ Lane → 事件时间映射（31 个元素）                   │
│ expirationTimes │ number[]       │ Lane → 过期时间映射                               │
└─────────────────┴────────────────┴───────────────────────────────────────────────────┘

使用场景:

callbackNode:
  - 写入: ensureRootIsScheduled() 中 scheduleCallback() 返回
  - 读取: performConcurrentWorkOnRoot() 检查是否被取消
  - 清除: 任务完成或取消时

callbackPriority:
  - 写入: ensureRootIsScheduled() 中设置
  - 读取: ensureRootIsScheduled() 判断是否需要重新调度
`;

// ============================================================
// Part 5: 数据结构关系图
// ============================================================

const dataStructureRelation = `
📊 数据结构关系图

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│    FiberRoot                                                                │
│    ┌────────────────────────────────┐                                       │
│    │ callbackNode ──────────────────────────────────────┐                   │
│    │ callbackPriority: Lane         │                   │                   │
│    │ pendingLanes: Lanes            │                   │                   │
│    │ expiredLanes: Lanes            │                   │                   │
│    └────────────────────────────────┘                   │                   │
│                                                         ▼                   │
│                                              ┌──────────────────┐           │
│                                              │      Task        │           │
│    Scheduler                                 │ ┌──────────────┐ │           │
│    ┌────────────────────────────────┐        │ │ id           │ │           │
│    │                                │        │ │ callback ────────┐         │
│    │  taskQueue (MinHeap)           │        │ │ priorityLevel│ │ │         │
│    │  ┌────┬────┬────┬────┐        │        │ │ startTime    │ │ │         │
│    │  │ T1 │ T2 │ T3 │ T4 │◄───────────────│ │ expirationT  │ │ │         │
│    │  └────┴────┴────┴────┘        │        │ │ sortIndex    │ │ │         │
│    │                                │        │ └──────────────┘ │ │         │
│    │  timerQueue (MinHeap)          │        └──────────────────┘ │         │
│    │  ┌────┬────┬────┐             │                              │         │
│    │  │ T5 │ T6 │ T7 │             │                              │         │
│    │  └────┴────┴────┘             │                              │         │
│    │                                │                              │         │
│    └────────────────────────────────┘                              │         │
│                                                                    ▼         │
│                                              ┌──────────────────────────┐   │
│                                              │ performConcurrentWorkOn  │   │
│                                              │ Root(root, didTimeout)   │   │
│                                              └──────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

export {
  taskFieldsTable,
  taskCreation,
  minHeapStructure,
  minHeapExample,
  laneModel,
  laneToSchedulerPriority,
  fiberRootSchedulerFieldsTable,
  dataStructureRelation,
  push,
  peek,
  pop,
  siftUp,
  siftDown,
  compare,
};

