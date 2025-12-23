/**
 * ============================================================
 * 📚 前端场景关联 - 算法思维在业务中的应用
 * ============================================================
 *
 * 算法不只是面试题，更是解决实际问题的工具。
 * 本文件展示算法思维如何应用到前端日常开发中。
 *
 * 本文件覆盖：
 * 1. 虚拟列表 - 二分查找
 * 2. DOM Diff - 最长公共子序列
 * 3. 权限树 - DFS/BFS 遍历
 * 4. 组件依赖 - 拓扑排序
 * 5. 防抖节流 - 滑动窗口思想
 * 6. 更多场景速查
 */

// ============================================================
// 1. 虚拟列表 - 二分查找
// ============================================================

/**
 * 【场景描述】
 *
 * 渲染 10 万条数据的列表，直接渲染会卡死。
 * 虚拟列表只渲染可视区域的数据。
 *
 * 【算法关联：二分查找】
 *
 * 问题：用户滚动到某个位置，如何快速找到应该渲染哪些数据？
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │                                                             │
 * │  滚动位置 scrollTop = 1500px                                │
 * │  每项高度 itemHeight = 50px                                 │
 * │                                                             │
 * │  如果高度固定：startIndex = Math.floor(scrollTop / height)  │
 * │                                                             │
 * │  如果高度不固定（动态高度）：                                │
 * │  → 需要用二分查找找到第一个 offsetTop > scrollTop 的元素    │
 * │                                                             │
 * └─────────────────────────────────────────────────────────────┘
 */

// 动态高度虚拟列表：二分查找起始索引
interface ItemPosition {
  index: number;
  top: number;
  bottom: number;
  height: number;
}

function findStartIndex(positions: ItemPosition[], scrollTop: number): number {
  let left = 0;
  let right = positions.length - 1;

  while (left < right) {
    const mid = Math.floor((left + right) / 2);
    if (positions[mid].bottom > scrollTop) {
      right = mid;
    } else {
      left = mid + 1;
    }
  }
  return left;
}

/**
 * 【为什么用二分】
 *
 * - positions 数组按 top 有序
 * - 直接遍历：O(n)，10万数据每次滚动都遍历会卡
 * - 二分查找：O(log n)，17 次比较就能定位
 *
 * 【React 虚拟列表库】
 *
 * - react-window
 * - react-virtualized
 * 它们内部都使用类似的二分查找优化
 */

// ============================================================
// 2. DOM Diff - 最长公共子序列
// ============================================================

/**
 * 【场景描述】
 *
 * React/Vue 更新时，需要找出新旧虚拟 DOM 的差异。
 * 如何用最少的操作（增、删、移）完成更新？
 *
 * 【算法关联：最长公共子序列 (LCS)】
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │                                                             │
 * │  旧列表：[A, B, C, D, E]                                    │
 * │  新列表：[A, C, B, E, F]                                    │
 * │                                                             │
 * │  LCS = [A, C, E] 或 [A, B, E]                              │
 * │  这些节点可以复用，只需要移动位置                            │
 * │  其他节点需要新增或删除                                      │
 * │                                                             │
 * └─────────────────────────────────────────────────────────────┘
 *
 * 【Vue3 的优化：最长递增子序列 (LIS)】
 *
 * Vue3 使用 LIS 而不是 LCS：
 * - 找到新列表中相对顺序没变的最长子序列
 * - 这些节点不需要移动，只移动其他节点
 * - 时间复杂度 O(n log n)
 */

// 最长递增子序列（Vue3 Diff 核心算法简化版）
function getSequence(arr: number[]): number[] {
  const n = arr.length;
  const result = [0]; // 存放索引
  const predecessor = new Array(n); // 前驱索引

  for (let i = 1; i < n; i++) {
    const current = arr[i];
    const lastIndex = result[result.length - 1];

    if (current > arr[lastIndex]) {
      predecessor[i] = lastIndex;
      result.push(i);
      continue;
    }

    // 二分查找插入位置
    let left = 0;
    let right = result.length - 1;
    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (arr[result[mid]] < current) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    if (current < arr[result[left]]) {
      if (left > 0) {
        predecessor[i] = result[left - 1];
      }
      result[left] = i;
    }
  }

  // 回溯得到最终结果
  let length = result.length;
  let index = result[length - 1];
  while (length-- > 0) {
    result[length] = index;
    index = predecessor[index];
  }

  return result;
}

/**
 * 【为什么这很重要】
 *
 * DOM 操作是昂贵的，减少 DOM 操作次数能显著提升性能。
 * 理解 Diff 算法有助于：
 * - 合理使用 key 属性
 * - 避免 key={index} 的陷阱
 * - 优化长列表渲染性能
 */

// ============================================================
// 3. 权限树 / 菜单树 - DFS/BFS 遍历
// ============================================================

/**
 * 【场景描述】
 *
 * 后台系统常见需求：
 * - 权限树的勾选/取消
 * - 菜单的递归渲染
 * - 组织架构的搜索
 * - 文件树的遍历
 *
 * 【算法关联：树的遍历】
 */

interface TreeNode {
  id: string;
  name: string;
  children?: TreeNode[];
  checked?: boolean;
}

// 场景一：查找节点（DFS）
function findNode(root: TreeNode, targetId: string): TreeNode | null {
  if (root.id === targetId) return root;

  if (root.children) {
    for (const child of root.children) {
      const found = findNode(child, targetId);
      if (found) return found;
    }
  }

  return null;
}

// 场景二：收集所有选中的叶子节点 ID
function getCheckedLeafIds(root: TreeNode): string[] {
  const result: string[] = [];

  function dfs(node: TreeNode): void {
    // 叶子节点且选中
    if (!node.children?.length && node.checked) {
      result.push(node.id);
      return;
    }

    // 递归子节点
    node.children?.forEach(dfs);
  }

  dfs(root);
  return result;
}

// 场景三：勾选节点时，子节点全选，父节点联动
function toggleCheck(root: TreeNode, targetId: string, checked: boolean): void {
  // 1. 找到目标节点，设置自身及所有子节点
  function setNodeAndChildren(node: TreeNode, value: boolean): void {
    node.checked = value;
    node.children?.forEach((child) => setNodeAndChildren(child, value));
  }

  // 2. 更新父节点状态（需要维护父节点引用或用其他方式）
  // 略：实际实现需要根据具体数据结构

  const target = findNode(root, targetId);
  if (target) {
    setNodeAndChildren(target, checked);
  }
}

// 场景四：树转扁平数组（BFS 层序遍历）
function flattenTree(root: TreeNode): TreeNode[] {
  const result: TreeNode[] = [];
  const queue: TreeNode[] = [root];

  while (queue.length > 0) {
    const node = queue.shift()!;
    result.push(node);
    if (node.children) {
      queue.push(...node.children);
    }
  }

  return result;
}

/**
 * 【延伸思考】
 *
 * Q: DFS 和 BFS 什么时候用？
 *
 * - DFS：找路径、判断连通性、需要递归结构时
 * - BFS：层序遍历、最短路径、同层处理时
 */

// ============================================================
// 4. 组件依赖分析 - 拓扑排序
// ============================================================

/**
 * 【场景描述】
 *
 * 前端工程化场景：
 * - 组件依赖分析（A 依赖 B，B 依赖 C）
 * - 打包顺序确定
 * - 任务调度（某些任务必须在其他任务完成后执行）
 * - 检测循环依赖
 *
 * 【算法关联：拓扑排序】
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │                                                             │
 * │  组件依赖关系：                                              │
 * │                                                             │
 * │  Button → (无依赖)                                          │
 * │  Input  → (无依赖)                                          │
 * │  Form   → [Button, Input]                                   │
 * │  Modal  → [Button]                                          │
 * │  Page   → [Form, Modal]                                     │
 * │                                                             │
 * │  正确的构建顺序：Button, Input, Form, Modal, Page           │
 * │  （被依赖的先构建）                                          │
 * │                                                             │
 * └─────────────────────────────────────────────────────────────┘
 */

// 拓扑排序：确定构建顺序
function topologicalSort(
  components: string[],
  dependencies: Map<string, string[]>
): string[] | null {
  // 计算入度
  const inDegree = new Map<string, number>();
  components.forEach((c) => inDegree.set(c, 0));

  dependencies.forEach((deps, component) => {
    deps.forEach((dep) => {
      inDegree.set(dep, (inDegree.get(dep) || 0) + 1);
    });
  });

  // 从入度为 0 的节点开始（没有被其他组件依赖的）
  const queue = components.filter((c) => inDegree.get(c) === 0);
  const result: string[] = [];

  while (queue.length > 0) {
    const current = queue.shift()!;
    result.push(current);

    // 当前组件的依赖项入度减 1
    const deps = dependencies.get(current) || [];
    deps.forEach((dep) => {
      const newDegree = (inDegree.get(dep) || 0) - 1;
      inDegree.set(dep, newDegree);
      if (newDegree === 0) {
        queue.push(dep);
      }
    });
  }

  // 如果结果长度不等于组件数，说明存在循环依赖
  return result.length === components.length ? result : null;
}

/**
 * 【实际应用】
 *
 * - Webpack/Rollup 的依赖分析
 * - ESLint 的插件加载顺序
 * - Monorepo 中包的发布顺序
 *
 * 【检测循环依赖】
 *
 * 如果拓扑排序无法完成（有节点入度永远不为 0）
 * → 存在循环依赖
 * → 需要报错提示
 */

// ============================================================
// 5. 防抖节流 - 滑动窗口思想
// ============================================================

/**
 * 【场景描述】
 *
 * - 搜索框输入时的请求优化（防抖）
 * - 滚动事件的性能优化（节流）
 * - 按钮防重复点击
 *
 * 【算法关联：滑动窗口思想】
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │                                                             │
 * │  防抖（Debounce）：                                          │
 * │  ────────────────                                           │
 * │  事件触发后等待一段时间，期间有新事件则重新计时               │
 * │  类似「窗口不断右移，直到稳定后才执行」                       │
 * │                                                             │
 * │  时间轴：  ─●─●─●─────────┼──────────                        │
 * │            触发 触发 触发  等待结束    执行                   │
 * │                          └─ delay ─┘                        │
 * │                                                             │
 * │  节流（Throttle）：                                          │
 * │  ────────────────                                           │
 * │  固定时间窗口内只执行一次                                    │
 * │  类似「固定大小的窗口，窗口内只采样一次」                     │
 * │                                                             │
 * │  时间轴：  ─●─●─●─●─┼─●─●─●─┼─●─●─●─┼──                      │
 * │            执行     执行     执行                            │
 * │           └─ 间隔 ─┘└─ 间隔 ─┘                               │
 * │                                                             │
 * └─────────────────────────────────────────────────────────────┘
 */

// 防抖实现
function debounce<T extends (...args: unknown[]) => void>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timer: ReturnType<typeof setTimeout> | null = null;

  return function (this: unknown, ...args: Parameters<T>) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      fn.apply(this, args);
    }, delay);
  };
}

// 节流实现
function throttle<T extends (...args: unknown[]) => void>(
  fn: T,
  interval: number
): (...args: Parameters<T>) => void {
  let lastTime = 0;

  return function (this: unknown, ...args: Parameters<T>) {
    const now = Date.now();
    if (now - lastTime >= interval) {
      lastTime = now;
      fn.apply(this, args);
    }
  };
}

/**
 * 【算法思想的体现】
 *
 * 虽然防抖节流不是「算法题」，但体现了滑动窗口的核心思想：
 * - 在一个时间窗口内做聚合/采样
 * - 窗口的移动策略不同，效果就不同
 *
 * 这种「抽象思维」正是算法学习的价值所在。
 */

// ============================================================
// 6. 更多场景速查表
// ============================================================

/**
 * 【前端场景 → 算法映射速查表】
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                     前端场景 → 算法映射                                  │
 * ├────────────────────────┬────────────────────┬───────────────────────────┤
 * │ 前端场景                │ 算法思想            │ 说明                       │
 * ├────────────────────────┼────────────────────┼───────────────────────────┤
 * │ 虚拟列表定位            │ 二分查找           │ 快速定位可视区域            │
 * │ DOM Diff               │ LCS/LIS           │ 最小化 DOM 操作             │
 * │ 权限树/菜单树           │ DFS/BFS           │ 树的遍历与操作              │
 * │ 组件依赖分析            │ 拓扑排序           │ 确定构建顺序、检测循环       │
 * │ 防抖节流                │ 滑动窗口思想       │ 时间窗口内的事件处理         │
 * │ 搜索高亮                │ 字符串匹配         │ 关键词匹配与高亮            │
 * │ 自动补全                │ 字典树(Trie)       │ 前缀匹配与建议              │
 * │ 路由匹配                │ 状态机/正则        │ URL 模式匹配               │
 * │ 布局计算                │ 贪心/DP            │ 瀑布流、网格布局            │
 * │ 撤销重做                │ 栈                │ 操作历史管理                │
 * │ 表单校验依赖            │ 拓扑排序           │ 字段依赖顺序校验            │
 * │ 拖拽排序                │ 数组操作           │ 插入、删除、移动            │
 * │ 数据去重                │ 哈希表/Set         │ 快速去重                   │
 * │ 列表搜索过滤            │ 遍历 + 哈希        │ 多条件筛选                  │
 * │ 富文本区间操作          │ 区间合并           │ 样式区间的合并与拆分         │
 * │ Canvas 碰撞检测         │ 空间划分           │ 四叉树、网格法              │
 * │ 数据可视化采样          │ 采样算法           │ 大数据量的降采样            │
 * │ 状态机管理              │ 有限状态机         │ 复杂交互状态管理            │
 * │ 缓存淘汰                │ LRU               │ 内存缓存管理                │
 * │ 任务队列                │ 队列/优先队列      │ 请求调度、任务优先级         │
 * └────────────────────────┴────────────────────┴───────────────────────────┘
 *
 * 【学习建议】
 *
 * 不需要刻意去「应用算法」，而是在遇到问题时：
 *
 * 1. 识别问题本质 - 这是什么类型的问题？
 * 2. 联想已知模型 - 有没有类似的算法模型？
 * 3. 评估复杂度   - 当前方案的性能如何？
 * 4. 选择性优化   - 是否需要用更优的算法？
 *
 * 算法思维是一种「直觉」，刷题和实践会逐渐培养这种直觉。
 */

// ============================================================
// 7. 实际案例：LRU 缓存
// ============================================================

/**
 * 【场景描述】
 *
 * 前端缓存策略：
 * - 图片缓存
 * - 接口数据缓存
 * - 组件实例缓存（Vue keep-alive）
 *
 * 缓存满了怎么办？→ 淘汰最久未使用的（LRU）
 *
 * 【算法关联：LRU Cache】
 */

class LRUCache<K, V> {
  private capacity: number;
  private cache: Map<K, V>;

  constructor(capacity: number) {
    this.capacity = capacity;
    this.cache = new Map();
  }

  get(key: K): V | undefined {
    if (!this.cache.has(key)) return undefined;

    // 访问后移到末尾（最新）
    const value = this.cache.get(key)!;
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }

  put(key: K, value: V): void {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.capacity) {
      // 删除最久未使用的（Map 的第一个元素）
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }
    this.cache.set(key, value);
  }
}

/**
 * 【为什么用 Map】
 *
 * JavaScript 的 Map 保持插入顺序，天然适合实现 LRU：
 * - 新元素在末尾
 * - 第一个元素就是最久未使用的
 * - get/set/delete 都是 O(1)
 */

export {
  findStartIndex,
  getSequence,
  findNode,
  getCheckedLeafIds,
  toggleCheck,
  flattenTree,
  topologicalSort,
  debounce,
  throttle,
  LRUCache,
};

