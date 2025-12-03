/**
 * 📝 题目：接雨水
 * 🔗 链接：https://leetcode.cn/problems/trapping-rain-water/
 * 🏷️ 难度：Hard
 * 🏷️ 标签：栈、数组、双指针、动态规划、单调栈
 *
 * 📋 题目描述：
 * 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，
 * 计算按此排列的柱子，下雨之后能接多少雨水。
 *
 * 示例：
 * 输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
 * 输出：6
 *
 * 图解：
 *                     █
 *         █ ▒ ▒ ▒ █ █ █ █
 *     █ ▒ █ █ ▒ █ █ █ █ █ █
 *   ─────────────────────────
 *     0 1 0 2 1 0 1 3 2 1 2 1
 *
 *   ▒ 表示雨水，总共 6 格
 */

// ============================================================
// 💡 思路分析：这道题的解法是怎么想出来的？
// ============================================================
//
// 核心洞察：每个位置能接的雨水 = min(左边最高, 右边最高) - 当前高度
//
// 1. 暴力思路：对每个位置，向左向右找最高 → O(n²)
//
// 2. 动态规划/预处理：
//    - 预计算每个位置左边的最大值 leftMax[i]
//    - 预计算每个位置右边的最大值 rightMax[i]
//    - water[i] = min(leftMax[i], rightMax[i]) - height[i]
//
// 3. 双指针优化：
//    - 不需要预处理数组，用两个变量记录左右最大值
//    - 关键：移动较小的一边，因为较小边决定了水位
//
// 📊 核心原理图解：
//
//    对于位置 i，能接多少水？
//
//    leftMax │              │ rightMax
//            │    water     │
//            │   ┌─────┐    │
//            │   │     │    │
//    ────────┴───┼─────┼────┴────
//                │ h[i]│
//                └─────┘
//
//    water[i] = min(leftMax, rightMax) - height[i]
//
//    如果 leftMax < rightMax：
//    - 当前位置的水位由 leftMax 决定
//    - 可以安全计算 water[left]
//    - 然后 left++

// ============================================================
// 解法一：动态规划/预处理（最直观）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(n)

/**
 * 📊 执行过程图解：
 *
 * height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
 *
 * leftMax[i] = max(height[0..i])
 * leftMax  = [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]
 *
 * rightMax[i] = max(height[i..n-1])
 * rightMax = [3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1]
 *
 * water[i] = min(leftMax[i], rightMax[i]) - height[i]
 * water    = [0, 0, 1, 0, 1, 2, 1, 0, 0, 1, 0, 0]
 *
 * 总和 = 0+0+1+0+1+2+1+0+0+1+0+0 = 6
 */
function trap_v1(height: number[]): number {
  const n = height.length;
  if (n <= 2) return 0;

  // 预计算左边最大值
  const leftMax: number[] = new Array(n);
  leftMax[0] = height[0];
  for (let i = 1; i < n; i++) {
    leftMax[i] = Math.max(leftMax[i - 1], height[i]);
  }

  // 预计算右边最大值
  const rightMax: number[] = new Array(n);
  rightMax[n - 1] = height[n - 1];
  for (let i = n - 2; i >= 0; i--) {
    rightMax[i] = Math.max(rightMax[i + 1], height[i]);
  }

  // 计算雨水
  let water = 0;
  for (let i = 0; i < n; i++) {
    water += Math.min(leftMax[i], rightMax[i]) - height[i];
  }

  return water;
}

// ============================================================
// 解法二：双指针（推荐）
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(1)
// ✅ 最优解：不需要额外数组

/**
 * 📊 双指针执行过程图解：
 *
 * height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
 *
 * 核心思想：
 * - 左右指针从两端向中间移动
 * - leftMax 记录 left 左边（含）的最大值
 * - rightMax 记录 right 右边（含）的最大值
 * - 移动较小 max 对应的指针
 *
 * 为什么移动较小的？
 * - 如果 leftMax < rightMax，那么 left 位置的水位由 leftMax 决定
 * - 不管 right 再往左走能不能找到更高的，left 位置的答案已经确定了
 *
 * 🔄 流程图 (Mermaid):
 * ```mermaid
 * flowchart TD
 *     A[left=0, right=n-1, leftMax=0, rightMax=0] --> B{left < right?}
 *     B -->|Yes| C{height[left] < height[right]?}
 *     C -->|Yes| D{height[left] >= leftMax?}
 *     D -->|Yes| E[更新 leftMax]
 *     D -->|No| F[累加 leftMax - height[left]]
 *     E --> G[left++]
 *     F --> G
 *     C -->|No| H{height[right] >= rightMax?}
 *     H -->|Yes| I[更新 rightMax]
 *     H -->|No| J[累加 rightMax - height[right]]
 *     I --> K[right--]
 *     J --> K
 *     G --> B
 *     K --> B
 *     B -->|No| L[返回 water]
 * ```
 */
function trap_v2(height: number[]): number {
  const n = height.length;
  if (n <= 2) return 0;

  let left = 0;
  let right = n - 1;
  let leftMax = 0;
  let rightMax = 0;
  let water = 0;

  while (left < right) {
    // 更新左右最大值
    leftMax = Math.max(leftMax, height[left]);
    rightMax = Math.max(rightMax, height[right]);

    // 移动较小的一边
    if (leftMax < rightMax) {
      // left 位置的水位由 leftMax 决定
      water += leftMax - height[left];
      left++;
    } else {
      // right 位置的水位由 rightMax 决定
      water += rightMax - height[right];
      right--;
    }
  }

  return water;
}

// ============================================================
// 解法三：单调栈
// ============================================================
// ⏱️ 时间复杂度：O(n) | 空间复杂度：O(n)
// 📝 思路：找到每个凹槽，横向计算水量

/**
 * 📊 单调栈思路图解：
 *
 * 维护一个单调递减栈（栈底到栈顶递减）
 * 当遇到更高的柱子时，可以计算凹槽中的水
 *
 *         █
 *     █   █ █
 *   █ █ █ █ █
 *   ─────────
 *   2 1 0 2 1
 *
 * 当遍历到高度 2 时，栈中有 [2, 1, 0]
 * 弹出 0，计算 0 和 1 之间、0 和 2 之间的水
 */
function trap_v3(height: number[]): number {
  const n = height.length;
  if (n <= 2) return 0;

  const stack: number[] = []; // 存储下标
  let water = 0;

  for (let i = 0; i < n; i++) {
    // 当前柱子比栈顶高，可能形成凹槽
    while (stack.length > 0 && height[i] > height[stack[stack.length - 1]]) {
      const bottom = stack.pop()!; // 凹槽底部

      if (stack.length === 0) break; // 没有左边界

      const left = stack[stack.length - 1]; // 左边界
      const right = i; // 右边界

      // 计算这一层的水量
      const h = Math.min(height[left], height[right]) - height[bottom];
      const w = right - left - 1;
      water += h * w;
    }

    stack.push(i);
  }

  return water;
}

// ============================================================
// 🔄 解法对比
// ============================================================
/**
 * | 解法       | 时间  | 空间  | 特点                           |
 * |-----------|-------|-------|-------------------------------|
 * | 预处理     | O(n)  | O(n)  | 最直观，按列计算                |
 * | 双指针     | O(n)  | O(1)  | 最优解，按列计算                |
 * | 单调栈     | O(n)  | O(n)  | 按行计算，思路独特              |
 *
 * 面试建议：
 * 1. 先说预处理思路（最直观）
 * 2. 再优化到双指针（展示优化能力）
 * 3. 如果时间充裕，可以提一下单调栈解法
 */

// ============================================================
// ⚠️ 易错点
// ============================================================
/**
 * 1. 边界条件：n <= 2 时返回 0（无法接水）
 *
 * 2. 双指针移动条件：
 *    - 比较的是 leftMax 和 rightMax，不是 height[left] 和 height[right]
 *    - 移动较小 max 对应的指针
 *
 * 3. 单调栈：
 *    - 栈中存储的是下标，不是高度
 *    - 需要考虑没有左边界的情况
 *
 * 4. 雨水量不会是负数：
 *    - min(leftMax, rightMax) >= height[i] 始终成立
 */

// ============================================================
// 🔗 举一反三：相似题目
// ============================================================
/**
 * - 盛最多水的容器 → 双指针（思路相似）
 * - 柱状图中最大的矩形 → 单调栈
 * - 最大矩形 → 单调栈 + 动态规划
 */

// ============================================================
// 🏢 前端业务场景
// ============================================================
/**
 * 1. 图表填充：在折线图中计算特定区域的面积
 * 2. 布局计算：计算元素之间的可用空间
 * 3. 瀑布流：计算每一行的剩余空间
 * 4. 音频可视化：计算波形图的"凹陷"区域
 */

// 导出主解法
export { trap_v1, trap_v2, trap_v3 };
export default trap_v2;

