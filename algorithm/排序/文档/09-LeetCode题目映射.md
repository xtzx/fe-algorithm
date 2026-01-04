# LeetCode 排序题目映射

> 将排序算法知识与算法面试题目关联，做到学以致用

## 📚 目录

1. [按算法分类](#1-按算法分类)
2. [按题型分类](#2-按题型分类)
3. [经典题目详解](#3-经典题目详解)
4. [链表排序专题](#4-链表排序专题)
5. [刷题路线推荐](#5-刷题路线推荐)

---

## 1. 按算法分类

### 1.1 快速排序相关

| 题号 | 题目 | 难度 | 核心考点 | 链接 |
|:---:|------|:---:|---------|------|
| 912 | 排序数组 | M | 基础实现、避免最坏情况 | [🔗](https://leetcode.cn/problems/sort-an-array/) |
| 75 | 颜色分类 | M | 三路快排/荷兰国旗 | [🔗](https://leetcode.cn/problems/sort-colors/) |
| 324 | 摆动排序 II | M | 快速选择 + 三路 | [🔗](https://leetcode.cn/problems/wiggle-sort-ii/) |

### 1.2 快速选择（TopK）

| 题号 | 题目 | 难度 | 核心考点 | 链接 |
|:---:|------|:---:|---------|------|
| 215 | 数组中的第 K 个最大元素 | M | 快速选择 / 堆 | [🔗](https://leetcode.cn/problems/kth-largest-element-in-an-array/) |
| 347 | 前 K 个高频元素 | M | 堆 + 哈希 / 快选 | [🔗](https://leetcode.cn/problems/top-k-frequent-elements/) |
| 692 | 前 K 个高频单词 | M | 堆 + 哈希 + 自定义比较 | [🔗](https://leetcode.cn/problems/top-k-frequent-words/) |
| 973 | 最接近原点的 K 个点 | M | 快速选择 | [🔗](https://leetcode.cn/problems/k-closest-points-to-origin/) |
| 703 | 数据流中的第 K 大元素 | E | 小顶堆 | [🔗](https://leetcode.cn/problems/kth-largest-element-in-a-stream/) |
| 295 | 数据流的中位数 | H | 双堆（大顶 + 小顶） | [🔗](https://leetcode.cn/problems/find-median-from-data-stream/) |

### 1.3 归并排序相关

| 题号 | 题目 | 难度 | 核心考点 | 链接 |
|:---:|------|:---:|---------|------|
| 148 | 排序链表 | M | 链表 + 归并 | [🔗](https://leetcode.cn/problems/sort-list/) |
| 23 | 合并 K 个升序链表 | H | 多路归并 / 堆 | [🔗](https://leetcode.cn/problems/merge-k-sorted-lists/) |
| 88 | 合并两个有序数组 | E | 归并的 merge 步骤 | [🔗](https://leetcode.cn/problems/merge-sorted-array/) |
| 剑指51 | 数组中的逆序对 | H | 归并排序计数 | [🔗](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/) |
| 315 | 计算右侧小于当前元素的个数 | H | 归并排序计数 | [🔗](https://leetcode.cn/problems/count-of-smaller-numbers-after-self/) |
| 327 | 区间和的个数 | H | 归并排序 + 前缀和 | [🔗](https://leetcode.cn/problems/count-of-range-sum/) |

### 1.4 堆排序相关

| 题号 | 题目 | 难度 | 核心考点 | 链接 |
|:---:|------|:---:|---------|------|
| 215 | 数组中的第 K 个最大元素 | M | 堆 / 快选 | [🔗](https://leetcode.cn/problems/kth-largest-element-in-an-array/) |
| 347 | 前 K 个高频元素 | M | 小顶堆 + 哈希 | [🔗](https://leetcode.cn/problems/top-k-frequent-elements/) |
| 23 | 合并 K 个升序链表 | H | 小顶堆 | [🔗](https://leetcode.cn/problems/merge-k-sorted-lists/) |
| 378 | 有序矩阵中第 K 小的元素 | M | 堆 / 二分 | [🔗](https://leetcode.cn/problems/kth-smallest-element-in-a-sorted-matrix/) |
| 264 | 丑数 II | M | 小顶堆 + 去重 | [🔗](https://leetcode.cn/problems/ugly-number-ii/) |

### 1.5 计数排序相关

| 题号 | 题目 | 难度 | 核心考点 | 链接 |
|:---:|------|:---:|---------|------|
| 274 | H 指数 | M | 计数排序思想 | [🔗](https://leetcode.cn/problems/h-index/) |
| 451 | 根据字符出现频率排序 | M | 计数 + 桶排序 | [🔗](https://leetcode.cn/problems/sort-characters-by-frequency/) |
| 1122 | 数组的相对排序 | E | 计数排序 | [🔗](https://leetcode.cn/problems/relative-sort-array/) |

### 1.6 桶排序相关

| 题号 | 题目 | 难度 | 核心考点 | 链接 |
|:---:|------|:---:|---------|------|
| 164 | 最大间距 | H | 桶排序 + 鸽巢原理 | [🔗](https://leetcode.cn/problems/maximum-gap/) |
| 220 | 存在重复元素 III | H | 桶排序思想 | [🔗](https://leetcode.cn/problems/contains-duplicate-iii/) |
| 451 | 根据字符出现频率排序 | M | 桶排序 | [🔗](https://leetcode.cn/problems/sort-characters-by-frequency/) |

### 1.7 基数排序相关

| 题号 | 题目 | 难度 | 核心考点 | 链接 |
|:---:|------|:---:|---------|------|
| 164 | 最大间距 | H | 基数排序（替代方案） | [🔗](https://leetcode.cn/problems/maximum-gap/) |
| 179 | 最大数 | M | 自定义比较（非典型） | [🔗](https://leetcode.cn/problems/largest-number/) |

### 1.8 拓扑排序

| 题号 | 题目 | 难度 | 核心考点 | 链接 |
|:---:|------|:---:|---------|------|
| 207 | 课程表 | M | 拓扑排序 / 检测环 | [🔗](https://leetcode.cn/problems/course-schedule/) |
| 210 | 课程表 II | M | 拓扑排序 + 输出顺序 | [🔗](https://leetcode.cn/problems/course-schedule-ii/) |
| 269 | 火星词典 | H | 拓扑排序（会员） | [🔗](https://leetcode.cn/problems/alien-dictionary/) |

---

## 2. 按题型分类

### 2.1 TopK 系列

```mermaid
flowchart LR
    TopK[TopK 问题] --> Method{选择方法}

    Method -->|k 很小| Heap[小顶堆 O(n log k)]
    Method -->|k ≈ n/2| QuickSelect[快速选择 O(n)]
    Method -->|需要全部有序| Sort[全排序 O(n log n)]

    Heap --> H215[215. 第K大]
    Heap --> H347[347. 前K高频]
    Heap --> H703[703. 数据流第K大]

    QuickSelect --> Q215[215. 第K大]
    QuickSelect --> Q973[973. 最近K点]
```

**核心题目**：
- 215, 347, 692, 973, 703, 295

### 2.2 排序变形

| 题号 | 题目 | 难度 | 技巧 | 链接 |
|:---:|------|:---:|------|------|
| 56 | 合并区间 | M | 排序 + 贪心 | [🔗](https://leetcode.cn/problems/merge-intervals/) |
| 57 | 插入区间 | M | 排序 + 区间处理 | [🔗](https://leetcode.cn/problems/insert-interval/) |
| 252 | 会议室 | E | 排序 + 判断重叠 | [🔗](https://leetcode.cn/problems/meeting-rooms/) |
| 253 | 会议室 II | M | 排序 + 堆/扫描线 | [🔗](https://leetcode.cn/problems/meeting-rooms-ii/) |

### 2.3 自定义比较

| 题号 | 题目 | 难度 | 技巧 | 链接 |
|:---:|------|:---:|------|------|
| 179 | 最大数 | M | 自定义字符串比较 | [🔗](https://leetcode.cn/problems/largest-number/) |
| 406 | 根据身高重建队列 | M | 多关键字排序 + 贪心 | [🔗](https://leetcode.cn/problems/queue-reconstruction-by-height/) |
| 452 | 用最少数量的箭引爆气球 | M | 排序 + 贪心 | [🔗](https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/) |
| 1029 | 两地调度 | M | 排序 + 贪心 | [🔗](https://leetcode.cn/problems/two-city-scheduling/) |

### 2.4 排序 + 双指针

| 题号 | 题目 | 难度 | 技巧 | 链接 |
|:---:|------|:---:|------|------|
| 15 | 三数之和 | M | 排序 + 双指针 | [🔗](https://leetcode.cn/problems/3sum/) |
| 16 | 最接近的三数之和 | M | 排序 + 双指针 | [🔗](https://leetcode.cn/problems/3sum-closest/) |
| 18 | 四数之和 | M | 排序 + 双指针 | [🔗](https://leetcode.cn/problems/4sum/) |
| 167 | 两数之和 II | M | 有序数组 + 双指针 | [🔗](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/) |

### 2.5 排序 + 贪心

| 题号 | 题目 | 难度 | 技巧 | 链接 |
|:---:|------|:---:|------|------|
| 435 | 无重叠区间 | M | 按结束时间排序 | [🔗](https://leetcode.cn/problems/non-overlapping-intervals/) |
| 452 | 用最少箭引爆气球 | M | 按结束时间排序 | [🔗](https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/) |
| 646 | 最长数对链 | M | 按结束时间排序 | [🔗](https://leetcode.cn/problems/maximum-length-of-pair-chain/) |
| 1288 | 删除被覆盖区间 | M | 双关键字排序 | [🔗](https://leetcode.cn/problems/remove-covered-intervals/) |

---

## 3. 经典题目详解

### 3.1 【215】数组中的第 K 个最大元素

**题目**：找出数组中第 k 个最大的元素。

**三种解法对比**：

| 方法 | 时间复杂度 | 空间复杂度 | 特点 |
|------|-----------|-----------|------|
| 全排序 | O(n log n) | O(log n) | 最简单 |
| 小顶堆 | O(n log k) | O(k) | k 小时高效 |
| 快速选择 | O(n) 平均 | O(1) | 最优但不稳定 |

**快速选择解法**：

```typescript
function findKthLargest(nums: number[], k: number): number {
  // 第 k 大 = 第 n-k 小（从 0 开始）
  const targetIndex = nums.length - k;

  function quickSelect(left: number, right: number): number {
    const pivotIndex = partition(left, right);

    if (pivotIndex === targetIndex) {
      return nums[pivotIndex];
    } else if (pivotIndex < targetIndex) {
      return quickSelect(pivotIndex + 1, right);
    } else {
      return quickSelect(left, pivotIndex - 1);
    }
  }

  function partition(left: number, right: number): number {
    // 随机选 pivot 避免最坏情况
    const randomIdx = left + Math.floor(Math.random() * (right - left + 1));
    [nums[randomIdx], nums[right]] = [nums[right], nums[randomIdx]];

    const pivot = nums[right];
    let i = left;

    for (let j = left; j < right; j++) {
      if (nums[j] < pivot) {
        [nums[i], nums[j]] = [nums[j], nums[i]];
        i++;
      }
    }

    [nums[i], nums[right]] = [nums[right], nums[i]];
    return i;
  }

  return quickSelect(0, nums.length - 1);
}
```

**关联章节**：[08-快速选择.md](./算法详解/比较类排序/08-快速选择.md)

---

### 3.2 【148】排序链表

**题目**：对链表进行排序，要求 O(n log n) 时间复杂度和 O(1) 空间复杂度。

**为什么用归并排序？**
- 链表不支持随机访问，快排的 partition 困难
- 归并排序的 merge 步骤很适合链表
- 可以做到 O(1) 空间（自底向上归并）

**解法（自顶向下）**：

```typescript
function sortList(head: ListNode | null): ListNode | null {
  if (!head || !head.next) return head;

  // 1. 快慢指针找中点
  let slow = head, fast = head.next;
  while (fast && fast.next) {
    slow = slow.next!;
    fast = fast.next.next;
  }

  // 2. 断开链表
  const mid = slow.next;
  slow.next = null;

  // 3. 递归排序
  const left = sortList(head);
  const right = sortList(mid);

  // 4. 合并
  return merge(left, right);
}

function merge(l1: ListNode | null, l2: ListNode | null): ListNode | null {
  const dummy = new ListNode(0);
  let curr = dummy;

  while (l1 && l2) {
    if (l1.val < l2.val) {
      curr.next = l1;
      l1 = l1.next;
    } else {
      curr.next = l2;
      l2 = l2.next;
    }
    curr = curr.next;
  }

  curr.next = l1 || l2;
  return dummy.next;
}
```

**关联章节**：[05-归并排序.md](./算法详解/比较类排序/05-归并排序.md)

---

### 3.3 【剑指51】数组中的逆序对

**题目**：统计数组中逆序对的数量。逆序对：i < j 且 nums[i] > nums[j]。

**核心思想**：在归并排序的 merge 阶段统计逆序对。

```typescript
function reversePairs(nums: number[]): number {
  let count = 0;

  function mergeSort(arr: number[], temp: number[], left: number, right: number): void {
    if (left >= right) return;

    const mid = Math.floor((left + right) / 2);
    mergeSort(arr, temp, left, mid);
    mergeSort(arr, temp, mid + 1, right);
    merge(arr, temp, left, mid, right);
  }

  function merge(arr: number[], temp: number[], left: number, mid: number, right: number): void {
    // 复制到临时数组
    for (let i = left; i <= right; i++) {
      temp[i] = arr[i];
    }

    let i = left, j = mid + 1;
    for (let k = left; k <= right; k++) {
      if (i > mid) {
        arr[k] = temp[j++];
      } else if (j > right) {
        arr[k] = temp[i++];
      } else if (temp[i] <= temp[j]) {
        arr[k] = temp[i++];
      } else {
        // temp[i] > temp[j]，产生逆序对
        // 左半部分 [i, mid] 都比 temp[j] 大
        count += mid - i + 1;
        arr[k] = temp[j++];
      }
    }
  }

  const temp = new Array(nums.length);
  mergeSort(nums, temp, 0, nums.length - 1);
  return count;
}
```

**关联章节**：[05-归并排序.md](./算法详解/比较类排序/05-归并排序.md)

---

### 3.4 【164】最大间距

**题目**：找出排序后相邻元素的最大差值，要求线性时间和空间。

**桶排序思想**：

```typescript
function maximumGap(nums: number[]): number {
  const n = nums.length;
  if (n < 2) return 0;

  const min = Math.min(...nums);
  const max = Math.max(...nums);
  if (min === max) return 0;

  // 桶大小：确保最大间距不会出现在桶内
  const bucketSize = Math.max(1, Math.floor((max - min) / (n - 1)));
  const bucketCount = Math.floor((max - min) / bucketSize) + 1;

  // 每个桶只记录最小和最大值
  const buckets: { min: number; max: number }[] = new Array(bucketCount);

  for (const num of nums) {
    const idx = Math.floor((num - min) / bucketSize);
    if (!buckets[idx]) {
      buckets[idx] = { min: num, max: num };
    } else {
      buckets[idx].min = Math.min(buckets[idx].min, num);
      buckets[idx].max = Math.max(buckets[idx].max, num);
    }
  }

  // 最大间距在相邻非空桶之间
  let maxGap = 0;
  let prevMax = min;

  for (const bucket of buckets) {
    if (bucket) {
      maxGap = Math.max(maxGap, bucket.min - prevMax);
      prevMax = bucket.max;
    }
  }

  return maxGap;
}
```

**关联章节**：[02-桶排序.md](./算法详解/非比较排序/02-桶排序.md)

---

### 3.5 【75】颜色分类（荷兰国旗问题）

**题目**：只包含 0、1、2 的数组，原地排序。

**三路快排/荷兰国旗**：

```typescript
function sortColors(nums: number[]): void {
  let p0 = 0;           // [0, p0) 都是 0
  let curr = 0;         // [p0, curr) 都是 1
  let p2 = nums.length; // [p2, n) 都是 2

  while (curr < p2) {
    if (nums[curr] === 0) {
      [nums[p0], nums[curr]] = [nums[curr], nums[p0]];
      p0++;
      curr++;
    } else if (nums[curr] === 2) {
      p2--;
      [nums[curr], nums[p2]] = [nums[p2], nums[curr]];
      // curr 不动，因为换来的值还没检查
    } else {
      curr++;
    }
  }
}
```

**关联章节**：[09-三路快排.md](./算法详解/比较类排序/09-三路快排.md)

---

## 4. 链表排序专题

### 4.1 为什么链表适合归并排序？

| 特性 | 数组 | 链表 |
|------|-----|------|
| 随机访问 | O(1) | O(n) |
| 找中点 | O(1) | O(n) 快慢指针 |
| 合并操作 | 需要额外空间 | 原地修改指针 |
| 适合的算法 | 快排、堆排 | 归并排序 |

### 4.2 为什么链表不适合快排？

```
快排需要：
1. 随机访问选 pivot → 链表 O(n)
2. partition 从两端向中间 → 链表只能单向

归并排序：
1. 只需要找中点 → 快慢指针 O(n)
2. merge 顺序访问 → 链表友好
```

### 4.3 链表排序题目

| 题号 | 题目 | 难度 | 核心技巧 | 链接 |
|:---:|------|:---:|---------|------|
| 148 | 排序链表 | M | 归并排序 | [🔗](https://leetcode.cn/problems/sort-list/) |
| 147 | 对链表进行插入排序 | M | 插入排序 | [🔗](https://leetcode.cn/problems/insertion-sort-list/) |
| 21 | 合并两个有序链表 | E | 归并的 merge | [🔗](https://leetcode.cn/problems/merge-two-sorted-lists/) |
| 23 | 合并 K 个有序链表 | H | 分治归并 / 堆 | [🔗](https://leetcode.cn/problems/merge-k-sorted-lists/) |

---

## 5. 刷题路线推荐

### 5.1 入门级（10 题）

```
基础排序实现 → TopK → 合并
```

1. 912 排序数组（快排实现）
2. 215 第 K 个最大元素（快选/堆）
3. 347 前 K 个高频元素（堆+哈希）
4. 88 合并两个有序数组（归并）
5. 21 合并两个有序链表（归并）
6. 75 颜色分类（三路快排）
7. 56 合并区间（排序+贪心）
8. 15 三数之和（排序+双指针）
9. 148 排序链表（链表归并）
10. 179 最大数（自定义比较）

### 5.2 进阶级（10 题）

```
计数技巧 → 区间问题 → 归并计数
```

1. 274 H 指数（计数排序）
2. 164 最大间距（桶排序）
3. 295 数据流中位数（双堆）
4. 23 合并 K 个有序链表（多路归并）
5. 剑指51 逆序对（归并计数）
6. 315 右侧小于的个数（归并计数）
7. 435 无重叠区间（排序+贪心）
8. 406 重建队列（多关键字排序）
9. 253 会议室 II（扫描线）
10. 327 区间和的个数（归并+前缀和）

### 5.3 刷题清单 Checklist

```
□ 能手写快速排序并避免最坏情况
□ 能手写归并排序（数组和链表版本）
□ 能用快速选择解决 TopK 问题
□ 能用堆解决流式 TopK 问题
□ 理解三路快排的应用场景
□ 能用归并排序思想解决逆序对问题
□ 能用桶排序思想解决最大间距问题
□ 掌握排序 + 双指针的组合技巧
□ 掌握区间问题的排序处理方式
```

---

## 📖 参考资源

- [LeetCode 排序标签](https://leetcode.cn/tag/sorting/)
- [代码随想录 - 排序专题](https://programmercarl.com/)
- [labuladong 的算法小抄](https://labuladong.github.io/algo/)

