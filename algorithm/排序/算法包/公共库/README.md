# 排序算法公共库

> 为所有排序算法提供可复用的工具函数。

---

## 📁 目录结构

```
公共库/
├── README.md           # 本文件
└── src/
    ├── 比较器.ts        # 比较函数规范
    ├── 数据生成器.ts    # 测试数据生成
    ├── 正确性校验.ts    # 排序结果验证
    ├── 稳定排序辅助.ts  # Schwartzian Transform
    └── 性能计时.ts      # 性能测量工具
```

---

## 🔧 使用方式

### 比较器

```typescript
import { numberAsc, numberDesc, byField, compose } from './src/比较器';

// 数字升序/降序
arr.sort(numberAsc);
arr.sort(numberDesc);

// 按对象字段排序
users.sort(byField('age', numberAsc));

// 组合多个比较器（多列排序）
users.sort(compose(
  byField('age', numberAsc),      // 先按年龄
  byField('name', stringAsc)      // 再按姓名
));
```

### 数据生成器

```typescript
import { generateNumbers, generateObjects } from './src/数据生成器';

// 生成 1000 个随机整数 [0, 10000)
const nums = generateNumbers(1000, 'random', { min: 0, max: 10000 });

// 生成近乎有序的数据
const nearlySorted = generateNumbers(1000, 'nearlySorted', { swapPercent: 5 });

// 生成表格数据
const tableData = generateObjects(100, 'tableRow');
```

### 正确性校验

```typescript
import { verifySorted, verifyPermutation, verifyStable } from './src/正确性校验';

const original = [3, 1, 4, 1, 5];
const sorted = mySort([...original], numberAsc);

verifySorted(sorted, numberAsc);           // 验证有序
verifyPermutation(original, sorted);       // 验证置换
verifyStable(original, sorted, numberAsc); // 验证稳定（如适用）
```

### 稳定排序辅助

```typescript
import { stableSort } from './src/稳定排序辅助';

// 无论底层 sort 是否稳定，保证输出稳定
const sorted = stableSort(arr, (a, b) => a.age - b.age);
```

### 性能计时

```typescript
import { measureSort, Metrics } from './src/性能计时';

const metrics: Metrics = measureSort(mySort, testData, numberAsc);
console.log(`时间: ${metrics.timeMs}ms`);
console.log(`比较次数: ${metrics.comparisons}`);
console.log(`交换次数: ${metrics.swaps}`);
```

---

## 📝 设计原则

1. **零依赖**：仅使用 TS/JS 标准能力
2. **类型安全**：完整的 TypeScript 类型
3. **可测试**：使用纯手写断言，不依赖测试框架
4. **可组合**：函数可自由组合使用

