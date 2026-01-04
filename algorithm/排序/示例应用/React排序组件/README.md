# React 排序组件

> 高性能的 React 排序组件，支持 useMemo 缓存、Web Worker 和虚拟滚动

## 🚀 功能特性

- ✅ **useMemo 缓存** - 避免不必要的重新排序
- ✅ **Web Worker** - 大数据量后台排序，不阻塞 UI
- ✅ **虚拟滚动** - 只渲染可见区域，支持百万级数据
- ✅ **多列排序** - 支持 Shift+点击 添加排序列
- ✅ **TypeScript** - 完整类型支持

## 📁 目录结构

```
React排序组件/
├── README.md
├── src/
│   ├── hooks/
│   │   ├── useSortedData.ts      # 排序数据 Hook
│   │   └── useVirtualList.ts     # 虚拟列表 Hook
│   ├── components/
│   │   ├── SortableTable.tsx     # 可排序表格
│   │   └── VirtualSortedList.tsx # 虚拟滚动排序列表
│   ├── utils/
│   │   ├── webWorkerSort.ts      # Web Worker 排序
│   │   └── comparators.ts        # 比较器工具
│   └── demo.tsx                   # 演示示例
└── test/
    └── index.test.ts              # 测试文件
```

## 🔧 使用方法

### 基础排序 Hook

```tsx
import { useSortedData } from './hooks/useSortedData';

function MyComponent({ data }) {
  const { sortedData, isLoading } = useSortedData(data, {
    key: 'name',
    order: 'asc'
  });

  return (
    <ul>
      {sortedData.map(item => (
        <li key={item.id}>{item.name}</li>
      ))}
    </ul>
  );
}
```

### Web Worker 排序

```tsx
const { sortedData, isLoading } = useSortedData(data, sortConfig, {
  useWorker: true,
  workerThreshold: 10000 // 超过 10000 条使用 Worker
});
```

### 可排序表格

```tsx
import { SortableTable } from './components/SortableTable';

function App() {
  const columns = [
    { key: 'name', title: '姓名', sortable: true },
    { key: 'age', title: '年龄', sortable: true },
    { key: 'email', title: '邮箱' },
  ];

  return (
    <SortableTable
      data={users}
      columns={columns}
      rowKey="id"
    />
  );
}
```

### 虚拟滚动列表

```tsx
import { VirtualSortedList } from './components/VirtualSortedList';

function App() {
  return (
    <VirtualSortedList
      data={largeDataset}
      itemHeight={50}
      containerHeight={600}
      sortConfig={{ key: 'name', order: 'asc' }}
      renderItem={(item) => <div>{item.name}</div>}
    />
  );
}
```

## 📊 性能指南

| 数据量 | 推荐方案 |
|-------:|---------|
| < 1,000 | 直接 sort |
| 1,000 - 10,000 | useMemo |
| 10,000 - 100,000 | Web Worker + useMemo |
| > 100,000 | Web Worker + 虚拟滚动 |

## 🧪 测试

```bash
npm test
```

