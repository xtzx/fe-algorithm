/**
 * Demo - React 排序组件演示
 */

import React, { useState, useMemo } from 'react';
import { SortableTable, type Column } from './components/SortableTable';
import { VirtualSortedListWithControls } from './components/VirtualSortedList';
import { useSortedData, useSortState } from './hooks/useSortedData';

// ============================================================================
// 数据类型
// ============================================================================

interface User {
  id: number;
  name: string;
  email: string;
  age: number;
  department: string;
  joinDate: string;
  salary: number;
}

// ============================================================================
// 数据生成
// ============================================================================

function generateUsers(count: number): User[] {
  const departments = ['Engineering', 'Design', 'Marketing', 'Sales', 'HR'];
  const firstNames = ['Alice', 'Bob', 'Charlie', 'Diana', 'Edward', 'Fiona', 'George', 'Hannah'];
  const lastNames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller'];

  return Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    name: `${firstNames[i % firstNames.length]} ${lastNames[i % lastNames.length]}`,
    email: `user${i + 1}@example.com`,
    age: 22 + Math.floor(Math.random() * 40),
    department: departments[i % departments.length],
    joinDate: new Date(2020 + Math.floor(i / 100), i % 12, (i % 28) + 1)
      .toISOString()
      .split('T')[0],
    salary: 50000 + Math.floor(Math.random() * 100000),
  }));
}

// ============================================================================
// Demo 1: 基础 Hook 使用
// ============================================================================

function BasicHookDemo(): React.ReactElement {
  const users = useMemo(() => generateUsers(100), []);
  const { sortConfig, handleSort, clearSort } = useSortState<User>();
  const { sortedData, isLoading, sortTime } = useSortedData(users, sortConfig);

  const buttonStyle: React.CSSProperties = {
    padding: '8px 16px',
    margin: '4px',
    cursor: 'pointer',
    border: '1px solid #d9d9d9',
    borderRadius: '4px',
    backgroundColor: '#fff',
  };

  const activeButtonStyle: React.CSSProperties = {
    ...buttonStyle,
    backgroundColor: '#1890ff',
    color: '#fff',
    borderColor: '#1890ff',
  };

  return (
    <div style={{ padding: '20px', maxWidth: '800px' }}>
      <h2>Demo 1: 基础 useSortedData Hook</h2>

      <div style={{ marginBottom: '16px' }}>
        <span>排序字段: </span>
        {(['name', 'age', 'department', 'salary'] as const).map(key => (
          <button
            key={key}
            style={sortConfig?.key === key ? activeButtonStyle : buttonStyle}
            onClick={() => handleSort(key)}
          >
            {key}
            {sortConfig?.key === key && (sortConfig.order === 'asc' ? ' ↑' : ' ↓')}
          </button>
        ))}
        <button style={buttonStyle} onClick={clearSort}>
          清除
        </button>
      </div>

      <div style={{ marginBottom: '16px', color: '#666', fontSize: '14px' }}>
        {isLoading ? '排序中...' : `排序耗时: ${sortTime.toFixed(2)}ms`}
      </div>

      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ backgroundColor: '#fafafa' }}>
            <th style={{ padding: '8px', textAlign: 'left' }}>Name</th>
            <th style={{ padding: '8px', textAlign: 'left' }}>Age</th>
            <th style={{ padding: '8px', textAlign: 'left' }}>Department</th>
            <th style={{ padding: '8px', textAlign: 'left' }}>Salary</th>
          </tr>
        </thead>
        <tbody>
          {sortedData.slice(0, 10).map(user => (
            <tr key={user.id} style={{ borderBottom: '1px solid #e8e8e8' }}>
              <td style={{ padding: '8px' }}>{user.name}</td>
              <td style={{ padding: '8px' }}>{user.age}</td>
              <td style={{ padding: '8px' }}>{user.department}</td>
              <td style={{ padding: '8px' }}>${user.salary.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: '8px', color: '#999', fontSize: '12px' }}>
        显示前 10 条，共 {sortedData.length} 条
      </div>
    </div>
  );
}

// ============================================================================
// Demo 2: SortableTable 组件
// ============================================================================

function SortableTableDemo(): React.ReactElement {
  const users = useMemo(() => generateUsers(500), []);

  const columns: Column<User>[] = [
    { key: 'id', title: 'ID', width: 60, sortable: true },
    { key: 'name', title: '姓名', sortable: true },
    { key: 'email', title: '邮箱', sortable: false },
    { key: 'age', title: '年龄', width: 80, sortable: true },
    { key: 'department', title: '部门', sortable: true },
    { key: 'joinDate', title: '入职日期', sortable: true },
    {
      key: 'salary',
      title: '薪资',
      sortable: true,
      render: (value) => `$${(value as number).toLocaleString()}`,
    },
  ];

  return (
    <div style={{ padding: '20px' }}>
      <h2>Demo 2: SortableTable 组件</h2>
      <p style={{ color: '#666', marginBottom: '16px' }}>
        点击表头排序，Shift+点击添加多列排序
      </p>
      <SortableTable
        data={users}
        columns={columns}
        rowKey="id"
        useWorker={false}
      />
    </div>
  );
}

// ============================================================================
// Demo 3: 虚拟滚动列表
// ============================================================================

function VirtualListDemo(): React.ReactElement {
  const [dataSize, setDataSize] = useState(10000);

  const users = useMemo(() => generateUsers(dataSize), [dataSize]);

  const sortableKeys: Array<{ key: keyof User; label: string }> = [
    { key: 'name', label: '姓名' },
    { key: 'age', label: '年龄' },
    { key: 'department', label: '部门' },
    { key: 'salary', label: '薪资' },
  ];

  const renderUser = (user: User) => (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      padding: '0 16px',
      height: '100%',
      gap: '16px',
    }}>
      <span style={{ width: '60px', color: '#999' }}>#{user.id}</span>
      <span style={{ width: '150px', fontWeight: 500 }}>{user.name}</span>
      <span style={{ width: '80px' }}>{user.age}岁</span>
      <span style={{ width: '120px' }}>{user.department}</span>
      <span style={{ color: '#52c41a' }}>${user.salary.toLocaleString()}</span>
    </div>
  );

  const buttonStyle: React.CSSProperties = {
    padding: '6px 12px',
    margin: '4px',
    cursor: 'pointer',
    border: '1px solid #d9d9d9',
    borderRadius: '4px',
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2>Demo 3: 虚拟滚动排序列表</h2>

      <div style={{ marginBottom: '16px' }}>
        <span>数据量: </span>
        {[1000, 10000, 100000].map(size => (
          <button
            key={size}
            style={{
              ...buttonStyle,
              backgroundColor: dataSize === size ? '#1890ff' : '#fff',
              color: dataSize === size ? '#fff' : '#000',
            }}
            onClick={() => setDataSize(size)}
          >
            {size.toLocaleString()}
          </button>
        ))}
      </div>

      <VirtualSortedListWithControls
        data={users}
        itemHeight={50}
        containerHeight={400}
        sortableKeys={sortableKeys}
        renderItem={renderUser}
        useWorker={dataSize > 10000}
      />
    </div>
  );
}

// ============================================================================
// Demo 4: Web Worker 对比
// ============================================================================

function WorkerComparisonDemo(): React.ReactElement {
  const [dataSize, setDataSize] = useState(50000);
  const [results, setResults] = useState<{
    mainThread: number | null;
    worker: number | null;
  }>({ mainThread: null, worker: null });
  const [isRunning, setIsRunning] = useState(false);

  const data = useMemo(() => generateUsers(dataSize), [dataSize]);

  const runBenchmark = async () => {
    setIsRunning(true);
    setResults({ mainThread: null, worker: null });

    // 主线程排序
    const mainStart = performance.now();
    [...data].sort((a, b) => a.name.localeCompare(b.name));
    const mainTime = performance.now() - mainStart;

    setResults(prev => ({ ...prev, mainThread: mainTime }));

    // 等待一下让 UI 更新
    await new Promise(r => setTimeout(r, 100));

    // Web Worker 排序（使用 useSortedData 的内部实现）
    const workerStart = performance.now();
    await new Promise<void>(resolve => {
      const workerCode = `
        self.onmessage = (e) => {
          const { data } = e.data;
          const sorted = [...data].sort((a, b) => a.name.localeCompare(b.name));
          self.postMessage({ sorted });
        };
      `;
      const blob = new Blob([workerCode], { type: 'application/javascript' });
      const worker = new Worker(URL.createObjectURL(blob));
      worker.onmessage = () => {
        worker.terminate();
        resolve();
      };
      worker.postMessage({ data });
    });
    const workerTime = performance.now() - workerStart;

    setResults(prev => ({ ...prev, worker: workerTime }));
    setIsRunning(false);
  };

  const buttonStyle: React.CSSProperties = {
    padding: '8px 16px',
    margin: '4px',
    cursor: 'pointer',
    border: '1px solid #d9d9d9',
    borderRadius: '4px',
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2>Demo 4: 主线程 vs Web Worker</h2>

      <div style={{ marginBottom: '16px' }}>
        <span>数据量: </span>
        {[10000, 50000, 100000, 500000].map(size => (
          <button
            key={size}
            style={{
              ...buttonStyle,
              backgroundColor: dataSize === size ? '#1890ff' : '#fff',
              color: dataSize === size ? '#fff' : '#000',
            }}
            onClick={() => setDataSize(size)}
            disabled={isRunning}
          >
            {size.toLocaleString()}
          </button>
        ))}
      </div>

      <button
        style={{
          ...buttonStyle,
          backgroundColor: '#52c41a',
          color: '#fff',
          borderColor: '#52c41a',
        }}
        onClick={runBenchmark}
        disabled={isRunning}
      >
        {isRunning ? '运行中...' : '运行对比'}
      </button>

      {(results.mainThread !== null || results.worker !== null) && (
        <div style={{ marginTop: '20px' }}>
          <table style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ padding: '8px 16px', border: '1px solid #e8e8e8' }}>方式</th>
                <th style={{ padding: '8px 16px', border: '1px solid #e8e8e8' }}>耗时</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ padding: '8px 16px', border: '1px solid #e8e8e8' }}>主线程</td>
                <td style={{ padding: '8px 16px', border: '1px solid #e8e8e8' }}>
                  {results.mainThread !== null ? `${results.mainThread.toFixed(2)}ms` : '-'}
                </td>
              </tr>
              <tr>
                <td style={{ padding: '8px 16px', border: '1px solid #e8e8e8' }}>Web Worker</td>
                <td style={{ padding: '8px 16px', border: '1px solid #e8e8e8' }}>
                  {results.worker !== null ? `${results.worker.toFixed(2)}ms` : '-'}
                </td>
              </tr>
            </tbody>
          </table>

          <p style={{ marginTop: '12px', color: '#666', fontSize: '14px' }}>
            💡 注意: Web Worker 有创建和通信开销，小数据量时可能更慢。
            <br />
            但 Worker 不阻塞 UI，大数据量时用户体验更好。
          </p>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// 主 App
// ============================================================================

export function App(): React.ReactElement {
  const [activeDemo, setActiveDemo] = useState(1);

  const tabStyle: React.CSSProperties = {
    padding: '12px 24px',
    cursor: 'pointer',
    border: 'none',
    borderBottom: '2px solid transparent',
    backgroundColor: 'transparent',
    fontSize: '14px',
  };

  const activeTabStyle: React.CSSProperties = {
    ...tabStyle,
    borderBottomColor: '#1890ff',
    color: '#1890ff',
    fontWeight: 500,
  };

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif' }}>
      <h1 style={{ padding: '20px', margin: 0, borderBottom: '1px solid #e8e8e8' }}>
        React 排序组件 Demo
      </h1>

      <div style={{ display: 'flex', borderBottom: '1px solid #e8e8e8', padding: '0 20px' }}>
        {[
          { id: 1, label: '基础 Hook' },
          { id: 2, label: 'SortableTable' },
          { id: 3, label: '虚拟滚动' },
          { id: 4, label: 'Worker 对比' },
        ].map(({ id, label }) => (
          <button
            key={id}
            style={activeDemo === id ? activeTabStyle : tabStyle}
            onClick={() => setActiveDemo(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {activeDemo === 1 && <BasicHookDemo />}
      {activeDemo === 2 && <SortableTableDemo />}
      {activeDemo === 3 && <VirtualListDemo />}
      {activeDemo === 4 && <WorkerComparisonDemo />}
    </div>
  );
}

export default App;

