/**
 * ============================================================
 * 📚 组件设计模式
 * ============================================================
 *
 * 面试考察重点：
 * 1. 组件设计原则
 * 2. 常用组件模式
 * 3. 组件库设计
 * 4. 最佳实践
 */

// ============================================================
// 1. 组件设计原则
// ============================================================

/**
 * 📊 SOLID 原则在组件中的应用
 *
 * S - 单一职责：一个组件只做一件事
 * O - 开闭原则：对扩展开放，对修改关闭
 * L - 里氏替换：子组件可替换父组件
 * I - 接口隔离：不依赖不需要的 props
 * D - 依赖倒置：依赖抽象而非具体
 *
 * 📊 其他原则
 *
 * - DRY：不重复自己
 * - KISS：保持简单
 * - YAGNI：不过度设计
 */

// ============================================================
// 2. 组合模式（Compound Components）
// ============================================================

/**
 * 📊 组合组件
 *
 * 将相关组件组合在一起，通过 Context 共享状态
 * 类似 HTML 的 <select> + <option>
 */

import React, { createContext, useContext, useState, ReactNode } from 'react';

// 1. 创建 Context
interface TabsContextValue {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const TabsContext = createContext<TabsContextValue | null>(null);

// 2. 父组件
interface TabsProps {
  defaultTab?: string;
  children: ReactNode;
  onChange?: (tab: string) => void;
}

function Tabs({ defaultTab = '', children, onChange }: TabsProps) {
  const [activeTab, setActiveTab] = useState(defaultTab);

  const handleSetActiveTab = (tab: string) => {
    setActiveTab(tab);
    onChange?.(tab);
  };

  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab: handleSetActiveTab }}>
      <div className="tabs">{children}</div>
    </TabsContext.Provider>
  );
}

// 3. 子组件
interface TabProps {
  value: string;
  children: ReactNode;
}

function Tab({ value, children }: TabProps) {
  const context = useContext(TabsContext);
  if (!context) throw new Error('Tab must be used within Tabs');

  const { activeTab, setActiveTab } = context;

  return (
    <button
      className={`tab ${activeTab === value ? 'active' : ''}`}
      onClick={() => setActiveTab(value)}
    >
      {children}
    </button>
  );
}

interface TabPanelProps {
  value: string;
  children: ReactNode;
}

function TabPanel({ value, children }: TabPanelProps) {
  const context = useContext(TabsContext);
  if (!context) throw new Error('TabPanel must be used within Tabs');

  return context.activeTab === value ? <div className="tab-panel">{children}</div> : null;
}

// 4. 组合导出
Tabs.Tab = Tab;
Tabs.Panel = TabPanel;

// 使用示例
const TabsUsageExample = `
<Tabs defaultTab="tab1" onChange={handleChange}>
  <Tabs.Tab value="tab1">Tab 1</Tabs.Tab>
  <Tabs.Tab value="tab2">Tab 2</Tabs.Tab>

  <Tabs.Panel value="tab1">Content 1</Tabs.Panel>
  <Tabs.Panel value="tab2">Content 2</Tabs.Panel>
</Tabs>
`;

// ============================================================
// 3. 渲染属性（Render Props）
// ============================================================

/**
 * 📊 Render Props
 *
 * 通过 props 传递渲染函数，实现逻辑复用
 */

interface MousePosition {
  x: number;
  y: number;
}

interface MouseTrackerProps {
  render: (position: MousePosition) => ReactNode;
}

function MouseTracker({ render }: MouseTrackerProps) {
  const [position, setPosition] = useState<MousePosition>({ x: 0, y: 0 });

  const handleMouseMove = (e: React.MouseEvent) => {
    setPosition({ x: e.clientX, y: e.clientY });
  };

  return (
    <div onMouseMove={handleMouseMove} style={{ height: '100vh' }}>
      {render(position)}
    </div>
  );
}

// 使用
const MouseTrackerUsage = `
<MouseTracker
  render={({ x, y }) => (
    <div>Mouse position: {x}, {y}</div>
  )}
/>
`;

// 也可以用 children 作为 render prop
interface ChildrenRenderProps {
  children: (position: MousePosition) => ReactNode;
}

function MouseTrackerWithChildren({ children }: ChildrenRenderProps) {
  const [position, setPosition] = useState<MousePosition>({ x: 0, y: 0 });

  const handleMouseMove = (e: React.MouseEvent) => {
    setPosition({ x: e.clientX, y: e.clientY });
  };

  return (
    <div onMouseMove={handleMouseMove}>
      {children(position)}
    </div>
  );
}

// ============================================================
// 4. 高阶组件（HOC）
// ============================================================

/**
 * 📊 高阶组件
 *
 * 接收组件，返回增强后的组件
 */

// 加载状态 HOC
function withLoading<P extends object>(
  WrappedComponent: React.ComponentType<P>
) {
  return function WithLoadingComponent(props: P & { isLoading?: boolean }) {
    const { isLoading, ...rest } = props;

    if (isLoading) {
      return <div className="loading">Loading...</div>;
    }

    return <WrappedComponent {...(rest as P)} />;
  };
}

// 权限 HOC
function withAuth<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  requiredRole?: string
) {
  return function WithAuthComponent(props: P) {
    const { user } = useAuth(); // 假设有 useAuth hook

    if (!user) {
      return <Navigate to="/login" />;
    }

    if (requiredRole && user.role !== requiredRole) {
      return <div>No permission</div>;
    }

    return <WrappedComponent {...props} />;
  };
}

// 模拟的 hooks 和组件
function useAuth() {
  return { user: { role: 'admin' } };
}
function Navigate({ to }: { to: string }) {
  return null;
}

// 使用
const EnhancedComponent = withLoading(withAuth(({ name }: { name: string }) => (
  <div>Hello {name}</div>
)));

// ============================================================
// 5. 自定义 Hooks 复用逻辑
// ============================================================

/**
 * 📊 自定义 Hooks
 *
 * React 16.8+ 推荐的逻辑复用方式
 */

// 表单 Hook
interface UseFormOptions<T> {
  initialValues: T;
  validate?: (values: T) => Partial<Record<keyof T, string>>;
  onSubmit: (values: T) => void | Promise<void>;
}

function useForm<T extends Record<string, any>>({
  initialValues,
  validate,
  onSubmit,
}: UseFormOptions<T>) {
  const [values, setValues] = useState<T>(initialValues);
  const [errors, setErrors] = useState<Partial<Record<keyof T, string>>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (name: keyof T) => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setValues(prev => ({ ...prev, [name]: e.target.value }));
    // 清除错误
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: undefined }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // 验证
    if (validate) {
      const validationErrors = validate(values);
      if (Object.keys(validationErrors).length > 0) {
        setErrors(validationErrors);
        return;
      }
    }

    setIsSubmitting(true);
    try {
      await onSubmit(values);
    } finally {
      setIsSubmitting(false);
    }
  };

  const reset = () => {
    setValues(initialValues);
    setErrors({});
  };

  return {
    values,
    errors,
    isSubmitting,
    handleChange,
    handleSubmit,
    reset,
    setValues,
    setErrors,
  };
}

// 使用
const UseFormExample = `
function LoginForm() {
  const { values, errors, handleChange, handleSubmit, isSubmitting } = useForm({
    initialValues: { email: '', password: '' },
    validate: (values) => {
      const errors: any = {};
      if (!values.email) errors.email = 'Required';
      if (!values.password) errors.password = 'Required';
      return errors;
    },
    onSubmit: async (values) => {
      await login(values);
    },
  });

  return (
    <form onSubmit={handleSubmit}>
      <input
        value={values.email}
        onChange={handleChange('email')}
      />
      {errors.email && <span>{errors.email}</span>}

      <button disabled={isSubmitting}>Submit</button>
    </form>
  );
}
`;

// ============================================================
// 6. 受控与非受控组件
// ============================================================

/**
 * 📊 受控 vs 非受控
 *
 * 受控组件：状态由 React 控制
 * 非受控组件：状态由 DOM 控制
 *
 * 最佳实践：支持两种模式
 */

interface InputProps {
  value?: string;
  defaultValue?: string;
  onChange?: (value: string) => void;
}

function ControlledInput({ value, defaultValue, onChange }: InputProps) {
  // 判断是否是受控模式
  const isControlled = value !== undefined;
  const [internalValue, setInternalValue] = useState(defaultValue ?? '');

  const currentValue = isControlled ? value : internalValue;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;

    if (!isControlled) {
      setInternalValue(newValue);
    }

    onChange?.(newValue);
  };

  return <input value={currentValue} onChange={handleChange} />;
}

// ============================================================
// 7. 组件库设计
// ============================================================

/**
 * 📊 组件库设计原则
 *
 * 1. 一致性：统一的 API 设计
 * 2. 可访问性：a11y 支持
 * 3. 主题化：支持自定义主题
 * 4. 按需加载：Tree Shaking
 * 5. 类型安全：完善的 TypeScript 类型
 */

// 组件库目录结构
const componentLibraryStructure = `
packages/
├── components/
│   ├── button/
│   │   ├── Button.tsx
│   │   ├── Button.test.tsx
│   │   ├── Button.stories.tsx
│   │   ├── Button.module.css
│   │   └── index.ts
│   ├── input/
│   └── ...
├── hooks/
│   ├── useClickOutside.ts
│   └── ...
├── utils/
├── styles/
│   ├── variables.css
│   └── theme.ts
└── index.ts
`;

// 主题系统
const themeSystemExample = `
// 1. 定义主题类型
interface Theme {
  colors: {
    primary: string;
    secondary: string;
    background: string;
    text: string;
  };
  spacing: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
  };
  typography: {
    fontFamily: string;
    fontSize: {
      sm: string;
      md: string;
      lg: string;
    };
  };
}

// 2. 默认主题
const defaultTheme: Theme = {
  colors: {
    primary: '#1890ff',
    secondary: '#52c41a',
    background: '#ffffff',
    text: '#333333',
  },
  // ...
};

// 3. ThemeProvider
const ThemeContext = createContext<Theme>(defaultTheme);

function ThemeProvider({ theme, children }) {
  const mergedTheme = { ...defaultTheme, ...theme };

  return (
    <ThemeContext.Provider value={mergedTheme}>
      <style>
        {\`:root { \${generateCSSVariables(mergedTheme)} }\`}
      </style>
      {children}
    </ThemeContext.Provider>
  );
}

// 4. 使用
function Button({ children }) {
  const theme = useContext(ThemeContext);
  return (
    <button style={{ backgroundColor: theme.colors.primary }}>
      {children}
    </button>
  );
}
`;

// ============================================================
// 8. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. 组件职责不清
 *    - 一个组件做太多事
 *    - 拆分成更小的组件
 *
 * 2. Props 过多
 *    - 超过 5 个考虑重新设计
 *    - 使用组合模式
 *
 * 3. 滥用 Context
 *    - Context 变化会触发所有消费者重渲染
 *    - 拆分 Context 或使用选择器
 *
 * 4. HOC 地狱
 *    - 多层 HOC 难以调试
 *    - 优先使用 Hooks
 *
 * 5. 忽视可访问性
 *    - 添加 aria 属性
 *    - 支持键盘操作
 */

// ============================================================
// 9. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: HOC、Render Props、Hooks 如何选择？
 * A:
 *    - Hooks：首选，逻辑复用
 *    - Render Props：需要灵活渲染
 *    - HOC：需要包装整个组件
 *
 * Q2: 如何设计一个通用的 Modal 组件？
 * A:
 *    - 受控/非受控模式
 *    - Portal 渲染到 body
 *    - 支持自定义内容
 *    - 键盘关闭、点击蒙层关闭
 *    - 动画支持
 *
 * Q3: 组件通信有哪些方式？
 * A:
 *    - Props 向下传递
 *    - 回调函数向上传递
 *    - Context 跨层级
 *    - 状态管理
 *    - EventBus
 *
 * Q4: 如何优化组件渲染性能？
 * A:
 *    - React.memo
 *    - useMemo/useCallback
 *    - 虚拟列表
 *    - 代码分割
 */

// ============================================================
// 10. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景：设计一个 Select 组件
 */

const selectDesignExample = `
// 组合组件模式
<Select
  value={value}
  onChange={onChange}
  placeholder="请选择"
  multiple
  searchable
>
  <Select.Option value="1">选项 1</Select.Option>
  <Select.Option value="2">选项 2</Select.Option>
  <Select.OptGroup label="分组">
    <Select.Option value="3">选项 3</Select.Option>
  </Select.OptGroup>
</Select>

// 功能清单
1. 基础功能：单选、多选
2. 搜索过滤
3. 分组
4. 远程搜索
5. 创建新选项
6. 虚拟滚动（大数据量）
7. 键盘导航
8. 可访问性

// 内部结构
Select
├── SelectTrigger     # 触发器
├── SelectDropdown    # 下拉面板
│   ├── SearchInput   # 搜索框
│   └── OptionList    # 选项列表
│       └── Option    # 单个选项
└── SelectContext     # 共享状态
`;

export {
  Tabs,
  Tab,
  TabPanel,
  MouseTracker,
  MouseTrackerWithChildren,
  withLoading,
  withAuth,
  useForm,
  ControlledInput,
  TabsUsageExample,
  MouseTrackerUsage,
  UseFormExample,
  componentLibraryStructure,
  themeSystemExample,
  selectDesignExample,
};

