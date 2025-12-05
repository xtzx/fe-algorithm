/**
 * ============================================================
 * 📚 Phase 9: Context 与跨组件状态传播 - Part 1: 概览与使用场景
 * ============================================================
 *
 * 📁 核心源码位置:
 * - packages/react/src/ReactContext.js
 * - packages/react-reconciler/src/ReactFiberNewContext.new.js
 *
 * ⏱️ 预计时间：1-2 小时
 * 🎯 面试权重：⭐⭐⭐⭐
 */

// ============================================================
// Part 1: Context 解决的问题
// ============================================================

/**
 * 📊 Props Drilling 问题
 */

const propsDrillingProblem = `
📊 Props Drilling 问题

什么是 Props Drilling？
═══════════════════════════════════════════════════════════════════════════════

当一个状态需要从顶层组件传递到深层嵌套的组件时，中间的所有组件都需要
传递这个 prop，即使它们并不使用这个数据。

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   典型场景：主题（Theme）需要传递给深层组件                                  │
│                                                                             │
│   <App theme="dark">                                                        │
│     └── <Layout theme="dark">           ← 只是传递，并不使用                │
│           └── <Sidebar theme="dark">    ← 只是传递，并不使用                │
│                 └── <Menu theme="dark"> ← 只是传递，并不使用                │
│                       └── <Button theme="dark">  ← 终于用到了！             │
│                                                                             │
│   问题：                                                                    │
│   1. 中间组件被迫接收并传递它们不需要的 props                               │
│   2. 组件签名变得臃肿                                                       │
│   3. 重构困难：改一个 prop 名需要改整条链                                   │
│   4. 性能问题：中间组件可能因为 theme 变化而重渲染                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


Context 的解决方案
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   使用 Context 后：                                                         │
│                                                                             │
│   <ThemeContext.Provider value="dark">                                      │
│     └── <App>                                                               │
│           └── <Layout>            ← 不需要传递 theme                        │
│                 └── <Sidebar>     ← 不需要传递 theme                        │
│                       └── <Menu>  ← 不需要传递 theme                        │
│                             └── <Button>  ← useContext(ThemeContext)       │
│                                           ← 直接获取 "dark"                 │
│                                                                             │
│   优势：                                                                    │
│   1. 消除 Props Drilling                                                    │
│   2. 组件更加解耦                                                           │
│   3. 数据流更清晰（Provider → Consumer）                                    │
│   4. React 可以精准更新只依赖 Context 的组件                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
`;

// ============================================================
// Part 2: Context API 概览
// ============================================================

/**
 * 📊 Context API 核心成员
 */

const contextApiOverview = `
📊 Context API 核心成员

createContext - 创建 Context 对象
═══════════════════════════════════════════════════════════════════════════════

const ThemeContext = React.createContext(defaultValue);

返回一个 Context 对象，包含：
• Provider: 用于提供值的组件
• Consumer: 用于消费值的组件（Render Props 方式）
• _currentValue: 当前值（内部使用）


Context.Provider - 提供 Context 值
═══════════════════════════════════════════════════════════════════════════════

<ThemeContext.Provider value={theme}>
  {children}
</ThemeContext.Provider>

• 接收 value prop 作为 Context 的当前值
• 所有嵌套的 Consumer/useContext 都能读取这个值
• value 变化时，依赖此 Context 的组件会更新


Context.Consumer - Render Props 消费方式
═══════════════════════════════════════════════════════════════════════════════

<ThemeContext.Consumer>
  {theme => <Button theme={theme} />}
</ThemeContext.Consumer>

• 使用 Render Props 模式
• 较老的消费方式，现在推荐使用 useContext


useContext - Hook 消费方式（推荐）
═══════════════════════════════════════════════════════════════════════════════

function Button() {
  const theme = useContext(ThemeContext);
  return <button className={theme}>Click</button>;
}

• React 16.8+ 引入的 Hook
• 更简洁、更直观的消费方式
• 在函数组件中直接调用
`;

// ============================================================
// Part 3: Context 典型使用场景
// ============================================================

/**
 * 📊 典型使用场景
 */

const typicalUseCases = `
📊 Context 的典型使用场景

场景 1: 主题（Theme）
═══════════════════════════════════════════════════════════════════════════════

// 创建 Context
const ThemeContext = createContext('light');

// 提供者
function App() {
  const [theme, setTheme] = useState('light');
  return (
    <ThemeContext.Provider value={theme}>
      <Header />
      <Main />
      <ThemeToggle onToggle={() => setTheme(t => t === 'light' ? 'dark' : 'light')} />
    </ThemeContext.Provider>
  );
}

// 消费者
function Button() {
  const theme = useContext(ThemeContext);
  return <button className={\`btn-\${theme}\`}>Click</button>;
}


场景 2: 用户认证状态
═══════════════════════════════════════════════════════════════════════════════

const AuthContext = createContext(null);

function AuthProvider({ children }) {
  const [user, setUser] = useState(null);

  const login = async (credentials) => {
    const user = await api.login(credentials);
    setUser(user);
  };

  const logout = () => setUser(null);

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

// 消费
function UserProfile() {
  const { user, logout } = useContext(AuthContext);
  if (!user) return null;
  return (
    <div>
      <span>{user.name}</span>
      <button onClick={logout}>Logout</button>
    </div>
  );
}


场景 3: 多语言国际化（i18n）
═══════════════════════════════════════════════════════════════════════════════

const I18nContext = createContext({ locale: 'en', t: key => key });

function I18nProvider({ locale, children }) {
  const translations = useTranslations(locale);
  const t = key => translations[key] || key;

  return (
    <I18nContext.Provider value={{ locale, t }}>
      {children}
    </I18nContext.Provider>
  );
}

// 消费
function WelcomeMessage() {
  const { t } = useContext(I18nContext);
  return <h1>{t('welcome')}</h1>;  // "欢迎" 或 "Welcome"
}


场景 4: 全局 UI 状态（Modal、Toast）
═══════════════════════════════════════════════════════════════════════════════

const ModalContext = createContext({
  showModal: () => {},
  hideModal: () => {},
});

function ModalProvider({ children }) {
  const [modalContent, setModalContent] = useState(null);

  const showModal = (content) => setModalContent(content);
  const hideModal = () => setModalContent(null);

  return (
    <ModalContext.Provider value={{ showModal, hideModal }}>
      {children}
      {modalContent && <Modal>{modalContent}</Modal>}
    </ModalContext.Provider>
  );
}

// 任何深层组件都可以弹出 Modal
function DeepNestedComponent() {
  const { showModal } = useContext(ModalContext);
  return (
    <button onClick={() => showModal(<ConfirmDialog />)}>
      打开确认框
    </button>
  );
}
`;

// ============================================================
// Part 4: Props 传递 vs Context 传递对比
// ============================================================

/**
 * 📊 Props vs Context 对比
 */

const propsVsContextComparison = `
📊 Props 传递 vs Context 传递 对比

传递机制对比
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────┬───────────────────────┬───────────────────────────────────┐
│ 特性            │ Props                 │ Context                           │
├─────────────────┼───────────────────────┼───────────────────────────────────┤
│ 传递方式        │ 显式逐层传递          │ 隐式广播式传递                    │
├─────────────────┼───────────────────────┼───────────────────────────────────┤
│ 数据流可追踪性  │ 非常清晰              │ 需要查找 Provider                 │
├─────────────────┼───────────────────────┼───────────────────────────────────┤
│ 组件耦合度      │ 中间组件被迫传递      │ 中间组件不感知                    │
├─────────────────┼───────────────────────┼───────────────────────────────────┤
│ 适用场景        │ 少量层级、明确的父子关系 │ 多层嵌套、全局状态              │
├─────────────────┼───────────────────────┼───────────────────────────────────┤
│ 性能考虑        │ 变化可能导致整条链渲染 │ 只有消费者重渲染                  │
├─────────────────┼───────────────────────┼───────────────────────────────────┤
│ 类型检查        │ 完整支持              │ 支持（需定义 Context 类型）       │
└─────────────────┴───────────────────────┴───────────────────────────────────┘


何时使用 Props，何时使用 Context？
═══════════════════════════════════════════════════════════════════════════════

✅ 使用 Props:
─────────────────────────────────────────────────────────────────
• 数据只需要传递 1-2 层
• 需要明确的数据流，方便调试
• 父子组件有明确的依赖关系
• 需要组件复用性，不想绑定特定 Context


✅ 使用 Context:
─────────────────────────────────────────────────────────────────
• 数据需要跨越多层传递（3层以上）
• 多个不相关的组件需要访问同一数据
• 全局配置性数据（主题、语言、用户信息）
• 避免 Props Drilling 污染中间组件


⚠️ Context 的注意事项:
─────────────────────────────────────────────────────────────────
• 不要过度使用：简单场景用 Props 更清晰
• 避免频繁更新的数据放 Context（可能导致大量重渲染）
• 考虑拆分：不同频率更新的数据用不同 Context
• 默认值要有意义：方便组件在没有 Provider 时降级
`;

// ============================================================
// Part 5: 面试要点
// ============================================================

const interviewPoints = `
💡 Part 1 面试要点

Q1: Context 解决了什么问题？
A: 解决 Props Drilling 问题。当需要跨多层组件传递数据时，
   避免中间组件被迫传递它们不需要的 props，
   使组件更加解耦，数据流更清晰。

Q2: Context.Provider 和 Context.Consumer 的作用？
A: - Provider：在组件树中提供 Context 的值，所有嵌套的消费者都能访问
   - Consumer：通过 Render Props 方式消费 Context 值（老方式）
   - useContext：Hook 方式消费 Context 值（推荐）

Q3: 什么时候应该使用 Context？什么时候用 Props？
A: Context：
   - 数据需要跨 3 层以上传递
   - 全局配置性数据（主题、语言、用户信息）
   - 多个不相关组件需要同一数据

   Props：
   - 1-2 层的简单传递
   - 需要明确数据流
   - 组件需要复用性

Q4: Context 使用有哪些注意事项？
A: 1. 不要过度使用，简单场景用 Props
   2. 避免放频繁更新的数据（会导致消费者大量重渲染）
   3. 考虑按更新频率拆分成多个 Context
   4. 设置有意义的默认值

Q5: Context 的典型使用场景有哪些？
A: - 主题（Theme）
   - 用户认证状态
   - 多语言国际化（i18n）
   - 全局 UI 状态（Modal、Toast、Loading）
   - 路由信息
   - Redux/MobX 等状态管理
`;

export {
  propsDrillingProblem,
  contextApiOverview,
  typicalUseCases,
  propsVsContextComparison,
  interviewPoints,
};

