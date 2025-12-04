/**
 * ============================================================
 * 📚 状态管理设计
 * ============================================================
 *
 * 面试考察重点：
 * 1. 状态管理的必要性
 * 2. 主流方案对比
 * 3. 设计原则
 * 4. 最佳实践
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 为什么需要状态管理？
 *
 * 1. 组件间共享状态
 * 2. 状态持久化
 * 3. 时间旅行/调试
 * 4. 状态可预测
 *
 * 📊 状态分类
 *
 * - 服务端状态：API 数据（用 React Query/SWR）
 * - 客户端状态：UI 状态、表单状态
 * - URL 状态：路由参数
 * - 表单状态：用户输入（用 React Hook Form）
 */

// ============================================================
// 2. 主流方案对比
// ============================================================

/**
 * 📊 状态管理方案对比
 *
 * ┌─────────────────┬────────────────────────────────────────────────┐
 * │ 方案             │ 特点                                           │
 * ├─────────────────┼────────────────────────────────────────────────┤
 * │ Redux           │ 单向数据流，可预测，生态好，但样板代码多         │
 * │ MobX            │ 响应式，使用简单，但隐式依赖                     │
 * │ Zustand         │ 轻量，API 简洁，支持 React 18                   │
 * │ Jotai           │ 原子化，细粒度更新                              │
 * │ Recoil          │ Facebook 出品，原子化 + 派生状态                │
 * │ Pinia           │ Vue 3 推荐，类型安全                            │
 * │ Vuex            │ Vue 2/3 官方，但样板代码多                      │
 * └─────────────────┴────────────────────────────────────────────────┘
 *
 * 推荐：
 * - 简单项目：Zustand / Jotai
 * - 复杂项目：Redux Toolkit
 * - Vue 项目：Pinia
 */

// ============================================================
// 3. Redux 核心原理
// ============================================================

/**
 * 📊 Redux 三大原则
 *
 * 1. 单一数据源（Single Source of Truth）
 * 2. State 只读（State is Read-Only）
 * 3. 纯函数修改（Changes with Pure Functions）
 *
 * 📊 数据流
 *
 * View → dispatch(Action) → Reducer → Store → View
 */

// 简化版 Redux 实现
type Reducer<S, A> = (state: S, action: A) => S;
type Listener = () => void;

function createStore<S, A>(reducer: Reducer<S, A>, initialState: S) {
  let state = initialState;
  let listeners: Listener[] = [];

  function getState(): S {
    return state;
  }

  function dispatch(action: A): A {
    state = reducer(state, action);
    listeners.forEach(listener => listener());
    return action;
  }

  function subscribe(listener: Listener): () => void {
    listeners.push(listener);
    return () => {
      listeners = listeners.filter(l => l !== listener);
    };
  }

  return { getState, dispatch, subscribe };
}

// Redux Toolkit 使用
const reduxToolkitExample = `
import { createSlice, configureStore } from '@reduxjs/toolkit';

// 创建 slice
const counterSlice = createSlice({
  name: 'counter',
  initialState: { value: 0 },
  reducers: {
    increment: (state) => {
      state.value += 1; // Immer 允许"可变"写法
    },
    decrement: (state) => {
      state.value -= 1;
    },
    incrementByAmount: (state, action) => {
      state.value += action.payload;
    },
  },
});

// 异步 action
const fetchUserById = createAsyncThunk(
  'users/fetchById',
  async (userId: string) => {
    const response = await fetch(\`/api/users/\${userId}\`);
    return response.json();
  }
);

// 配置 store
const store = configureStore({
  reducer: {
    counter: counterSlice.reducer,
  },
});

// 使用
const { increment, decrement } = counterSlice.actions;
dispatch(increment());
`;

// ============================================================
// 4. Zustand 实现原理
// ============================================================

/**
 * 📊 Zustand 特点
 *
 * - 轻量（< 1KB）
 * - 无样板代码
 * - 支持 React 18 并发模式
 * - 支持中间件
 */

// 简化版 Zustand 实现
type SetState<T> = (partial: Partial<T> | ((state: T) => Partial<T>)) => void;
type GetState<T> = () => T;
type StoreApi<T> = {
  getState: GetState<T>;
  setState: SetState<T>;
  subscribe: (listener: Listener) => () => void;
};

function createZustand<T>(createState: (set: SetState<T>, get: GetState<T>) => T): StoreApi<T> {
  let state: T;
  const listeners = new Set<Listener>();

  const getState: GetState<T> = () => state;

  const setState: SetState<T> = (partial) => {
    const nextState = typeof partial === 'function'
      ? (partial as (state: T) => Partial<T>)(state)
      : partial;

    if (!Object.is(nextState, state)) {
      state = { ...state, ...nextState };
      listeners.forEach(listener => listener());
    }
  };

  const subscribe = (listener: Listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  };

  state = createState(setState, getState);

  return { getState, setState, subscribe };
}

// Zustand 使用示例
const zustandExample = `
import { create } from 'zustand';
import { persist, devtools } from 'zustand/middleware';

interface BearState {
  bears: number;
  increase: () => void;
  decrease: () => void;
}

const useBearStore = create<BearState>()(
  devtools(
    persist(
      (set) => ({
        bears: 0,
        increase: () => set((state) => ({ bears: state.bears + 1 })),
        decrease: () => set((state) => ({ bears: state.bears - 1 })),
      }),
      { name: 'bear-storage' }
    )
  )
);

// 使用
function BearCounter() {
  const bears = useBearStore((state) => state.bears);
  const increase = useBearStore((state) => state.increase);
  
  return (
    <div>
      <span>{bears}</span>
      <button onClick={increase}>+</button>
    </div>
  );
}
`;

// ============================================================
// 5. 原子化状态管理（Jotai/Recoil）
// ============================================================

/**
 * 📊 原子化状态管理
 *
 * 特点：
 * - 细粒度更新
 * - 按需订阅
 * - 天然代码分割
 *
 * 概念：
 * - Atom：最小状态单元
 * - Derived/Selector：派生状态
 */

// 简化版 Atom 实现
class Atom<T> {
  private value: T;
  private listeners = new Set<(value: T) => void>();

  constructor(initialValue: T) {
    this.value = initialValue;
  }

  get(): T {
    return this.value;
  }

  set(newValue: T): void {
    if (!Object.is(this.value, newValue)) {
      this.value = newValue;
      this.listeners.forEach(listener => listener(newValue));
    }
  }

  subscribe(listener: (value: T) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}

// 派生 Atom
function derived<T, R>(atoms: Atom<T>[], compute: (values: T[]) => R): Atom<R> {
  const derivedAtom = new Atom(compute(atoms.map(a => a.get())));

  atoms.forEach(atom => {
    atom.subscribe(() => {
      derivedAtom.set(compute(atoms.map(a => a.get())));
    });
  });

  return derivedAtom;
}

// Jotai 使用示例
const jotaiExample = `
import { atom, useAtom } from 'jotai';

// 基础 atom
const countAtom = atom(0);
const textAtom = atom('hello');

// 派生 atom（只读）
const doubleCountAtom = atom((get) => get(countAtom) * 2);

// 派生 atom（可写）
const incrementAtom = atom(
  (get) => get(countAtom),
  (get, set, by: number) => set(countAtom, get(countAtom) + by)
);

// 异步 atom
const userAtom = atom(async (get) => {
  const id = get(userIdAtom);
  const response = await fetch(\`/api/users/\${id}\`);
  return response.json();
});

// 使用
function Counter() {
  const [count, setCount] = useAtom(countAtom);
  const doubleCount = useAtomValue(doubleCountAtom);
  
  return <div>{count} x 2 = {doubleCount}</div>;
}
`;

// ============================================================
// 6. 服务端状态管理
// ============================================================

/**
 * 📊 React Query / SWR
 *
 * 专注于服务端状态：
 * - 自动缓存
 * - 自动重新获取
 * - 乐观更新
 * - 分页/无限滚动
 */

const reactQueryExample = `
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

// 查询
function useUser(userId: string) {
  return useQuery({
    queryKey: ['user', userId],
    queryFn: () => fetchUser(userId),
    staleTime: 5 * 60 * 1000, // 5 分钟内不重新获取
    cacheTime: 30 * 60 * 1000, // 缓存 30 分钟
  });
}

// 变更
function useUpdateUser() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: updateUser,
    // 乐观更新
    onMutate: async (newUser) => {
      await queryClient.cancelQueries(['user', newUser.id]);
      const previousUser = queryClient.getQueryData(['user', newUser.id]);
      queryClient.setQueryData(['user', newUser.id], newUser);
      return { previousUser };
    },
    onError: (err, newUser, context) => {
      // 回滚
      queryClient.setQueryData(['user', newUser.id], context?.previousUser);
    },
    onSettled: (data, error, variables) => {
      // 重新获取
      queryClient.invalidateQueries(['user', variables.id]);
    },
  });
}

// 使用
function UserProfile({ userId }) {
  const { data, isLoading, error } = useUser(userId);
  const updateUser = useUpdateUser();
  
  if (isLoading) return <Loading />;
  if (error) return <Error />;
  
  return (
    <div>
      <h1>{data.name}</h1>
      <button onClick={() => updateUser.mutate({ ...data, name: 'New Name' })}>
        Update
      </button>
    </div>
  );
}
`;

// ============================================================
// 7. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. 过度使用全局状态
 *    - 不是所有状态都需要全局
 *    - 优先考虑组件本地状态
 *
 * 2. 状态设计不合理
 *    - 状态扁平化
 *    - 避免冗余数据
 *
 * 3. 不必要的重渲染
 *    - 选择器返回新对象
 *    - 使用 shallow compare
 *
 * 4. 混淆服务端状态和客户端状态
 *    - 服务端状态用 React Query/SWR
 *    - 客户端状态用 Zustand/Redux
 *
 * 5. 忘记清理订阅
 *    - useEffect 中返回清理函数
 */

// ============================================================
// 8. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: Redux 和 MobX 的区别？
 * A:
 *    Redux：
 *    - 函数式，不可变
 *    - 显式更新
 *    - 可预测性强
 *
 *    MobX：
 *    - 响应式，可变
 *    - 隐式更新
 *    - 使用更简单
 *
 * Q2: 为什么 Redux 要求纯函数？
 * A:
 *    - 可预测性
 *    - 时间旅行调试
 *    - 热重载
 *
 * Q3: Zustand 和 Redux 的区别？
 * A:
 *    Zustand：
 *    - 轻量
 *    - 无样板代码
 *    - hooks 友好
 *
 *    Redux：
 *    - 生态完善
 *    - DevTools 强大
 *    - 适合大型项目
 *
 * Q4: 什么时候用 React Query？
 * A:
 *    处理服务端状态时：
 *    - API 数据缓存
 *    - 自动重新获取
 *    - 乐观更新
 */

// ============================================================
// 9. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景：电商应用状态设计
 */

const ecommerceStateDesign = `
// 状态分层设计

// 1. 服务端状态（React Query）
const { data: products } = useQuery(['products'], fetchProducts);
const { data: user } = useQuery(['user'], fetchUser);

// 2. 客户端全局状态（Zustand）
const useStore = create((set) => ({
  // 购物车
  cart: [],
  addToCart: (product) => set((state) => ({
    cart: [...state.cart, product]
  })),
  
  // UI 状态
  sidebarOpen: false,
  toggleSidebar: () => set((state) => ({
    sidebarOpen: !state.sidebarOpen
  })),
}));

// 3. 表单状态（React Hook Form）
const { register, handleSubmit } = useForm();

// 4. URL 状态（路由参数）
const { id } = useParams();
const [searchParams] = useSearchParams();
`;

/**
 * 🏢 场景：状态持久化
 */

const persistenceExample = `
import { persist, createJSONStorage } from 'zustand/middleware';

const useStore = create(
  persist(
    (set) => ({
      token: null,
      setToken: (token) => set({ token }),
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ token: state.token }), // 只持久化 token
    }
  )
);
`;

export {
  createStore,
  createZustand,
  Atom,
  derived,
  reduxToolkitExample,
  zustandExample,
  jotaiExample,
  reactQueryExample,
  ecommerceStateDesign,
  persistenceExample,
};

