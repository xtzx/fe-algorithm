/**
 * ============================================================
 * 📚 测试体系
 * ============================================================
 *
 * 面试考察重点：
 * 1. 测试金字塔
 * 2. 单元测试
 * 3. 组件测试
 * 4. E2E 测试
 */

// ============================================================
// 1. 测试金字塔
// ============================================================

/**
 * 📊 测试金字塔
 *
 *              /\\
 *             /  \\      E2E 测试（少）
 *            /────\\     端到端，慢但真实
 *           /      \\
 *          /────────\\   集成测试（中）
 *         /          \\  模块间交互
 *        /────────────\\ 单元测试（多）
 *       /              \\ 函数、组件，快速反馈
 *
 * 📊 测试策略
 *
 * - 单元测试：70%（核心逻辑）
 * - 集成测试：20%（模块交互）
 * - E2E 测试：10%（关键流程）
 */

// ============================================================
// 2. 单元测试
// ============================================================

/**
 * 📊 测试框架选择
 *
 * - Jest：功能全面，生态好
 * - Vitest：Vite 原生，速度快
 * - Testing Library：组件测试首选
 */

// Vitest 配置
const vitestConfigExample = `
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './vitest.setup.ts',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/**/*.d.ts',
        'src/**/*.stories.tsx',
        'src/test/',
      ],
      thresholds: {
        branches: 80,
        functions: 80,
        lines: 80,
        statements: 80,
      },
    },
  },
});

// vitest.setup.ts
import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

afterEach(() => {
  cleanup();
});
`;

// 单元测试示例
const unitTestExample = `
// utils/format.ts
export function formatPrice(price: number): string {
  return price.toFixed(2).replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
}

export function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toISOString().split('T')[0];
}

// utils/format.test.ts
import { describe, it, expect } from 'vitest';
import { formatPrice, formatDate } from './format';

describe('formatPrice', () => {
  it('should format integer price', () => {
    expect(formatPrice(1000)).toBe('1,000.00');
  });

  it('should format decimal price', () => {
    expect(formatPrice(1234.5)).toBe('1,234.50');
  });

  it('should handle zero', () => {
    expect(formatPrice(0)).toBe('0.00');
  });

  it('should handle large numbers', () => {
    expect(formatPrice(1234567.89)).toBe('1,234,567.89');
  });
});

describe('formatDate', () => {
  it('should format Date object', () => {
    expect(formatDate(new Date('2024-01-15'))).toBe('2024-01-15');
  });

  it('should format date string', () => {
    expect(formatDate('2024-01-15T10:30:00')).toBe('2024-01-15');
  });
});
`;

// ============================================================
// 3. 组件测试
// ============================================================

/**
 * 📊 React Testing Library
 *
 * 核心理念：
 * - 测试用户行为，不测试实现细节
 * - 通过可访问性查询元素
 */

const componentTestExample = `
// components/LoginForm.tsx
import { useState } from 'react';

interface LoginFormProps {
  onSubmit: (data: { email: string; password: string }) => Promise<void>;
}

export function LoginForm({ onSubmit }: LoginFormProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await onSubmit({ email, password });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="email"
        placeholder="Email"
        value={email}
        onChange={e => setEmail(e.target.value)}
        aria-label="Email"
      />
      <input
        type="password"
        placeholder="Password"
        value={password}
        onChange={e => setPassword(e.target.value)}
        aria-label="Password"
      />
      {error && <div role="alert">{error}</div>}
      <button type="submit" disabled={loading}>
        {loading ? 'Loading...' : 'Login'}
      </button>
    </form>
  );
}

// components/LoginForm.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { LoginForm } from './LoginForm';

describe('LoginForm', () => {
  it('should render form fields', () => {
    render(<LoginForm onSubmit={vi.fn()} />);

    expect(screen.getByLabelText('Email')).toBeInTheDocument();
    expect(screen.getByLabelText('Password')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Login' })).toBeInTheDocument();
  });

  it('should submit form with user input', async () => {
    const user = userEvent.setup();
    const handleSubmit = vi.fn().mockResolvedValue(undefined);

    render(<LoginForm onSubmit={handleSubmit} />);

    await user.type(screen.getByLabelText('Email'), 'test@example.com');
    await user.type(screen.getByLabelText('Password'), 'password123');
    await user.click(screen.getByRole('button', { name: 'Login' }));

    await waitFor(() => {
      expect(handleSubmit).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'password123',
      });
    });
  });

  it('should show error message on failure', async () => {
    const user = userEvent.setup();
    const handleSubmit = vi.fn().mockRejectedValue(new Error('Invalid credentials'));

    render(<LoginForm onSubmit={handleSubmit} />);

    await user.type(screen.getByLabelText('Email'), 'test@example.com');
    await user.type(screen.getByLabelText('Password'), 'wrong');
    await user.click(screen.getByRole('button', { name: 'Login' }));

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Invalid credentials');
    });
  });

  it('should disable button while loading', async () => {
    const user = userEvent.setup();
    const handleSubmit = vi.fn(() => new Promise(() => {})); // 永不 resolve

    render(<LoginForm onSubmit={handleSubmit} />);

    await user.type(screen.getByLabelText('Email'), 'test@example.com');
    await user.type(screen.getByLabelText('Password'), 'password');
    await user.click(screen.getByRole('button', { name: 'Login' }));

    expect(screen.getByRole('button')).toBeDisabled();
    expect(screen.getByRole('button')).toHaveTextContent('Loading...');
  });
});
`;

// ============================================================
// 4. Hook 测试
// ============================================================

const hookTestExample = `
// hooks/useCounter.ts
import { useState, useCallback } from 'react';

export function useCounter(initialValue = 0) {
  const [count, setCount] = useState(initialValue);

  const increment = useCallback(() => setCount(c => c + 1), []);
  const decrement = useCallback(() => setCount(c => c - 1), []);
  const reset = useCallback(() => setCount(initialValue), [initialValue]);

  return { count, increment, decrement, reset };
}

// hooks/useCounter.test.ts
import { renderHook, act } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { useCounter } from './useCounter';

describe('useCounter', () => {
  it('should initialize with default value', () => {
    const { result } = renderHook(() => useCounter());
    expect(result.current.count).toBe(0);
  });

  it('should initialize with custom value', () => {
    const { result } = renderHook(() => useCounter(10));
    expect(result.current.count).toBe(10);
  });

  it('should increment counter', () => {
    const { result } = renderHook(() => useCounter());

    act(() => {
      result.current.increment();
    });

    expect(result.current.count).toBe(1);
  });

  it('should decrement counter', () => {
    const { result } = renderHook(() => useCounter(10));

    act(() => {
      result.current.decrement();
    });

    expect(result.current.count).toBe(9);
  });

  it('should reset counter', () => {
    const { result } = renderHook(() => useCounter(5));

    act(() => {
      result.current.increment();
      result.current.increment();
    });
    expect(result.current.count).toBe(7);

    act(() => {
      result.current.reset();
    });
    expect(result.current.count).toBe(5);
  });
});
`;

// ============================================================
// 5. E2E 测试
// ============================================================

/**
 * 📊 E2E 测试工具
 *
 * - Playwright：微软出品，跨浏览器
 * - Cypress：开发体验好，调试方便
 */

const e2eTestExample = `
// Playwright 配置
// playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
  webServer: {
    command: 'pnpm dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
});

// e2e/login.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Login Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
  });

  test('should login successfully', async ({ page }) => {
    await page.fill('[aria-label="Email"]', 'user@example.com');
    await page.fill('[aria-label="Password"]', 'password123');
    await page.click('button[type="submit"]');

    // 等待跳转到首页
    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('h1')).toHaveText('Welcome');
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.fill('[aria-label="Email"]', 'user@example.com');
    await page.fill('[aria-label="Password"]', 'wrong');
    await page.click('button[type="submit"]');

    await expect(page.locator('[role="alert"]')).toHaveText('Invalid credentials');
    await expect(page).toHaveURL('/login');
  });

  test('should validate required fields', async ({ page }) => {
    await page.click('button[type="submit"]');

    // 检查 HTML5 表单验证
    const email = page.locator('[aria-label="Email"]');
    await expect(email).toBeFocused();
  });
});
`;

// ============================================================
// 6. Mock 与测试替身
// ============================================================

const mockingExample = `
// Mock 函数
import { vi, describe, it, expect, beforeEach } from 'vitest';

describe('Mocking', () => {
  // Mock 函数
  it('should mock function', () => {
    const mockFn = vi.fn();
    mockFn('arg1', 'arg2');

    expect(mockFn).toHaveBeenCalled();
    expect(mockFn).toHaveBeenCalledWith('arg1', 'arg2');
  });

  // Mock 返回值
  it('should mock return value', () => {
    const mockFn = vi.fn()
      .mockReturnValueOnce(1)
      .mockReturnValueOnce(2)
      .mockReturnValue(0);

    expect(mockFn()).toBe(1);
    expect(mockFn()).toBe(2);
    expect(mockFn()).toBe(0);
  });

  // Mock 模块
  it('should mock module', async () => {
    vi.mock('./api', () => ({
      fetchUser: vi.fn().mockResolvedValue({ id: 1, name: 'Test' }),
    }));

    const { fetchUser } = await import('./api');
    const user = await fetchUser(1);

    expect(user).toEqual({ id: 1, name: 'Test' });
  });

  // Spy
  it('should spy on method', () => {
    const obj = {
      method: () => 'original',
    };

    const spy = vi.spyOn(obj, 'method').mockReturnValue('mocked');

    expect(obj.method()).toBe('mocked');
    expect(spy).toHaveBeenCalled();

    spy.mockRestore();
    expect(obj.method()).toBe('original');
  });
});

// Mock API 请求（MSW）
import { setupServer } from 'msw/node';
import { rest } from 'msw';

const server = setupServer(
  rest.get('/api/user/:id', (req, res, ctx) => {
    return res(ctx.json({ id: req.params.id, name: 'Test User' }));
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
`;

// ============================================================
// 7. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见问题
 *
 * 1. 测试实现而非行为
 *    - 不要测试内部状态
 *    - 测试用户可见的行为
 *
 * 2. 测试覆盖率误区
 *    - 高覆盖率 ≠ 高质量
 *    - 关注有意义的测试
 *
 * 3. 测试不稳定
 *    - 避免依赖外部服务
 *    - 避免依赖时间
 *    - 使用 Mock
 *
 * 4. 测试太慢
 *    - 减少 E2E 测试
 *    - 并行执行
 *    - Mock 网络请求
 */

// ============================================================
// 8. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 测试金字塔是什么？
 * A:
 *    底层：大量单元测试（快、稳定）
 *    中层：适量集成测试
 *    顶层：少量 E2E 测试（慢、脆弱）
 *
 * Q2: 如何测试 React 组件？
 * A:
 *    - 使用 React Testing Library
 *    - 测试用户行为
 *    - 通过可访问性查询元素
 *
 * Q3: 什么情况下需要 E2E 测试？
 * A:
 *    - 关键业务流程
 *    - 跨页面交互
 *    - 第三方集成
 *
 * Q4: 如何提高测试覆盖率？
 * A:
 *    - 设置覆盖率门禁
 *    - Code Review 检查测试
 *    - 优先测试核心逻辑
 */

export {
  vitestConfigExample,
  unitTestExample,
  componentTestExample,
  hookTestExample,
  e2eTestExample,
  mockingExample,
};

