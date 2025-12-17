/**
 * SWC 转换示例
 *
 * 本文件展示了 SWC 如何处理各种 TypeScript/JSX 语法。
 * 使用命令: npx swc transform-demo.ts -o output.js
 */

// ============================================
// 1. TypeScript 类型会被移除
// ============================================

interface User {
  id: number;
  name: string;
  email: string;
}

// 类型注解在编译后会被移除
function getUser(id: number): User {
  return {
    id,
    name: 'John',
    email: 'john@example.com',
  };
}

// ============================================
// 2. 装饰器转换 (需要配置 decorators: true)
// ============================================

function Log(target: any, key: string, descriptor: PropertyDescriptor) {
  const original = descriptor.value;
  descriptor.value = function (...args: any[]) {
    console.log(`Calling ${key} with`, args);
    return original.apply(this, args);
  };
  return descriptor;
}

class Calculator {
  @Log
  add(a: number, b: number): number {
    return a + b;
  }
}

// ============================================
// 3. JSX 转换 (需要配置 tsx: true)
// ============================================

import React from 'react';

interface ButtonProps {
  onClick: () => void;
  children: React.ReactNode;
}

// JSX 会被转换为 React.createElement (classic) 或 _jsx (automatic)
const Button: React.FC<ButtonProps> = ({ onClick, children }) => {
  return (
    <button
      className="btn-primary"
      onClick={onClick}
    >
      {children}
    </button>
  );
};

// ============================================
// 4. ES 新语法转换
// ============================================

// 可选链 (Optional Chaining)
const user: User | null = null;
const userName = user?.name ?? 'Anonymous';

// 空值合并 (Nullish Coalescing)
const defaultValue = null ?? 'default';

// 私有字段 (Private Fields)
class Counter {
  #count = 0;

  increment() {
    this.#count++;
  }

  get value() {
    return this.#count;
  }
}

// 类静态块 (Class Static Blocks)
class Config {
  static values: Record<string, string>;

  static {
    this.values = {
      API_URL: 'https://api.example.com',
      VERSION: '1.0.0',
    };
  }
}

// ============================================
// 5. 异步语法
// ============================================

// async/await
async function fetchData(): Promise<User[]> {
  const response = await fetch('/api/users');
  const data = await response.json();
  return data as User[];
}

// Top-level await (需要 target >= es2022 或 module: es6)
// const users = await fetchData();

// ============================================
// 6. 模块语法
// ============================================

// 命名导出
export { getUser, Button };

// 默认导出
export default Calculator;

// 动态导入
async function loadModule() {
  const module = await import('./some-module');
  return module.default;
}

// ============================================
// 编译后对比 (示意)
// ============================================

/*
编译前 (TypeScript + JSX):
const Button: React.FC<ButtonProps> = ({ onClick, children }) => {
  return (
    <button className="btn-primary" onClick={onClick}>
      {children}
    </button>
  );
};

编译后 (JavaScript, runtime: automatic):
import { jsx as _jsx } from "react/jsx-runtime";
const Button = ({ onClick, children }) => {
  return _jsx("button", {
    className: "btn-primary",
    onClick: onClick,
    children: children
  });
};

编译后 (JavaScript, runtime: classic):
const Button = ({ onClick, children }) => {
  return React.createElement("button", {
    className: "btn-primary",
    onClick: onClick
  }, children);
};
*/

