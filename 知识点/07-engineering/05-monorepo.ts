/**
 * ============================================================
 * 📚 Monorepo 方案
 * ============================================================
 *
 * 面试考察重点：
 * 1. Monorepo 的概念和优势
 * 2. 常见 Monorepo 工具
 * 3. 包管理和依赖管理
 * 4. 构建优化
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 什么是 Monorepo？
 *
 * Monorepo = 单一仓库管理多个项目/包
 *
 * 📊 Monorepo vs Multirepo
 *
 * Multirepo（多仓库）：
 * - 每个项目独立仓库
 * - 独立版本管理
 * - 独立 CI/CD
 *
 * Monorepo（单仓库）：
 * - 所有项目在一个仓库
 * - 统一版本管理
 * - 代码复用方便
 *
 * 📊 Monorepo 优势
 *
 * 1. 代码复用：共享组件、工具
 * 2. 统一规范：一套 lint、test 配置
 * 3. 原子提交：跨项目修改一次提交
 * 4. 依赖管理：统一版本，避免冲突
 * 5. 协作方便：一个仓库全部代码
 *
 * 📊 Monorepo 挑战
 *
 * 1. 仓库体积大
 * 2. 权限管理复杂
 * 3. 构建优化需要
 * 4. 学习成本
 */

// ============================================================
// 2. 常见 Monorepo 工具
// ============================================================

/**
 * 📊 工具对比
 *
 * ┌─────────────────┬────────────────────────────────────────────────┐
 * │ 工具             │ 特点                                           │
 * ├─────────────────┼────────────────────────────────────────────────┤
 * │ pnpm workspace  │ 原生支持，性能好，推荐                          │
 * │ Turborepo       │ 增量构建，远程缓存，Vercel 出品                  │
 * │ Nx              │ 功能强大，适合大型项目                          │
 * │ Lerna           │ 老牌，专注版本发布                              │
 * │ Rush            │ 微软出品，企业级                                │
 * └─────────────────┴────────────────────────────────────────────────┘
 *
 * 推荐组合：pnpm workspace + Turborepo
 */

// ============================================================
// 3. pnpm Workspace
// ============================================================

const pnpmWorkspaceSetup = `
# 项目结构
monorepo/
├── package.json
├── pnpm-workspace.yaml
├── pnpm-lock.yaml
├── turbo.json
├── packages/
│   ├── ui/                 # 组件库
│   │   ├── package.json
│   │   └── src/
│   ├── utils/              # 工具库
│   │   ├── package.json
│   │   └── src/
│   └── eslint-config/      # ESLint 配置
│       └── package.json
├── apps/
│   ├── web/                # Web 应用
│   │   └── package.json
│   └── admin/              # 管理后台
│       └── package.json
└── tooling/
    └── tsconfig/           # TypeScript 配置
        └── package.json
`;

const pnpmWorkspaceYaml = `
# pnpm-workspace.yaml
packages:
  - 'packages/*'
  - 'apps/*'
  - 'tooling/*'
`;

const rootPackageJson = `
// package.json (根目录)
{
  "name": "monorepo",
  "private": true,
  "scripts": {
    "dev": "turbo dev",
    "build": "turbo build",
    "lint": "turbo lint",
    "test": "turbo test",
    "clean": "turbo clean && rm -rf node_modules"
  },
  "devDependencies": {
    "turbo": "^2.0.0",
    "typescript": "^5.0.0"
  }
}
`;

const packageJson = `
// packages/ui/package.json
{
  "name": "@monorepo/ui",
  "version": "1.0.0",
  "main": "./dist/index.js",
  "module": "./dist/index.mjs",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.mjs",
      "require": "./dist/index.js",
      "types": "./dist/index.d.ts"
    },
    "./button": {
      "import": "./dist/button.mjs",
      "require": "./dist/button.js",
      "types": "./dist/button.d.ts"
    }
  },
  "scripts": {
    "dev": "tsup src/index.ts --watch",
    "build": "tsup src/index.ts --dts",
    "lint": "eslint src"
  },
  "dependencies": {
    "@monorepo/utils": "workspace:*"
  },
  "devDependencies": {
    "@monorepo/eslint-config": "workspace:*",
    "@monorepo/tsconfig": "workspace:*"
  }
}

// apps/web/package.json
{
  "name": "@monorepo/web",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@monorepo/ui": "workspace:*",
    "@monorepo/utils": "workspace:*",
    "react": "^18.0.0"
  }
}
`;

// ============================================================
// 4. Turborepo 配置
// ============================================================

/**
 * 📊 Turborepo 特性
 *
 * 1. 增量构建：只构建变更的包
 * 2. 任务缓存：本地 + 远程缓存
 * 3. 并行执行：自动分析依赖，并行构建
 * 4. 任务管道：定义任务依赖关系
 */

const turboConfig = `
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": ["**/.env.*local"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],  // 先构建依赖的包
      "outputs": ["dist/**", ".next/**"],
      "cache": true
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "lint": {
      "dependsOn": ["^build"],
      "outputs": [],
      "cache": true
    },
    "test": {
      "dependsOn": ["build"],
      "outputs": ["coverage/**"],
      "cache": true
    },
    "clean": {
      "cache": false
    }
  }
}
`;

/**
 * 📊 Turborepo 远程缓存
 *
 * 团队共享构建缓存，避免重复构建
 */

const turboRemoteCache = `
// 登录 Vercel
npx turbo login

// 链接远程缓存
npx turbo link

// 或自建缓存服务器
// turbo.json
{
  "remoteCache": {
    "signature": true,
    "enabled": true
  }
}

// 环境变量
TURBO_API=https://your-cache-server.com
TURBO_TOKEN=your-token
TURBO_TEAM=your-team
`;

// ============================================================
// 5. 依赖管理
// ============================================================

/**
 * 📊 workspace 协议
 *
 * pnpm 的 workspace: 协议用于引用本地包
 *
 * "workspace:*"   - 任意版本，发布时替换为实际版本
 * "workspace:^"   - 发布时替换为 ^x.y.z
 * "workspace:~"   - 发布时替换为 ~x.y.z
 */

/**
 * 📊 依赖提升
 *
 * pnpm 默认不提升依赖（严格模式）
 * 只有显式声明的依赖才能使用
 *
 * .npmrc 配置：
 * shamefully-hoist=false  # 不提升
 * public-hoist-pattern[]="*eslint*"  # 只提升特定包
 */

const npmrcConfig = `
# .npmrc
shamefully-hoist=false
strict-peer-dependencies=false
auto-install-peers=true

# 使用国内镜像
registry=https://registry.npmmirror.com
`;

// ============================================================
// 6. 版本管理与发布
// ============================================================

/**
 * 📊 Changesets 版本管理
 *
 * 1. 添加变更记录
 * 2. 版本升级
 * 3. 生成 CHANGELOG
 * 4. 发布 npm
 */

const changesetsSetup = `
# 安装
pnpm add -Dw @changesets/cli

# 初始化
pnpm changeset init

# 添加变更记录
pnpm changeset
# 选择包、版本类型、描述

# 版本升级
pnpm changeset version

# 发布
pnpm changeset publish
`;

const changesetsConfig = `
// .changeset/config.json
{
  "$schema": "https://unpkg.com/@changesets/config@2.0.0/schema.json",
  "changelog": "@changesets/cli/changelog",
  "commit": false,
  "fixed": [],
  "linked": [],
  "access": "public",
  "baseBranch": "main",
  "updateInternalDependencies": "patch",
  "ignore": ["@monorepo/web", "@monorepo/admin"]  // 忽略私有包
}
`;

// ============================================================
// 7. 共享配置
// ============================================================

const sharedTsConfig = `
// tooling/tsconfig/base.json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  }
}

// tooling/tsconfig/react.json
{
  "extends": "./base.json",
  "compilerOptions": {
    "jsx": "react-jsx",
    "lib": ["DOM", "DOM.Iterable", "ES2020"]
  }
}

// packages/ui/tsconfig.json
{
  "extends": "@monorepo/tsconfig/react.json",
  "compilerOptions": {
    "outDir": "./dist"
  },
  "include": ["src"]
}
`;

const sharedEslintConfig = `
// packages/eslint-config/index.js
module.exports = {
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'prettier',
  ],
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint'],
  rules: {
    '@typescript-eslint/no-unused-vars': 'error',
    '@typescript-eslint/no-explicit-any': 'warn',
  },
};

// packages/eslint-config/react.js
module.exports = {
  extends: [
    './index.js',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
  ],
  settings: {
    react: {
      version: 'detect',
    },
  },
};

// apps/web/.eslintrc.js
module.exports = {
  root: true,
  extends: ['@monorepo/eslint-config/react'],
};
`;

// ============================================================
// 8. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. 循环依赖
 *    - 包 A 依赖包 B，包 B 依赖包 A
 *    - 解决：提取公共部分到新包
 *
 * 2. 幽灵依赖
 *    - 使用未声明的依赖（被提升的）
 *    - 解决：使用 pnpm 严格模式
 *
 * 3. 构建顺序错误
 *    - 依赖的包未先构建
 *    - 解决：Turborepo dependsOn 配置
 *
 * 4. 版本不一致
 *    - 同一依赖多个版本
 *    - 解决：使用 pnpm overrides
 *
 * 5. 缓存失效
 *    - 全局依赖未配置
 *    - 解决：turbo.json globalDependencies
 */

// ============================================================
// 9. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: Monorepo 和 Multirepo 如何选择？
 * A:
 *    Monorepo 适合：
 *    - 项目间共享代码多
 *    - 团队协作紧密
 *    - 需要统一规范
 *
 *    Multirepo 适合：
 *    - 项目独立性强
 *    - 不同团队负责
 *    - 权限隔离需求
 *
 * Q2: 如何处理 Monorepo 构建慢的问题？
 * A:
 *    - 增量构建（Turborepo）
 *    - 远程缓存
 *    - 并行构建
 *    - 只构建变更的包
 *
 * Q3: pnpm 为什么比 npm/yarn 快？
 * A:
 *    - 硬链接：所有项目共享同一份依赖
 *    - 非扁平化：避免幽灵依赖
 *    - 增量安装：只下载新的包
 *
 * Q4: 如何处理 Monorepo 的权限管理？
 * A:
 *    - Git CODEOWNERS 文件
 *    - 分支保护规则
 *    - CI 检查变更范围
 */

// ============================================================
// 10. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景：从零搭建 Monorepo
 *
 * 1. 初始化
 */

const initMonorepo = `
# 创建目录
mkdir my-monorepo && cd my-monorepo

# 初始化 pnpm
pnpm init

# 创建 workspace 配置
echo "packages:
  - 'packages/*'
  - 'apps/*'" > pnpm-workspace.yaml

# 安装 Turborepo
pnpm add -Dw turbo

# 创建目录结构
mkdir -p packages/ui packages/utils apps/web

# 初始化各个包
cd packages/ui && pnpm init
cd ../utils && pnpm init
cd ../../apps/web && pnpm init
`;

/**
 * 2. 配置共享依赖
 */

const setupSharedDeps = `
# 在根目录安装公共开发依赖
pnpm add -Dw typescript eslint prettier

# 在 packages/ui 中添加本地依赖
cd packages/ui
pnpm add @monorepo/utils@workspace:*

# 安装所有依赖
pnpm install
`;

/**
 * 3. 运行命令
 */

const runCommands = `
# 构建所有包
pnpm build

# 只构建某个包
pnpm --filter @monorepo/ui build

# 构建某个包及其依赖
pnpm --filter @monorepo/web... build

# 运行开发服务器
pnpm dev

# 只运行某个应用
pnpm --filter @monorepo/web dev
`;

export {
  pnpmWorkspaceSetup,
  pnpmWorkspaceYaml,
  rootPackageJson,
  packageJson,
  turboConfig,
  turboRemoteCache,
  npmrcConfig,
  changesetsSetup,
  changesetsConfig,
  sharedTsConfig,
  sharedEslintConfig,
  initMonorepo,
  setupSharedDeps,
  runCommands,
};

