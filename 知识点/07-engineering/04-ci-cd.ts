/**
 * ============================================================
 * 📚 CI/CD 流程
 * ============================================================
 *
 * 面试考察重点：
 * 1. CI/CD 的概念和价值
 * 2. 常见 CI/CD 工具
 * 3. 流水线设计
 * 4. 自动化测试和部署
 */

// ============================================================
// 1. 核心概念
// ============================================================

/**
 * 📖 什么是 CI/CD？
 *
 * CI（Continuous Integration）持续集成：
 * - 频繁合并代码到主干
 * - 自动构建和测试
 * - 尽早发现问题
 *
 * CD（Continuous Delivery/Deployment）持续交付/部署：
 * - Delivery：代码随时可以部署（手动触发）
 * - Deployment：自动部署到生产环境
 *
 * 📊 CI/CD 流程
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                                                                 │
 * │  代码提交 ──► 构建 ──► 测试 ──► 代码检查 ──► 部署 ──► 监控      │
 * │                                                                 │
 * │  Git Push   Build    Test    Lint/Audit   Deploy   Monitor     │
 * │                                                                 │
 * └─────────────────────────────────────────────────────────────────┘
 */

// ============================================================
// 2. 常见 CI/CD 工具
// ============================================================

/**
 * 📊 CI/CD 工具对比
 *
 * ┌─────────────────┬────────────────────────────────────────────────┐
 * │ 工具             │ 特点                                           │
 * ├─────────────────┼────────────────────────────────────────────────┤
 * │ GitHub Actions  │ GitHub 原生，免费额度充足，生态好               │
 * │ GitLab CI       │ GitLab 原生，私有部署友好                       │
 * │ Jenkins         │ 老牌，功能强大，需要自建服务器                   │
 * │ CircleCI        │ 云服务，配置简单                                │
 * │ Travis CI       │ 开源项目免费，配置简单                          │
 * └─────────────────┴────────────────────────────────────────────────┘
 */

// ============================================================
// 3. GitHub Actions 配置
// ============================================================

const githubActionsExample = `
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  # 代码检查
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run lint
      - run: npm run type-check

  # 单元测试
  test:
    runs-on: ubuntu-latest
    needs: lint  # 依赖 lint 通过
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run test:coverage
      - uses: codecov/codecov-action@v3  # 上传覆盖率

  # 构建
  build:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-artifact@v4  # 上传构建产物
        with:
          name: dist
          path: dist

  # 部署
  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'  # 只在 main 分支部署
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
      - name: Deploy to server
        run: |
          # 部署脚本
          rsync -avz dist/ user@server:/var/www/app/
`;

// ============================================================
// 4. 完整的前端 CI/CD 流水线
// ============================================================

/**
 * 📊 前端 CI/CD 最佳实践
 *
 * 1. 代码提交阶段（本地）
 *    - husky + lint-staged：提交前检查
 *    - commitlint：规范提交信息
 *
 * 2. CI 阶段
 *    - 代码检查：ESLint、TypeScript
 *    - 单元测试：Jest、Vitest
 *    - 构建：Webpack、Vite
 *    - 分析：Bundle 分析、性能预算
 *
 * 3. CD 阶段
 *    - 预览环境：PR Preview
 *    - 部署：CDN、服务器
 *    - 通知：钉钉、飞书
 */

// husky + lint-staged 配置
const huskyConfig = `
// package.json
{
  "scripts": {
    "prepare": "husky install"
  },
  "lint-staged": {
    "*.{js,jsx,ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{css,scss,less}": [
      "stylelint --fix",
      "prettier --write"
    ]
  }
}

// .husky/pre-commit
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"
npx lint-staged

// .husky/commit-msg
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"
npx commitlint --edit $1
`;

// commitlint 配置
const commitlintConfig = `
// commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat',     // 新功能
        'fix',      // 修复
        'docs',     // 文档
        'style',    // 格式
        'refactor', // 重构
        'perf',     // 性能
        'test',     // 测试
        'chore',    // 构建/工具
        'revert',   // 回滚
      ],
    ],
    'subject-case': [0], // 允许中文
  },
};

// 提交格式
// <type>(<scope>): <subject>
// 例：feat(user): 添加用户登录功能
`;

// ============================================================
// 5. 自动化测试
// ============================================================

/**
 * 📊 测试金字塔
 *
 *          /\\
 *         /  \\      E2E 测试（少）
 *        /────\\     端到端测试
 *       /      \\
 *      /────────\\   集成测试（中）
 *     /          \\  模块间交互
 *    /────────────\\ 单元测试（多）
 *   /              \\ 函数、组件
 */

const testConfig = `
// jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '\\\\.(css|less|scss)$': 'identity-obj-proxy',
  },
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/*.stories.tsx',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
};

// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './vitest.setup.ts',
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: ['node_modules/', 'test/'],
    },
  },
});
`;

// ============================================================
// 6. 部署策略
// ============================================================

/**
 * 📊 常见部署策略
 *
 * 1. 蓝绿部署（Blue-Green）
 *    - 维护两套环境
 *    - 切换流量
 *    - 快速回滚
 *
 * 2. 金丝雀部署（Canary）
 *    - 逐步放量
 *    - 观察指标
 *    - 问题早发现
 *
 * 3. 滚动部署（Rolling）
 *    - 逐个更新实例
 *    - 节省资源
 *
 * 4. 灰度发布
 *    - 按用户/地域分流
 *    - A/B 测试
 */

/**
 * 📊 前端部署方案
 *
 * 1. 静态资源 CDN
 *    - 上传到 OSS/S3
 *    - CDN 分发
 *    - 缓存策略
 *
 * 2. Docker 容器化
 *    - Nginx + 静态文件
 *    - 便于编排和扩展
 *
 * 3. Serverless
 *    - Vercel、Netlify
 *    - 自动扩缩容
 *    - 边缘部署
 */

const dockerfileExample = `
# Dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# 生产镜像
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
`;

const nginxConfigExample = `
# nginx.conf
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # 静态资源缓存
    location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # HTML 不缓存
    location ~* \\.html$ {
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }

    # SPA 路由支持
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API 代理
    location /api {
        proxy_pass http://backend:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # gzip
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;
    gzip_min_length 1024;
}
`;

// ============================================================
// 7. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见错误
 *
 * 1. 缓存未更新
 *    - 使用 contenthash
 *    - HTML 不缓存
 *    - 发布后刷新 CDN 缓存
 *
 * 2. 环境变量泄露
 *    - 敏感信息用 CI 的 Secrets
 *    - 前端只放公开配置
 *
 * 3. 回滚困难
 *    - 保留历史版本
 *    - 版本化部署
 *
 * 4. 测试不充分
 *    - 自动化测试覆盖率
 *    - E2E 测试关键流程
 *
 * 5. 通知不到位
 *    - 部署成功/失败通知
 *    - 错误监控告警
 */

// ============================================================
// 8. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 如何设计一个前端 CI/CD 流程？
 * A:
 *    1. 代码提交：husky + lint-staged
 *    2. CI：lint → test → build
 *    3. CD：preview → deploy
 *    4. 监控：性能、错误监控
 *
 * Q2: 如何实现前端的灰度发布？
 * A:
 *    1. Nginx + 用户标识分流
 *    2. CDN 边缘节点配置
 *    3. 前端配置中心控制
 *    4. A/B 测试平台集成
 *
 * Q3: 如何保证发布安全？
 * A:
 *    - Code Review
 *    - 自动化测试
 *    - 分支保护
 *    - 逐步放量
 *    - 监控告警
 *    - 快速回滚
 *
 * Q4: CI 流程太慢怎么优化？
 * A:
 *    - 缓存 node_modules
 *    - 并行执行任务
 *    - 增量构建
 *    - 只测试变更部分
 *    - 使用更快的 runner
 */

// ============================================================
// 9. 🏢 实战场景
// ============================================================

/**
 * 🏢 场景 1：PR Preview 环境
 *
 * 每个 PR 自动部署预览环境，方便 Review
 */

const prPreviewWorkflow = `
# .github/workflows/preview.yml
name: Deploy Preview

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  deploy-preview:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm ci
      - run: npm run build
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: \${{ secrets.VERCEL_TOKEN }}
          vercel-project-id: \${{ secrets.VERCEL_PROJECT_ID }}
          vercel-org-id: \${{ secrets.VERCEL_ORG_ID }}
      - name: Comment PR
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '🚀 Preview deployed: \${{ steps.deploy.outputs.preview-url }}'
            })
`;

/**
 * 🏢 场景 2：自动发布 npm 包
 */

const npmPublishWorkflow = `
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          registry-url: 'https://registry.npmjs.org'
      - run: npm ci
      - run: npm run build
      - run: npm publish
        env:
          NODE_AUTH_TOKEN: \${{ secrets.NPM_TOKEN }}
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          generate_release_notes: true
`;

export {
  githubActionsExample,
  huskyConfig,
  commitlintConfig,
  testConfig,
  dockerfileExample,
  nginxConfigExample,
  prPreviewWorkflow,
  npmPublishWorkflow,
};

