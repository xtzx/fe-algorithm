/**
 * ============================================================
 * 📚 代码质量体系
 * ============================================================
 *
 * 面试考察重点：
 * 1. 代码规范
 * 2. 静态检查
 * 3. Code Review
 * 4. 质量门禁
 */

// ============================================================
// 1. 代码规范体系
// ============================================================

/**
 * 📊 代码规范层次
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                      代码规范金字塔                              │
 * │                                                                 │
 * │                         /\\                                      │
 * │                        /  \\     架构规范                        │
 * │                       /────\\    模块划分、依赖管理               │
 * │                      /      \\                                   │
 * │                     /────────\\  编码规范                        │
 * │                    /          \\ 命名、注释、复杂度               │
 * │                   /────────────\\ 风格规范                       │
 * │                  /              \\ 缩进、引号、分号               │
 * │                 /────────────────\\                              │
 * │                                                                 │
 * │   自动化程度：风格 > 编码 > 架构                                  │
 * └─────────────────────────────────────────────────────────────────┘
 */

// ESLint 配置
const eslintConfigExample = `
// .eslintrc.js
module.exports = {
  root: true,
  env: {
    browser: true,
    es2022: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'plugin:import/recommended',
    'plugin:import/typescript',
    'prettier', // 放最后，关闭冲突规则
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: './tsconfig.json',
  },
  plugins: ['@typescript-eslint', 'react', 'import'],
  settings: {
    react: { version: 'detect' },
    'import/resolver': {
      typescript: true,
    },
  },
  rules: {
    // TypeScript
    '@typescript-eslint/no-unused-vars': 'error',
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/explicit-function-return-type': 'off',
    
    // React
    'react/react-in-jsx-scope': 'off',
    'react/prop-types': 'off',
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'warn',
    
    // Import
    'import/order': [
      'error',
      {
        groups: ['builtin', 'external', 'internal', 'parent', 'sibling', 'index'],
        'newlines-between': 'always',
        alphabetize: { order: 'asc' },
      },
    ],
    'import/no-cycle': 'error',
    
    // 通用
    'no-console': ['warn', { allow: ['warn', 'error'] }],
    'prefer-const': 'error',
    'no-var': 'error',
  },
};
`;

// Prettier 配置
const prettierConfigExample = `
// .prettierrc
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 100,
  "bracketSpacing": true,
  "arrowParens": "avoid",
  "endOfLine": "lf"
}

// .prettierignore
node_modules
dist
coverage
*.min.js
`;

// ============================================================
// 2. Git 提交规范
// ============================================================

/**
 * 📊 Conventional Commits
 *
 * 格式：<type>(<scope>): <subject>
 *
 * type：
 * - feat: 新功能
 * - fix: 修复
 * - docs: 文档
 * - style: 格式
 * - refactor: 重构
 * - perf: 性能
 * - test: 测试
 * - chore: 构建/工具
 */

const commitlintConfigExample = `
// commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      ['feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'chore', 'revert'],
    ],
    'scope-case': [2, 'always', 'lower-case'],
    'subject-case': [0], // 允许中文
    'subject-max-length': [2, 'always', 72],
  },
};

// husky + lint-staged
// package.json
{
  "scripts": {
    "prepare": "husky install"
  },
  "lint-staged": {
    "*.{ts,tsx}": ["eslint --fix", "prettier --write"],
    "*.{css,scss}": ["stylelint --fix", "prettier --write"],
    "*.{json,md}": ["prettier --write"]
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

// ============================================================
// 3. Code Review
// ============================================================

/**
 * 📊 Code Review 最佳实践
 *
 * 关注点：
 * 1. 正确性：逻辑是否正确
 * 2. 设计：架构是否合理
 * 3. 可读性：代码是否清晰
 * 4. 安全性：是否有安全隐患
 * 5. 性能：是否有性能问题
 *
 * 流程：
 * 1. 自我 Review
 * 2. 自动检查通过
 * 3. 指定 Reviewer
 * 4. 讨论修改
 * 5. 批准合并
 */

const codeReviewChecklist = `
// Code Review Checklist

## 功能
- [ ] 代码是否实现了需求
- [ ] 边界情况是否处理
- [ ] 错误处理是否完善

## 设计
- [ ] 模块划分是否合理
- [ ] 是否遵循 DRY 原则
- [ ] 是否过度设计

## 代码质量
- [ ] 命名是否清晰
- [ ] 注释是否必要且准确
- [ ] 复杂度是否可接受

## 性能
- [ ] 是否有不必要的重渲染
- [ ] 是否有内存泄漏风险
- [ ] 是否有 N+1 查询

## 安全
- [ ] 用户输入是否验证
- [ ] 是否有 XSS 风险
- [ ] 敏感信息是否暴露

## 测试
- [ ] 是否有测试覆盖
- [ ] 测试用例是否充分
`;

// GitHub CODEOWNERS
const codeownersExample = `
# .github/CODEOWNERS

# 默认 Owner
* @team-lead

# 按目录指定
/src/components/ @frontend-team
/src/api/ @backend-team
/src/utils/ @core-team

# 按文件类型
*.ts @typescript-reviewers

# 敏感文件需要 Tech Lead 审批
package.json @tech-lead
tsconfig.json @tech-lead
`;

// ============================================================
// 4. 质量门禁
// ============================================================

/**
 * 📊 质量门禁配置
 *
 * 阻止不合格代码合并
 */

const qualityGateExample = `
// GitHub Actions 质量门禁
name: Quality Gate

on:
  pull_request:
    branches: [main, develop]

jobs:
  quality-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      # 1. 代码风格检查
      - name: Lint
        run: pnpm lint
      
      # 2. 类型检查
      - name: Type Check
        run: pnpm type-check
      
      # 3. 单元测试 + 覆盖率
      - name: Test
        run: pnpm test:coverage
      
      # 4. 覆盖率门禁
      - name: Coverage Gate
        uses: codecov/codecov-action@v3
        with:
          fail_ci_if_error: true
          # 覆盖率低于 80% 失败
          
      # 5. 构建检查
      - name: Build
        run: pnpm build
      
      # 6. Bundle 大小检查
      - name: Bundle Size
        uses: preactjs/compressed-size-action@v2
        with:
          repo-token: "\${{ secrets.GITHUB_TOKEN }}"
          # 增加超过 10KB 警告

// 分支保护规则
// Settings → Branches → Branch protection rules
// - Require status checks to pass
// - Require review from Code Owners
// - Require linear history
`;

// SonarQube 集成
const sonarqubeExample = `
// sonar-project.properties
sonar.projectKey=my-project
sonar.organization=my-org
sonar.sources=src
sonar.tests=src
sonar.test.inclusions=**/*.test.ts,**/*.spec.ts
sonar.typescript.lcov.reportPaths=coverage/lcov.info
sonar.coverage.exclusions=**/*.test.ts,**/*.spec.ts

// 质量门禁规则
// - 新代码覆盖率 >= 80%
// - 新代码重复率 <= 3%
// - 新代码 Bug 数 = 0
// - 新代码漏洞数 = 0
// - 新代码异味数 <= 10
`;

// ============================================================
// 5. 复杂度管理
// ============================================================

/**
 * 📊 代码复杂度指标
 *
 * - 圈复杂度（Cyclomatic Complexity）
 * - 认知复杂度（Cognitive Complexity）
 * - 代码行数
 * - 依赖数量
 */

const complexityRulesExample = `
// ESLint 复杂度规则
{
  "rules": {
    // 圈复杂度 <= 10
    "complexity": ["error", { "max": 10 }],
    
    // 函数最大行数
    "max-lines-per-function": ["warn", { "max": 50 }],
    
    // 文件最大行数
    "max-lines": ["warn", { "max": 300 }],
    
    // 最大嵌套深度
    "max-depth": ["error", { "max": 4 }],
    
    // 最大回调嵌套
    "max-nested-callbacks": ["error", { "max": 3 }],
    
    // 函数最大参数
    "max-params": ["warn", { "max": 4 }]
  }
}

// 复杂度分析工具
// 1. plato - 可视化复杂度报告
npx plato -r -d report src

// 2. madge - 依赖分析
npx madge --circular src
npx madge --image graph.svg src
`;

// ============================================================
// 6. ⚠️ 注意事项（易错点）
// ============================================================

/**
 * ⚠️ 常见问题
 *
 * 1. 规范过严
 *    - 阻碍开发效率
 *    - 渐进式引入
 *
 * 2. 规范不落地
 *    - 只有文档没有工具
 *    - 自动化检查
 *
 * 3. Code Review 形式化
 *    - LGTM 敷衍了事
 *    - 明确 Review 标准
 *
 * 4. 门禁过松或过紧
 *    - 过松：质量问题流出
 *    - 过紧：影响效率
 */

// ============================================================
// 7. 💡 面试追问
// ============================================================

/**
 * 💡 深度追问
 *
 * Q1: 如何保证代码质量？
 * A:
 *    - 代码规范（ESLint/Prettier）
 *    - Git 规范（commitlint）
 *    - Code Review
 *    - 自动化测试
 *    - 质量门禁
 *
 * Q2: ESLint 和 Prettier 的区别？
 * A:
 *    ESLint：代码质量（错误、最佳实践）
 *    Prettier：代码风格（格式化）
 *    配合使用，eslint-config-prettier 关闭冲突
 *
 * Q3: 如何做好 Code Review？
 * A:
 *    - 明确 Review 标准
 *    - 小批量 PR
 *    - 及时 Review
 *    - 建设性反馈
 *
 * Q4: 什么是圈复杂度？
 * A:
 *    - 代码路径数量
 *    - if/else/for 等增加复杂度
 *    - 一般建议 <= 10
 */

export {
  eslintConfigExample,
  prettierConfigExample,
  commitlintConfigExample,
  codeReviewChecklist,
  codeownersExample,
  qualityGateExample,
  sonarqubeExample,
  complexityRulesExample,
};

