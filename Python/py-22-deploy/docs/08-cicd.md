# CI/CD 实战

> GitHub Actions / GitLab CI 自动化部署

## CI/CD 概述

```
代码提交 → 自动测试 → 构建 → 部署

CI (持续集成): 代码提交后自动测试
CD (持续部署): 测试通过后自动部署
```

---

## GitHub Actions

### 项目结构

```
.github/
└── workflows/
    ├── test.yml        # 测试工作流
    ├── lint.yml        # 代码检查
    └── deploy.yml      # 部署工作流
```

### 测试工作流

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync

      - name: Run tests
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379
        run: |
          uv run pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
```

### 代码检查工作流

```yaml
# .github/workflows/lint.yml
name: Lint

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install ruff pyright

      - name: Run ruff
        run: ruff check .

      - name: Run ruff format check
        run: ruff format --check .

      - name: Run pyright
        run: pyright
```

### 部署工作流

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]
    tags:
      - 'v*'

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - name: Deploy to server
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SERVER_SSH_KEY }}
          script: |
            cd /app
            docker compose pull
            docker compose up -d
```

### 多环境部署

```yaml
# .github/workflows/deploy-multi-env.yml
name: Deploy Multi Environment

on:
  push:
    branches:
      - main
      - develop
  release:
    types: [published]

jobs:
  deploy-staging:
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to Staging
        env:
          DEPLOY_URL: ${{ secrets.STAGING_URL }}
        run: |
          # 部署到 staging

  deploy-production:
    if: github.event_name == 'release'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to Production
        env:
          DEPLOY_URL: ${{ secrets.PRODUCTION_URL }}
        run: |
          # 部署到 production
```

---

## GitLab CI

### 基础配置

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip-cache"

cache:
  paths:
    - .pip-cache/
    - .venv/

test:
  stage: test
  image: python:3.12
  services:
    - postgres:15
    - redis:7
  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: test
    POSTGRES_PASSWORD: test
    DATABASE_URL: postgresql://test:test@postgres:5432/test_db
    REDIS_URL: redis://redis:6379
  before_script:
    - pip install uv
    - uv sync
  script:
    - uv run pytest --cov=src
  coverage: '/TOTAL.*\s+(\d+%)$/'

lint:
  stage: test
  image: python:3.12
  before_script:
    - pip install ruff pyright
  script:
    - ruff check .
    - ruff format --check .
    - pyright

build:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
    - tags

deploy_staging:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache openssh-client
    - eval $(ssh-agent -s)
    - echo "$SSH_PRIVATE_KEY" | ssh-add -
  script:
    - ssh -o StrictHostKeyChecking=no $SERVER_USER@$STAGING_SERVER "
        cd /app &&
        docker compose pull &&
        docker compose up -d
      "
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - develop

deploy_production:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache openssh-client
    - eval $(ssh-agent -s)
    - echo "$SSH_PRIVATE_KEY" | ssh-add -
  script:
    - ssh -o StrictHostKeyChecking=no $SERVER_USER@$PROD_SERVER "
        cd /app &&
        docker compose pull &&
        docker compose up -d
      "
  environment:
    name: production
    url: https://example.com
  only:
    - tags
  when: manual
```

---

## 自动化版本管理

### Semantic Release

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    branches: [main]

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      issues: write
      pull-requests: write

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install dependencies
        run: npm install -g semantic-release @semantic-release/git @semantic-release/changelog

      - name: Release
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: npx semantic-release
```

```json
// .releaserc.json
{
  "branches": ["main"],
  "plugins": [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    "@semantic-release/changelog",
    "@semantic-release/github",
    [
      "@semantic-release/git",
      {
        "assets": ["CHANGELOG.md", "pyproject.toml"],
        "message": "chore(release): ${nextRelease.version}"
      }
    ]
  ]
}
```

### Commit 规范

```
feat: 新功能
fix: 修复 bug
docs: 文档更新
style: 代码格式
refactor: 重构
perf: 性能优化
test: 测试
chore: 构建/工具
```

---

## 完整 Python 项目配置

### pyproject.toml

```toml
[project]
name = "myproject"
version = "0.1.0"
requires-python = ">=3.12"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pyright]
pythonVersion = "3.12"
typeCheckingMode = "strict"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

### Makefile

```makefile
.PHONY: install test lint format type-check ci

install:
	uv sync

test:
	uv run pytest --cov=src

lint:
	ruff check .

format:
	ruff format .

type-check:
	pyright

ci: lint type-check test
	@echo "All checks passed!"
```

---

## 部署策略

### 蓝绿部署

```yaml
# docker-compose.blue-green.yml
version: '3.8'

services:
  app-blue:
    image: myapp:${BLUE_VERSION}
    ports:
      - "8001:8000"

  app-green:
    image: myapp:${GREEN_VERSION}
    ports:
      - "8002:8000"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### 滚动更新

```yaml
# docker-compose.yml with Docker Swarm
version: '3.8'

services:
  app:
    image: myapp:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      rollback_config:
        parallelism: 1
        delay: 10s
```

---

## 常见坑

| 坑 | 说明 | 正确做法 |
|----|------|---------|
| 密钥泄露 | 硬编码在代码中 | 使用 Secrets |
| 缓存未命中 | 依赖每次重装 | 配置正确的缓存路径 |
| 并行冲突 | 多个部署同时运行 | 使用 concurrency |
| 测试不稳定 | 随机失败 | 隔离测试环境 |
| 版本混乱 | 不知道部署了什么 | 使用 Git tag |

---

## 小结

1. **GitHub Actions**：`.github/workflows/*.yml`
2. **GitLab CI**：`.gitlab-ci.yml`
3. **阶段**：test → build → deploy
4. **多环境**：staging / production
5. **版本管理**：semantic-release + 规范 commit

