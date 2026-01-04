# 脚本分发

## 1. ZipApp

Python 内置的打包方式，将应用打包为单个 `.pyz` 文件。

### 1.1 基础使用

```bash
# 目录结构
myapp/
├── __main__.py  # 入口文件
└── app.py

# 打包
python -m zipapp myapp -o myapp.pyz

# 运行
python myapp.pyz
```

### 1.2 添加解释器指定

```bash
# 添加 shebang
python -m zipapp myapp -o myapp.pyz -p "/usr/bin/env python3"

# 可直接执行（Unix）
chmod +x myapp.pyz
./myapp.pyz
```

### 1.3 包含依赖

```bash
# 安装依赖到目录
pip install -r requirements.txt --target myapp/

# 打包
python -m zipapp myapp -o myapp.pyz
```

### 1.4 限制

- 不能包含 C 扩展
- 依赖需要预先安装或打包
- 适合简单脚本

## 2. Shiv

创建独立的 Python 应用程序。

### 2.1 安装

```bash
pip install shiv
```

### 2.2 使用

```bash
# 打包当前项目
shiv -c myapp -o myapp.pyz .

# 打包指定入口
shiv -e myapp.cli:main -o myapp.pyz .

# 指定 Python 版本
shiv -p "/usr/bin/python3.11" -c myapp -o myapp.pyz .
```

### 2.3 特点

- 自动处理依赖
- 支持 C 扩展（解压运行）
- 缓存机制

## 3. PEX

Python EXecutable，类似 Shiv。

### 3.1 安装

```bash
pip install pex
```

### 3.2 使用

```bash
# 创建 PEX
pex . -c myapp -o myapp.pex

# 指定依赖
pex requests click -c mycli -o mycli.pex

# 从 requirements.txt
pex -r requirements.txt -c myapp -o myapp.pex
```

## 4. PyInstaller

创建真正的独立可执行文件。

### 4.1 安装

```bash
pip install pyinstaller
```

### 4.2 基础使用

```bash
# 单文件打包
pyinstaller --onefile main.py

# 带图标
pyinstaller --onefile --icon=app.ico main.py

# 不显示控制台（GUI 应用）
pyinstaller --onefile --noconsole main.py
```

### 4.3 配置文件

`myapp.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('config/', 'config/')],  # 数据文件
    hiddenimports=['uvicorn', 'fastapi'],  # 隐式依赖
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='myapp',
    debug=False,
    strip=False,
    upx=True,  # 压缩
    console=True,
)
```

```bash
pyinstaller myapp.spec
```

### 4.4 跨平台

PyInstaller 只能在目标平台上打包：
- 在 Windows 上打包 Windows 版本
- 在 macOS 上打包 macOS 版本
- 在 Linux 上打包 Linux 版本

## 5. 方案对比

| 方案 | 独立性 | C 扩展 | 大小 | 复杂度 |
|------|--------|--------|------|--------|
| zipapp | 低 | ❌ | 小 | 简单 |
| shiv | 中 | ✅ | 中 | 简单 |
| pex | 中 | ✅ | 中 | 简单 |
| PyInstaller | 高 | ✅ | 大 | 复杂 |

## 6. 选择建议

- **内部工具/脚本**: zipapp 或 shiv
- **命令行工具分发**: pex 或 shiv
- **桌面应用**: PyInstaller
- **跨平台分发**: PyInstaller（各平台分别打包）

## 7. 示例：CLI 工具打包

```bash
# 项目结构
mycli/
├── mycli/
│   ├── __init__.py
│   ├── __main__.py
│   └── cli.py
├── pyproject.toml
└── README.md

# 使用 shiv 打包
pip install shiv
shiv -c mycli -o mycli.pyz .

# 测试
./mycli.pyz --help
```


