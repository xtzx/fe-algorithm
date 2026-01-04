# 常见自动化任务

> 实战：批量重命名、文件分类、格式转换、数据迁移

## 1. 批量重命名

### 1.1 正则替换

```python
from pathlib import Path
import re
from dataclasses import dataclass


@dataclass
class RenameResult:
    old_name: str
    new_name: str
    success: bool
    error: str | None = None


def batch_regex_rename(
    directory: Path,
    pattern: str,
    replacement: str,
    file_glob: str = "*",
    dry_run: bool = True,
) -> list[RenameResult]:
    """
    批量正则重命名

    Args:
        directory: 目标目录
        pattern: 正则表达式
        replacement: 替换字符串（支持 \1, \2 等分组引用）
        file_glob: 文件过滤
        dry_run: 预览模式
    """
    results: list[RenameResult] = []
    regex = re.compile(pattern)

    for file_path in sorted(directory.glob(file_glob)):
        if not file_path.is_file():
            continue

        old_name = file_path.name
        new_name = regex.sub(replacement, old_name)

        if old_name == new_name:
            continue  # 无需重命名

        new_path = file_path.parent / new_name

        if dry_run:
            print(f"  [DRY-RUN] {old_name} → {new_name}")
            results.append(RenameResult(old_name, new_name, True))
            continue

        try:
            if new_path.exists():
                raise FileExistsError(f"目标已存在: {new_name}")
            file_path.rename(new_path)
            results.append(RenameResult(old_name, new_name, True))
            print(f"  ✓ {old_name} → {new_name}")
        except Exception as e:
            results.append(RenameResult(old_name, new_name, False, str(e)))
            print(f"  ✗ {old_name}: {e}")

    return results


# 使用示例
# report_20240101.pdf → 2024-01-01_report.pdf
results = batch_regex_rename(
    Path("./reports"),
    pattern=r"report_(\d{4})(\d{2})(\d{2})\.pdf",
    replacement=r"\1-\2-\3_report.pdf",
    dry_run=True,
)
```

### 1.2 序号重命名

```python
def batch_sequential_rename(
    directory: Path,
    prefix: str,
    file_glob: str = "*",
    start: int = 1,
    width: int = 3,
    sort_key: str = "name",  # name, date, size
    dry_run: bool = True,
) -> list[RenameResult]:
    """
    批量序号重命名

    例: photo_001.jpg, photo_002.jpg, ...
    """
    results: list[RenameResult] = []
    files = list(directory.glob(file_glob))

    # 排序
    if sort_key == "name":
        files.sort(key=lambda f: f.name)
    elif sort_key == "date":
        files.sort(key=lambda f: f.stat().st_mtime)
    elif sort_key == "size":
        files.sort(key=lambda f: f.stat().st_size)

    # 只处理文件
    files = [f for f in files if f.is_file()]

    for i, file_path in enumerate(files, start=start):
        old_name = file_path.name
        suffix = file_path.suffix
        new_name = f"{prefix}{i:0{width}d}{suffix}"

        if old_name == new_name:
            continue

        new_path = file_path.parent / new_name

        if dry_run:
            print(f"  [DRY-RUN] {old_name} → {new_name}")
            results.append(RenameResult(old_name, new_name, True))
            continue

        try:
            # 使用临时名称避免冲突
            temp_path = file_path.parent / f"__temp_{i}_{suffix}"
            file_path.rename(temp_path)
            temp_path.rename(new_path)
            results.append(RenameResult(old_name, new_name, True))
        except Exception as e:
            results.append(RenameResult(old_name, new_name, False, str(e)))

    return results


# 使用示例
results = batch_sequential_rename(
    Path("./photos"),
    prefix="vacation_2024_",
    file_glob="*.jpg",
    sort_key="date",
    dry_run=True,
)
```

### 1.3 日期时间重命名

```python
from datetime import datetime


def batch_datetime_rename(
    directory: Path,
    file_glob: str = "*",
    date_format: str = "%Y%m%d_%H%M%S",
    use_exif: bool = False,  # 图片可使用 EXIF 日期
    dry_run: bool = True,
) -> list[RenameResult]:
    """
    根据文件修改时间重命名
    """
    results: list[RenameResult] = []

    for file_path in sorted(directory.glob(file_glob)):
        if not file_path.is_file():
            continue

        old_name = file_path.name
        suffix = file_path.suffix

        # 获取文件时间
        mtime = file_path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime)

        # 生成新名称
        date_str = dt.strftime(date_format)
        new_name = f"{date_str}{suffix}"

        # 处理重名
        counter = 1
        new_path = file_path.parent / new_name
        while new_path.exists() and new_path != file_path:
            new_name = f"{date_str}_{counter}{suffix}"
            new_path = file_path.parent / new_name
            counter += 1

        if old_name == new_name:
            continue

        if dry_run:
            print(f"  [DRY-RUN] {old_name} → {new_name}")
            results.append(RenameResult(old_name, new_name, True))
        else:
            try:
                file_path.rename(new_path)
                results.append(RenameResult(old_name, new_name, True))
            except Exception as e:
                results.append(RenameResult(old_name, new_name, False, str(e)))

    return results
```

## 2. 文件分类整理

### 2.1 按扩展名分类

```python
from collections import defaultdict


# 文件类型映射
FILE_CATEGORIES = {
    "images": [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp"],
    "documents": [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".md"],
    "videos": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"],
    "audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"],
    "archives": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"],
    "code": [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"],
    "data": [".json", ".xml", ".csv", ".yaml", ".yml", ".toml"],
}


def get_category(suffix: str) -> str:
    """获取文件类别"""
    suffix_lower = suffix.lower()
    for category, extensions in FILE_CATEGORIES.items():
        if suffix_lower in extensions:
            return category
    return "others"


def organize_by_extension(
    source_dir: Path,
    target_dir: Path | None = None,
    dry_run: bool = True,
) -> dict[str, list[Path]]:
    """
    按扩展名分类文件

    Args:
        source_dir: 源目录
        target_dir: 目标目录（默认在源目录下创建子目录）
        dry_run: 预览模式
    """
    if target_dir is None:
        target_dir = source_dir

    # 收集文件
    categorized: dict[str, list[Path]] = defaultdict(list)

    for file_path in source_dir.iterdir():
        if not file_path.is_file():
            continue

        category = get_category(file_path.suffix)
        categorized[category].append(file_path)

    # 执行分类
    for category, files in categorized.items():
        category_dir = target_dir / category

        if not dry_run:
            category_dir.mkdir(exist_ok=True)

        for file_path in files:
            new_path = category_dir / file_path.name

            if dry_run:
                print(f"  [DRY-RUN] {file_path.name} → {category}/")
            else:
                try:
                    shutil.move(file_path, new_path)
                    print(f"  ✓ {file_path.name} → {category}/")
                except Exception as e:
                    print(f"  ✗ {file_path.name}: {e}")

    return dict(categorized)


# 使用
result = organize_by_extension(Path("./downloads"), dry_run=True)
print(f"\n分类统计:")
for category, files in result.items():
    print(f"  {category}: {len(files)} 个文件")
```

### 2.2 按日期分类

```python
def organize_by_date(
    source_dir: Path,
    target_dir: Path | None = None,
    date_format: str = "%Y/%Y-%m",  # 2024/2024-01
    dry_run: bool = True,
) -> dict[str, list[Path]]:
    """
    按修改日期分类文件
    """
    if target_dir is None:
        target_dir = source_dir

    categorized: dict[str, list[Path]] = defaultdict(list)

    for file_path in source_dir.iterdir():
        if not file_path.is_file():
            continue

        # 获取文件日期
        mtime = file_path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime)
        date_str = dt.strftime(date_format)

        categorized[date_str].append(file_path)

    # 执行分类
    for date_str, files in categorized.items():
        date_dir = target_dir / date_str

        if not dry_run:
            date_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            new_path = date_dir / file_path.name

            if dry_run:
                print(f"  [DRY-RUN] {file_path.name} → {date_str}/")
            else:
                try:
                    shutil.move(file_path, new_path)
                except Exception as e:
                    print(f"  ✗ {file_path.name}: {e}")

    return dict(categorized)
```

### 2.3 按大小分类

```python
def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def get_size_category(size_bytes: int) -> str:
    """获取大小类别"""
    if size_bytes < 1024 * 100:  # < 100KB
        return "tiny"
    elif size_bytes < 1024 * 1024:  # < 1MB
        return "small"
    elif size_bytes < 1024 * 1024 * 10:  # < 10MB
        return "medium"
    elif size_bytes < 1024 * 1024 * 100:  # < 100MB
        return "large"
    else:  # >= 100MB
        return "huge"


def organize_by_size(
    source_dir: Path,
    target_dir: Path | None = None,
    dry_run: bool = True,
) -> dict[str, list[tuple[Path, int]]]:
    """
    按文件大小分类
    """
    if target_dir is None:
        target_dir = source_dir

    categorized: dict[str, list[tuple[Path, int]]] = defaultdict(list)

    for file_path in source_dir.iterdir():
        if not file_path.is_file():
            continue

        size = file_path.stat().st_size
        category = get_size_category(size)
        categorized[category].append((file_path, size))

    # 执行分类
    for category, files in categorized.items():
        category_dir = target_dir / category

        if not dry_run:
            category_dir.mkdir(exist_ok=True)

        for file_path, size in files:
            new_path = category_dir / file_path.name

            if dry_run:
                print(f"  [DRY-RUN] {file_path.name} ({format_size(size)}) → {category}/")
            else:
                try:
                    shutil.move(file_path, new_path)
                except Exception as e:
                    print(f"  ✗ {file_path.name}: {e}")

    return dict(categorized)
```

## 3. 批量格式转换

### 3.1 文本编码转换

```python
import chardet  # pip install chardet


def detect_encoding(file_path: Path) -> str:
    """检测文件编码"""
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"] or "utf-8"


def batch_convert_encoding(
    directory: Path,
    target_encoding: str = "utf-8",
    file_glob: str = "*.txt",
    dry_run: bool = True,
) -> list[dict]:
    """
    批量转换文本编码
    """
    results = []

    for file_path in directory.glob(file_glob):
        if not file_path.is_file():
            continue

        # 检测原编码
        source_encoding = detect_encoding(file_path)

        if source_encoding.lower() == target_encoding.lower():
            continue  # 已经是目标编码

        result = {
            "file": str(file_path),
            "from": source_encoding,
            "to": target_encoding,
            "success": False,
        }

        if dry_run:
            print(f"  [DRY-RUN] {file_path.name}: {source_encoding} → {target_encoding}")
            result["success"] = True
            results.append(result)
            continue

        try:
            # 读取原内容
            with open(file_path, "r", encoding=source_encoding) as f:
                content = f.read()

            # 写入新编码
            with open(file_path, "w", encoding=target_encoding) as f:
                f.write(content)

            result["success"] = True
            print(f"  ✓ {file_path.name}: {source_encoding} → {target_encoding}")
        except Exception as e:
            result["error"] = str(e)
            print(f"  ✗ {file_path.name}: {e}")

        results.append(result)

    return results
```

### 3.2 行尾符转换

```python
def batch_convert_line_endings(
    directory: Path,
    target: str = "lf",  # lf, crlf, cr
    file_glob: str = "*.txt",
    dry_run: bool = True,
) -> list[dict]:
    """
    批量转换行尾符

    - lf: Unix/Linux/macOS (\n)
    - crlf: Windows (\r\n)
    - cr: 旧 Mac (\r)
    """
    endings = {"lf": "\n", "crlf": "\r\n", "cr": "\r"}
    target_ending = endings[target.lower()]

    results = []

    for file_path in directory.glob(file_glob):
        if not file_path.is_file():
            continue

        try:
            content = file_path.read_bytes()

            # 检测当前行尾
            has_crlf = b"\r\n" in content
            has_lf = b"\n" in content and not has_crlf
            has_cr = b"\r" in content and b"\r\n" not in content

            current = "crlf" if has_crlf else ("lf" if has_lf else ("cr" if has_cr else "unknown"))

            if current == target.lower():
                continue

            result = {
                "file": str(file_path),
                "from": current,
                "to": target,
                "success": False,
            }

            if dry_run:
                print(f"  [DRY-RUN] {file_path.name}: {current} → {target}")
                result["success"] = True
                results.append(result)
                continue

            # 统一转换为 LF，再转为目标格式
            text = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            if target.lower() == "crlf":
                text = text.replace(b"\n", b"\r\n")
            elif target.lower() == "cr":
                text = text.replace(b"\n", b"\r")

            file_path.write_bytes(text)
            result["success"] = True
            print(f"  ✓ {file_path.name}: {current} → {target}")

        except Exception as e:
            result = {"file": str(file_path), "error": str(e), "success": False}

        results.append(result)

    return results
```

## 4. 数据迁移

### 4.1 目录同步

```python
import hashlib


def file_hash(file_path: Path) -> str:
    """计算文件 MD5 哈希"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sync_directories(
    source: Path,
    target: Path,
    delete_extra: bool = False,  # 删除目标中多余的文件
    dry_run: bool = True,
) -> dict:
    """
    同步两个目录
    """
    stats = {
        "copied": 0,
        "updated": 0,
        "deleted": 0,
        "skipped": 0,
    }

    # 收集源文件
    source_files: dict[str, Path] = {}
    for file_path in source.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(source)
            source_files[str(rel_path)] = file_path

    # 收集目标文件
    target_files: dict[str, Path] = {}
    if target.exists():
        for file_path in target.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(target)
                target_files[str(rel_path)] = file_path

    # 复制/更新
    for rel_path, src_file in source_files.items():
        dst_file = target / rel_path

        if str(rel_path) in target_files:
            # 检查是否需要更新
            if file_hash(src_file) == file_hash(dst_file):
                stats["skipped"] += 1
                continue
            action = "UPDATE"
            stats["updated"] += 1
        else:
            action = "COPY"
            stats["copied"] += 1

        if dry_run:
            print(f"  [DRY-RUN] {action}: {rel_path}")
        else:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            print(f"  ✓ {action}: {rel_path}")

    # 删除多余文件
    if delete_extra:
        extra_files = set(target_files.keys()) - set(source_files.keys())
        for rel_path in extra_files:
            file_path = target / rel_path

            if dry_run:
                print(f"  [DRY-RUN] DELETE: {rel_path}")
            else:
                file_path.unlink()
                print(f"  ✓ DELETE: {rel_path}")

            stats["deleted"] += 1

    return stats
```

### 4.2 增量备份

```python
from datetime import datetime


def incremental_backup(
    source: Path,
    backup_root: Path,
    dry_run: bool = True,
) -> dict:
    """
    增量备份

    目录结构:
    backup_root/
    ├── latest/          # 符号链接指向最新备份
    ├── 2024-01-01_120000/
    ├── 2024-01-02_120000/
    └── ...
    """
    # 创建带时间戳的备份目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup_dir = backup_root / timestamp

    # 获取上一次备份
    latest_link = backup_root / "latest"
    previous_backup = latest_link.resolve() if latest_link.exists() else None

    stats = {"new": 0, "unchanged": 0, "hardlinked": 0}

    if dry_run:
        print(f"备份目录: {backup_dir}")
        if previous_backup:
            print(f"增量基于: {previous_backup}")
    else:
        backup_dir.mkdir(parents=True, exist_ok=True)

    for file_path in source.rglob("*"):
        if not file_path.is_file():
            continue

        rel_path = file_path.relative_to(source)
        backup_file = backup_dir / rel_path

        # 检查是否可以硬链接（文件未修改）
        if previous_backup:
            prev_file = previous_backup / rel_path
            if prev_file.exists():
                if file_hash(file_path) == file_hash(prev_file):
                    # 创建硬链接而非复制
                    if dry_run:
                        print(f"  [DRY-RUN] LINK: {rel_path}")
                    else:
                        backup_file.parent.mkdir(parents=True, exist_ok=True)
                        backup_file.hardlink_to(prev_file)
                    stats["hardlinked"] += 1
                    continue

        # 复制文件
        if dry_run:
            print(f"  [DRY-RUN] COPY: {rel_path}")
        else:
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_file)
        stats["new"] += 1

    # 更新 latest 链接
    if not dry_run:
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(backup_dir)

    return stats
```

## 5. 清理任务

### 5.1 清理空目录

```python
def remove_empty_directories(
    root: Path,
    dry_run: bool = True,
) -> list[Path]:
    """
    递归删除空目录
    """
    removed = []

    # 从最深层开始（后序遍历）
    for dirpath in sorted(root.rglob("*"), key=lambda p: -len(p.parts)):
        if not dirpath.is_dir():
            continue

        # 检查是否为空
        if not any(dirpath.iterdir()):
            if dry_run:
                print(f"  [DRY-RUN] 删除空目录: {dirpath}")
            else:
                dirpath.rmdir()
                print(f"  ✓ 删除空目录: {dirpath}")
            removed.append(dirpath)

    return removed
```

### 5.2 清理临时文件

```python
TEMP_PATTERNS = [
    "*.tmp",
    "*.temp",
    "*.bak",
    "*.swp",
    "*~",
    ".DS_Store",
    "Thumbs.db",
    "__pycache__",
    "*.pyc",
    "node_modules",
    ".git",
]


def clean_temp_files(
    root: Path,
    patterns: list[str] | None = None,
    dry_run: bool = True,
) -> dict:
    """
    清理临时文件
    """
    if patterns is None:
        patterns = TEMP_PATTERNS

    stats = {"files": 0, "directories": 0, "size_freed": 0}

    for pattern in patterns:
        for path in root.rglob(pattern):
            size = 0
            if path.is_file():
                size = path.stat().st_size
                if dry_run:
                    print(f"  [DRY-RUN] 删除文件: {path} ({format_size(size)})")
                else:
                    path.unlink()
                stats["files"] += 1
            elif path.is_dir():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                if dry_run:
                    print(f"  [DRY-RUN] 删除目录: {path} ({format_size(size)})")
                else:
                    shutil.rmtree(path)
                stats["directories"] += 1

            stats["size_freed"] += size

    return stats
```

### 5.3 清理旧文件

```python
from datetime import timedelta


def clean_old_files(
    root: Path,
    max_age: timedelta,
    file_glob: str = "*",
    dry_run: bool = True,
) -> list[Path]:
    """
    清理超过指定时间的旧文件
    """
    cutoff = datetime.now() - max_age
    removed = []

    for file_path in root.rglob(file_glob):
        if not file_path.is_file():
            continue

        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

        if mtime < cutoff:
            if dry_run:
                print(f"  [DRY-RUN] 删除: {file_path} (修改于 {mtime.date()})")
            else:
                file_path.unlink()
                print(f"  ✓ 删除: {file_path}")
            removed.append(file_path)

    return removed


# 使用：删除 30 天前的日志
old_files = clean_old_files(
    Path("./logs"),
    max_age=timedelta(days=30),
    file_glob="*.log",
    dry_run=True,
)
```

## 小结

| 任务类型 | 典型场景 |
|---------|---------|
| 批量重命名 | 正则替换、序号、日期时间 |
| 文件分类 | 按扩展名、日期、大小 |
| 格式转换 | 编码、行尾符 |
| 数据迁移 | 目录同步、增量备份 |
| 清理任务 | 空目录、临时文件、旧文件 |

所有任务都应该支持 **dry-run 模式**，让用户先预览再执行！

