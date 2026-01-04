#!/usr/bin/env python3
"""
文件整理器 - 按扩展名分类文件

功能：
1. 扫描指定目录
2. 按扩展名分类文件
3. 移动或复制文件到分类目录
4. 支持重命名重复文件
5. 生成整理报告

使用方法：
    python main.py <source_dir> [options]
    python main.py ./downloads --target ./organized --move
    python main.py ./downloads --dry-run
"""

import argparse
import logging
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# 文件类型分类映射
FILE_CATEGORIES = {
    "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".ico"],
    "documents": [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".rtf"],
    "videos": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"],
    "audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a"],
    "archives": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"],
    "code": [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".css", ".html", ".json"],
    "data": [".csv", ".xml", ".yaml", ".yml", ".sql", ".db"],
}


@dataclass
class FileInfo:
    """文件信息"""
    path: Path
    name: str
    extension: str
    size: int
    category: str

    @classmethod
    def from_path(cls, path: Path) -> "FileInfo":
        extension = path.suffix.lower()
        category = get_category(extension)
        return cls(
            path=path,
            name=path.name,
            extension=extension,
            size=path.stat().st_size,
            category=category
        )


@dataclass
class OrganizeResult:
    """整理结果"""
    total_files: int = 0
    moved_files: int = 0
    skipped_files: int = 0
    errors: list[str] = field(default_factory=list)
    by_category: dict[str, int] = field(default_factory=Counter)
    total_size: int = 0


def get_category(extension: str) -> str:
    """根据扩展名获取分类"""
    for category, extensions in FILE_CATEGORIES.items():
        if extension in extensions:
            return category
    return "others"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)


def scan_directory(source: Path, recursive: bool = False) -> list[FileInfo]:
    """扫描目录获取文件列表"""
    files = []

    if recursive:
        file_paths = source.rglob("*")
    else:
        file_paths = source.iterdir()

    for path in file_paths:
        if path.is_file():
            try:
                files.append(FileInfo.from_path(path))
            except Exception as e:
                logging.warning(f"无法读取文件 {path}: {e}")

    return files


def get_unique_path(target: Path) -> Path:
    """获取唯一路径（处理重名）"""
    if not target.exists():
        return target

    counter = 1
    stem = target.stem
    suffix = target.suffix
    parent = target.parent

    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1


def organize_files(
    source: Path,
    target: Optional[Path] = None,
    move: bool = False,
    recursive: bool = False,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None
) -> OrganizeResult:
    """整理文件"""
    if logger is None:
        logger = logging.getLogger(__name__)

    result = OrganizeResult()

    # 目标目录默认为源目录下的 organized 子目录
    if target is None:
        target = source / "organized"

    # 扫描文件
    files = scan_directory(source, recursive)
    result.total_files = len(files)

    logger.info(f"扫描到 {len(files)} 个文件")

    for file_info in files:
        try:
            # 目标路径
            category_dir = target / file_info.category
            target_path = category_dir / file_info.name

            # 处理重名
            target_path = get_unique_path(target_path)

            if dry_run:
                logger.debug(f"[DRY-RUN] {file_info.path} -> {target_path}")
            else:
                # 创建目录
                category_dir.mkdir(parents=True, exist_ok=True)

                # 移动或复制
                if move:
                    shutil.move(str(file_info.path), str(target_path))
                    logger.debug(f"移动: {file_info.name} -> {file_info.category}/")
                else:
                    shutil.copy2(str(file_info.path), str(target_path))
                    logger.debug(f"复制: {file_info.name} -> {file_info.category}/")

            result.moved_files += 1
            result.by_category[file_info.category] += 1
            result.total_size += file_info.size

        except Exception as e:
            result.skipped_files += 1
            result.errors.append(f"{file_info.name}: {e}")
            logger.error(f"处理失败 {file_info.name}: {e}")

    return result


def generate_report(result: OrganizeResult, output_path: Optional[Path] = None) -> str:
    """生成报告"""
    lines = [
        "=" * 50,
        "文件整理报告",
        "=" * 50,
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "统计:",
        f"  总文件数: {result.total_files}",
        f"  已处理: {result.moved_files}",
        f"  跳过: {result.skipped_files}",
        f"  总大小: {result.total_size / 1024 / 1024:.2f} MB",
        "",
        "按分类:",
    ]

    for category, count in sorted(result.by_category.items()):
        lines.append(f"  {category}: {count} 个文件")

    if result.errors:
        lines.extend([
            "",
            "错误:",
        ])
        for error in result.errors[:10]:  # 只显示前 10 个错误
            lines.append(f"  - {error}")
        if len(result.errors) > 10:
            lines.append(f"  ... 共 {len(result.errors)} 个错误")

    lines.append("=" * 50)

    report = "\n".join(lines)

    if output_path:
        output_path.write_text(report, encoding="utf-8")

    return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="文件整理器 - 按扩展名分类文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s ./downloads                    # 复制文件到 ./downloads/organized
  %(prog)s ./downloads --target ./sorted  # 复制到指定目录
  %(prog)s ./downloads --move             # 移动文件（而非复制）
  %(prog)s ./downloads --recursive        # 递归处理子目录
  %(prog)s ./downloads --dry-run          # 预览模式，不实际操作
        """
    )

    parser.add_argument("source", type=Path, help="源目录")
    parser.add_argument("-t", "--target", type=Path, help="目标目录")
    parser.add_argument("-m", "--move", action="store_true", help="移动文件（默认复制）")
    parser.add_argument("-r", "--recursive", action="store_true", help="递归处理子目录")
    parser.add_argument("-n", "--dry-run", action="store_true", help="预览模式")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("--report", type=Path, help="报告输出路径")

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging(args.verbose)

    # 检查源目录
    if not args.source.exists():
        logger.error(f"源目录不存在: {args.source}")
        return 1

    if not args.source.is_dir():
        logger.error(f"不是目录: {args.source}")
        return 1

    # 执行整理
    action = "移动" if args.move else "复制"
    if args.dry_run:
        action = f"[预览] {action}"

    logger.info(f"开始整理: {args.source}")
    logger.info(f"操作模式: {action}")

    result = organize_files(
        source=args.source,
        target=args.target,
        move=args.move,
        recursive=args.recursive,
        dry_run=args.dry_run,
        logger=logger
    )

    # 生成报告
    report = generate_report(result, args.report)
    print(report)

    if args.report:
        logger.info(f"报告已保存到: {args.report}")

    return 0


if __name__ == "__main__":
    exit(main())


