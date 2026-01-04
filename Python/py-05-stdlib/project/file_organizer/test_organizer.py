#!/usr/bin/env python3
"""文件整理器测试"""

import shutil
import tempfile
from pathlib import Path

from main import (
    FileInfo, OrganizeResult,
    get_category, scan_directory, organize_files, generate_report
)


def test_get_category():
    """测试分类函数"""
    print("测试 get_category...")

    assert get_category(".jpg") == "images"
    assert get_category(".png") == "images"
    assert get_category(".pdf") == "documents"
    assert get_category(".mp4") == "videos"
    assert get_category(".mp3") == "audio"
    assert get_category(".zip") == "archives"
    assert get_category(".py") == "code"
    assert get_category(".csv") == "data"
    assert get_category(".xyz") == "others"

    print("  ✓ get_category 测试通过")


def test_file_info():
    """测试 FileInfo"""
    print("测试 FileInfo...")

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"test content")
        temp_path = Path(f.name)

    try:
        info = FileInfo.from_path(temp_path)

        assert info.extension == ".txt"
        assert info.category == "documents"
        assert info.size > 0
        assert info.name == temp_path.name

        print("  ✓ FileInfo 测试通过")
    finally:
        temp_path.unlink()


def test_scan_directory():
    """测试目录扫描"""
    print("测试 scan_directory...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建测试文件
        (tmpdir / "test1.txt").write_text("content1")
        (tmpdir / "test2.py").write_text("content2")
        (tmpdir / "test3.jpg").write_bytes(b"fake image")

        # 创建子目录和文件
        subdir = tmpdir / "subdir"
        subdir.mkdir()
        (subdir / "test4.pdf").write_text("content4")

        # 非递归扫描
        files = scan_directory(tmpdir, recursive=False)
        assert len(files) == 3

        # 递归扫描
        files = scan_directory(tmpdir, recursive=True)
        assert len(files) == 4

        print("  ✓ scan_directory 测试通过")


def test_organize_files():
    """测试文件整理"""
    print("测试 organize_files...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        source = tmpdir / "source"
        target = tmpdir / "target"
        source.mkdir()

        # 创建测试文件
        (source / "photo.jpg").write_bytes(b"fake image")
        (source / "document.pdf").write_text("fake pdf")
        (source / "script.py").write_text("print('hello')")
        (source / "unknown.xyz").write_text("unknown")

        # 整理文件（复制模式）
        result = organize_files(source, target, move=False)

        assert result.total_files == 4
        assert result.moved_files == 4
        assert result.skipped_files == 0
        assert len(result.errors) == 0

        # 验证目标目录结构
        assert (target / "images" / "photo.jpg").exists()
        assert (target / "documents" / "document.pdf").exists()
        assert (target / "code" / "script.py").exists()
        assert (target / "others" / "unknown.xyz").exists()

        # 源文件仍存在（复制模式）
        assert (source / "photo.jpg").exists()

        print("  ✓ organize_files 测试通过")


def test_organize_files_move():
    """测试移动模式"""
    print("测试 organize_files (移动模式)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        source = tmpdir / "source"
        target = tmpdir / "target"
        source.mkdir()

        # 创建测试文件
        (source / "photo.jpg").write_bytes(b"fake image")

        # 整理文件（移动模式）
        result = organize_files(source, target, move=True)

        assert result.moved_files == 1

        # 验证移动
        assert (target / "images" / "photo.jpg").exists()
        assert not (source / "photo.jpg").exists()

        print("  ✓ organize_files (移动模式) 测试通过")


def test_dry_run():
    """测试预览模式"""
    print("测试 dry_run 模式...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        source = tmpdir / "source"
        target = tmpdir / "target"
        source.mkdir()

        # 创建测试文件
        (source / "photo.jpg").write_bytes(b"fake image")

        # 预览模式
        result = organize_files(source, target, dry_run=True)

        assert result.moved_files == 1

        # 文件未实际移动
        assert not target.exists()
        assert (source / "photo.jpg").exists()

        print("  ✓ dry_run 测试通过")


def test_duplicate_handling():
    """测试重名处理"""
    print("测试重名处理...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        source = tmpdir / "source"
        target = tmpdir / "target"
        source.mkdir()
        target.mkdir()
        (target / "images").mkdir()

        # 创建源文件
        (source / "photo.jpg").write_bytes(b"new image")

        # 目标已存在同名文件
        (target / "images" / "photo.jpg").write_bytes(b"existing image")

        # 整理
        result = organize_files(source, target, move=False)

        assert result.moved_files == 1

        # 验证重命名
        assert (target / "images" / "photo.jpg").exists()
        assert (target / "images" / "photo_1.jpg").exists()

        print("  ✓ 重名处理测试通过")


def test_generate_report():
    """测试报告生成"""
    print("测试 generate_report...")

    result = OrganizeResult(
        total_files=10,
        moved_files=8,
        skipped_files=2,
        total_size=1024 * 1024,
        by_category={"images": 5, "documents": 3}
    )

    report = generate_report(result)

    assert "总文件数: 10" in report
    assert "已处理: 8" in report
    assert "跳过: 2" in report
    assert "images: 5" in report
    assert "documents: 3" in report

    print("  ✓ generate_report 测试通过")


def main():
    """运行所有测试"""
    print("=" * 50)
    print("文件整理器测试")
    print("=" * 50)

    test_get_category()
    test_file_info()
    test_scan_directory()
    test_organize_files()
    test_organize_files_move()
    test_dry_run()
    test_duplicate_handling()
    test_generate_report()

    print("=" * 50)
    print("✅ 所有测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    main()


