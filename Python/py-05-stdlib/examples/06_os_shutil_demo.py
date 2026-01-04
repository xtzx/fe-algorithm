#!/usr/bin/env python3
"""os 和 shutil 模块演示"""

import os
import shutil
import tempfile
from pathlib import Path


def demo_environ():
    """环境变量"""
    print("=" * 50)
    print("1. 环境变量")
    print("=" * 50)

    # 获取环境变量
    home = os.environ.get("HOME")
    print(f"HOME: {home}")

    path = os.environ.get("PATH", "").split(os.pathsep)[:3]
    print(f"PATH (前3个): {path}")

    # 使用 getenv
    user = os.getenv("USER", "unknown")
    print(f"USER: {user}")

    # 设置环境变量（仅当前进程）
    os.environ["MY_TEST_VAR"] = "test_value"
    print(f"MY_TEST_VAR: {os.getenv('MY_TEST_VAR')}")

    # 删除
    del os.environ["MY_TEST_VAR"]
    print(f"删除后 MY_TEST_VAR: {os.getenv('MY_TEST_VAR', 'None')}")


def demo_os_basic():
    """os 基本操作"""
    print("\n" + "=" * 50)
    print("2. os 基本操作")
    print("=" * 50)

    # 当前目录
    cwd = os.getcwd()
    print(f"当前目录: {cwd}")

    # 列出目录
    items = os.listdir(".")[:5]
    print(f"当前目录内容 (前5个): {items}")

    # 创建临时目录进行演示
    temp_dir = "temp_os_demo"
    os.makedirs(temp_dir, exist_ok=True)
    print(f"创建目录: {temp_dir}")

    # 创建文件
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("Hello, OS!")
    print(f"创建文件: {test_file}")

    # 文件信息
    stat = os.stat(test_file)
    print(f"文件大小: {stat.st_size} 字节")

    # 重命名
    new_file = os.path.join(temp_dir, "renamed.txt")
    os.rename(test_file, new_file)
    print(f"重命名: {test_file} -> {new_file}")

    # 清理
    os.remove(new_file)
    os.rmdir(temp_dir)
    print(f"删除文件和目录")


def demo_os_path():
    """os.path 操作"""
    print("\n" + "=" * 50)
    print("3. os.path 操作")
    print("=" * 50)

    path = "/home/user/documents/report.txt"

    print(f"路径: {path}")
    print(f"dirname: {os.path.dirname(path)}")
    print(f"basename: {os.path.basename(path)}")
    print(f"splitext: {os.path.splitext(path)}")
    print(f"exists (当前目录): {os.path.exists('.')}")
    print(f"isfile: {os.path.isfile('.')}")
    print(f"isdir: {os.path.isdir('.')}")
    print(f"abspath ('.'): {os.path.abspath('.')}")

    # 拼接
    joined = os.path.join("dir1", "dir2", "file.txt")
    print(f"join: {joined}")


def demo_shutil():
    """shutil 操作"""
    print("\n" + "=" * 50)
    print("4. shutil 操作")
    print("=" * 50)

    # 创建测试目录和文件
    src_dir = Path("temp_shutil_src")
    dst_dir = Path("temp_shutil_dst")

    src_dir.mkdir(exist_ok=True)
    (src_dir / "file1.txt").write_text("content1")
    (src_dir / "file2.txt").write_text("content2")
    (src_dir / "subdir").mkdir(exist_ok=True)
    (src_dir / "subdir" / "file3.txt").write_text("content3")

    print(f"创建源目录: {src_dir}")

    # 复制文件
    shutil.copy(src_dir / "file1.txt", src_dir / "file1_copy.txt")
    print("复制文件: file1.txt -> file1_copy.txt")

    # 复制目录
    shutil.copytree(src_dir, dst_dir)
    print(f"复制目录: {src_dir} -> {dst_dir}")

    # 列出目标目录内容
    print(f"目标目录内容: {list(dst_dir.rglob('*'))}")

    # 移动文件
    shutil.move(str(dst_dir / "file1.txt"), str(dst_dir / "moved.txt"))
    print("移动文件: file1.txt -> moved.txt")

    # 磁盘使用
    usage = shutil.disk_usage("/")
    print(f"磁盘使用:")
    print(f"  总计: {usage.total / 1e9:.1f} GB")
    print(f"  已用: {usage.used / 1e9:.1f} GB")
    print(f"  可用: {usage.free / 1e9:.1f} GB")

    # 清理
    shutil.rmtree(src_dir)
    shutil.rmtree(dst_dir)
    print("清理完成")


def demo_tempfile():
    """tempfile 操作"""
    print("\n" + "=" * 50)
    print("5. tempfile 操作")
    print("=" * 50)

    # 临时目录路径
    print(f"临时目录: {tempfile.gettempdir()}")

    # 临时文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("临时内容")
        temp_path = f.name
    print(f"临时文件: {temp_path}")

    # 读取临时文件
    with open(temp_path) as f:
        content = f.read()
    print(f"临时文件内容: {content}")

    # 删除
    os.unlink(temp_path)

    # 临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"临时目录: {tmpdir}")
        # 在临时目录中创建文件
        temp_file = Path(tmpdir) / "test.txt"
        temp_file.write_text("test content")
        print(f"  创建文件: {temp_file}")
        print(f"  文件内容: {temp_file.read_text()}")
    # 退出 with 块后自动删除
    print("临时目录已自动删除")


def demo_archive():
    """归档操作"""
    print("\n" + "=" * 50)
    print("6. 归档操作")
    print("=" * 50)

    # 创建测试目录
    src_dir = Path("temp_archive_src")
    src_dir.mkdir(exist_ok=True)
    (src_dir / "file1.txt").write_text("content1")
    (src_dir / "file2.txt").write_text("content2")

    # 创建 zip 归档
    archive_name = shutil.make_archive("temp_backup", "zip", src_dir)
    print(f"创建归档: {archive_name}")

    # 解压
    extract_dir = "temp_extract"
    shutil.unpack_archive(archive_name, extract_dir)
    print(f"解压到: {extract_dir}")
    print(f"解压内容: {list(Path(extract_dir).iterdir())}")

    # 清理
    shutil.rmtree(src_dir)
    shutil.rmtree(extract_dir)
    os.remove(archive_name)
    print("清理完成")


if __name__ == "__main__":
    demo_environ()
    demo_os_basic()
    demo_os_path()
    demo_shutil()
    demo_tempfile()
    demo_archive()

    print("\n✅ os 和 shutil 演示完成!")


