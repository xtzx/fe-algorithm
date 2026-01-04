#!/usr/bin/env python3
"""pathlib 模块演示"""

from pathlib import Path


def demo_path_creation():
    """Path 对象创建"""
    print("=" * 50)
    print("1. Path 对象创建")
    print("=" * 50)

    # 当前目录
    cwd = Path.cwd()
    print(f"当前目录: {cwd}")

    # 主目录
    home = Path.home()
    print(f"主目录: {home}")

    # 相对路径
    p = Path("data/file.txt")
    print(f"相对路径: {p}")

    # 路径拼接
    p = Path.home() / "Documents" / "file.txt"
    print(f"拼接路径: {p}")


def demo_path_properties():
    """路径属性"""
    print("\n" + "=" * 50)
    print("2. 路径属性")
    print("=" * 50)

    p = Path("/home/user/documents/report.tar.gz")

    print(f"完整路径: {p}")
    print(f"name (文件名): {p.name}")
    print(f"stem (不含扩展名): {p.stem}")
    print(f"suffix (扩展名): {p.suffix}")
    print(f"suffixes (所有扩展名): {p.suffixes}")
    print(f"parent (父目录): {p.parent}")
    print(f"parts (各部分): {p.parts}")


def demo_path_checks():
    """路径检查"""
    print("\n" + "=" * 50)
    print("3. 路径检查")
    print("=" * 50)

    p = Path(".")

    print(f"路径: {p}")
    print(f"exists(): {p.exists()}")
    print(f"is_file(): {p.is_file()}")
    print(f"is_dir(): {p.is_dir()}")
    print(f"is_absolute(): {p.is_absolute()}")
    print(f"resolve(): {p.resolve()}")


def demo_glob():
    """文件查找"""
    print("\n" + "=" * 50)
    print("4. 文件查找 (glob)")
    print("=" * 50)

    p = Path(".")

    # 查找 Python 文件
    print("当前目录下的 .py 文件:")
    for f in p.glob("*.py"):
        print(f"  {f}")

    # 递归查找
    print("\n递归查找所有 .py 文件 (rglob):")
    count = 0
    for f in p.rglob("*.py"):
        print(f"  {f}")
        count += 1
        if count >= 5:
            print("  ...")
            break


def demo_file_operations():
    """文件操作"""
    print("\n" + "=" * 50)
    print("5. 文件操作")
    print("=" * 50)

    # 创建临时文件
    temp_file = Path("temp_demo.txt")

    # 写入
    temp_file.write_text("Hello, pathlib!\n这是一个测试文件。", encoding="utf-8")
    print(f"写入文件: {temp_file}")

    # 读取
    content = temp_file.read_text(encoding="utf-8")
    print(f"读取内容: {content}")

    # 文件信息
    stat = temp_file.stat()
    print(f"文件大小: {stat.st_size} 字节")

    # 改变扩展名
    new_path = temp_file.with_suffix(".md")
    print(f"改变扩展名: {temp_file} -> {new_path}")

    # 删除
    temp_file.unlink()
    print(f"删除文件: {temp_file}")


def demo_directory_operations():
    """目录操作"""
    print("\n" + "=" * 50)
    print("6. 目录操作")
    print("=" * 50)

    # 创建目录
    temp_dir = Path("temp_demo_dir")
    temp_dir.mkdir(exist_ok=True)
    print(f"创建目录: {temp_dir}")

    # 创建子目录
    sub_dir = temp_dir / "sub1" / "sub2"
    sub_dir.mkdir(parents=True, exist_ok=True)
    print(f"创建嵌套目录: {sub_dir}")

    # 创建文件
    (temp_dir / "file1.txt").write_text("file1")
    (temp_dir / "file2.txt").write_text("file2")

    # 遍历目录
    print("目录内容:")
    for item in temp_dir.iterdir():
        item_type = "目录" if item.is_dir() else "文件"
        print(f"  [{item_type}] {item.name}")

    # 清理
    import shutil
    shutil.rmtree(temp_dir)
    print(f"删除目录: {temp_dir}")


if __name__ == "__main__":
    demo_path_creation()
    demo_path_properties()
    demo_path_checks()
    demo_glob()
    demo_file_operations()
    demo_directory_operations()

    print("\n✅ pathlib 演示完成!")


