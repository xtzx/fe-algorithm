"""测试共享 fixture"""

import pytest
from pathlib import Path


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """创建示例项目目录"""
    project = tmp_path / "sample_project"
    project.mkdir()

    # Python 文件
    py_file = project / "main.py"
    py_file.write_text('''#!/usr/bin/env python3
"""Main module."""

def main():
    """Main function."""
    # This is a comment
    print("Hello, World!")

    # Another comment
    return 0


if __name__ == "__main__":
    main()
''')

    # JavaScript 文件
    js_file = project / "script.js"
    js_file.write_text('''// JavaScript file
function greet(name) {
    // Say hello
    console.log("Hello, " + name);
}

/*
 * Multi-line comment
 */
greet("World");
''')

    # 空文件
    empty_file = project / "empty.py"
    empty_file.write_text("")

    # 仅注释
    comments_only = project / "comments.py"
    comments_only.write_text('''# Just comments
# More comments
# And more
''')

    # 子目录
    subdir = project / "src"
    subdir.mkdir()

    module_file = subdir / "module.py"
    module_file.write_text('''"""A module."""

class MyClass:
    """A class."""

    def method(self):
        """A method."""
        pass
''')

    # 测试目录（应该被排除）
    test_dir = project / "__pycache__"
    test_dir.mkdir()
    cache_file = test_dir / "main.cpython-312.pyc"
    cache_file.write_bytes(b"fake bytecode")

    return project


@pytest.fixture
def python_file(tmp_path: Path) -> Path:
    """创建单个 Python 文件"""
    file = tmp_path / "test.py"
    file.write_text('''"""Module docstring."""

# A comment
def foo():
    """Function docstring."""
    x = 1  # inline comment
    return x


# Another comment

class Bar:
    pass
''')
    return file


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """创建配置文件"""
    config = tmp_path / ".code-counter.toml"
    config.write_text('''
exclude = ["node_modules", ".git", "dist"]
default_format = "json"
include_hidden = false
use_gitignore = true

[languages]
".custom" = "Custom"
''')
    return config

