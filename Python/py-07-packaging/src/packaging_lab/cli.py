"""命令行接口模块"""

import click

from . import __version__
from .core import greet, calculate


@click.group()
@click.version_option(version=__version__, prog_name="pkglab")
def main():
    """Packaging Lab - Python 包管理学习示例 CLI

    这是一个用于学习 Python 包管理的示例项目。
    """
    pass


@main.command()
@click.argument("name")
@click.option("-g", "--greeting", default="Hello", help="问候语前缀")
@click.option("-n", "--times", default=1, type=int, help="重复次数")
def hello(name: str, greeting: str, times: int):
    """向某人问好

    NAME: 要问候的名字
    """
    for _ in range(times):
        message = greet(name, greeting)
        click.echo(message)


@main.command()
@click.argument("a", type=float)
@click.argument("b", type=float)
@click.option(
    "-o", "--operation",
    type=click.Choice(["add", "sub", "mul", "div"]),
    default="add",
    help="运算类型"
)
def calc(a: float, b: float, operation: str):
    """执行数学运算

    A: 第一个操作数
    B: 第二个操作数
    """
    try:
        result = calculate(a, b, operation)
        click.echo(f"{a} {operation} {b} = {result}")
    except ZeroDivisionError:
        click.echo("错误: 除以零", err=True)
        raise SystemExit(1)


@main.command()
def info():
    """显示包信息"""
    click.echo(f"Packaging Lab v{__version__}")
    click.echo()
    click.echo("这是一个用于学习 Python 包管理的示例项目。")
    click.echo()
    click.echo("功能:")
    click.echo("  - hello: 问候某人")
    click.echo("  - calc: 数学运算")
    click.echo("  - info: 显示此信息")


if __name__ == "__main__":
    main()

