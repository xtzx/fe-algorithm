"""
ZipApp 入口文件

当使用 python zipapp 打包后，这个文件会作为入口点
python myapp.pyz 等同于 python -m myapp
"""

from app import main

if __name__ == "__main__":
    main()


