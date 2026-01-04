# 文件整理器

按扩展名自动分类文件的命令行工具。

## 功能

- 扫描指定目录中的文件
- 按扩展名自动分类（图片、文档、视频、音频、代码等）
- 支持复制或移动模式
- 自动处理重名文件
- 支持递归处理子目录
- 生成整理报告

## 使用方法

```bash
# 基本用法（复制文件到 source/organized）
python main.py ./downloads

# 指定目标目录
python main.py ./downloads --target ./sorted

# 移动文件（而非复制）
python main.py ./downloads --move

# 递归处理子目录
python main.py ./downloads --recursive

# 预览模式（不实际操作）
python main.py ./downloads --dry-run

# 详细输出
python main.py ./downloads --verbose

# 保存报告
python main.py ./downloads --report report.txt
```

## 文件分类

| 分类 | 扩展名 |
|------|--------|
| images | .jpg, .jpeg, .png, .gif, .bmp, .webp, .svg, .ico |
| documents | .pdf, .doc, .docx, .xls, .xlsx, .ppt, .pptx, .txt, .rtf |
| videos | .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm |
| audio | .mp3, .wav, .flac, .aac, .ogg, .wma, .m4a |
| archives | .zip, .rar, .7z, .tar, .gz, .bz2 |
| code | .py, .js, .ts, .java, .cpp, .c, .h, .css, .html, .json |
| data | .csv, .xml, .yaml, .yml, .sql, .db |
| others | 其他扩展名 |

## 命令行选项

| 选项 | 说明 |
|------|------|
| `source` | 源目录（必填） |
| `-t, --target` | 目标目录 |
| `-m, --move` | 移动文件（默认复制） |
| `-r, --recursive` | 递归处理子目录 |
| `-n, --dry-run` | 预览模式 |
| `-v, --verbose` | 详细输出 |
| `--report` | 报告输出路径 |

## 示例输出

```
==================================================
文件整理报告
==================================================
生成时间: 2024-01-15 14:30:45

统计:
  总文件数: 100
  已处理: 98
  跳过: 2
  总大小: 256.50 MB

按分类:
  code: 25 个文件
  documents: 20 个文件
  images: 35 个文件
  others: 10 个文件
  videos: 8 个文件
==================================================
```

## 使用的标准库

- `pathlib` - 文件路径操作
- `shutil` - 文件复制和移动
- `argparse` - 命令行参数解析
- `logging` - 日志输出
- `collections` - Counter 统计
- `dataclasses` - 数据类
- `datetime` - 日期时间
- `typing` - 类型提示


