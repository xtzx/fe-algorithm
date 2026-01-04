"""
评测命令行工具
"""

import argparse
import json
import sys
from pathlib import Path

from knowledge_assistant.evaluation.dataset import EvaluationDataset, create_sample_dataset
from knowledge_assistant.evaluation.runner import EvaluationRunner


def main():
    """评测命令行入口"""
    parser = argparse.ArgumentParser(
        description="知识库助手评测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 创建示例数据集
    create_parser = subparsers.add_parser("create-dataset", help="创建示例评测数据集")
    create_parser.add_argument(
        "-o", "--output",
        default="./data/eval_dataset/sample.json",
        help="输出文件路径",
    )
    
    # 运行评测
    run_parser = subparsers.add_parser("run", help="运行评测")
    run_parser.add_argument(
        "-d", "--dataset",
        required=True,
        help="数据集文件路径",
    )
    run_parser.add_argument(
        "-o", "--output",
        help="结果输出文件路径",
    )
    run_parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API 服务地址",
    )
    
    args = parser.parse_args()
    
    if args.command == "create-dataset":
        create_dataset_command(args)
    elif args.command == "run":
        run_evaluation_command(args)
    else:
        parser.print_help()
        sys.exit(1)


def create_dataset_command(args):
    """创建数据集命令"""
    dataset = create_sample_dataset()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset.save(output_path)
    print(f"示例数据集已创建: {output_path}")
    print(f"包含 {len(dataset)} 个测试用例")


def run_evaluation_command(args):
    """运行评测命令"""
    import httpx
    
    # 加载数据集
    dataset = EvaluationDataset.load(args.dataset)
    print(f"加载数据集: {dataset.name}")
    print(f"测试用例数: {len(dataset)}")
    
    # 创建 API 客户端
    client = httpx.Client(base_url=args.api_url, timeout=60.0)
    
    def model_fn(input_text: str) -> str:
        """调用 API 获取回答"""
        response = client.post(
            "/api/v1/query/",
            json={"question": input_text, "stream": False},
        )
        response.raise_for_status()
        return response.json()["answer"]
    
    # 运行评测
    runner = EvaluationRunner()
    results = runner.run(model_fn, dataset)
    
    # 输出结果
    print("\n" + results.summary())
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存: {output_path}")
    
    client.close()


if __name__ == "__main__":
    main()


