import os
import argparse
from run_experiment import run_experiment


# 命令行解析
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiments with different datasets, models, and conditions."
    )

    # 添加参数
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, choose from: cifar-10, cifar-100, animals-10, flowers-102",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name, choose from: resnet18, vgg16",
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        help="Condition for the experiment: original_data, removed_50_percent, noisy_data, combined",
    )
    parser.add_argument(
        "--classes_remove",
        type=int,
        nargs="+",
        required=False,
        help="List of classes to remove samples from, e.g., --classes_remove 0 1 2 3 4",
    )
    parser.add_argument(
        "--classes_noise",
        type=int,
        nargs="+",
        required=False,
        help="List of classes to add noise to, e.g., --classes_noise 5 6 7 8 9",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Specify the GPU(s) to use, e.g., --gpu 0,1 for multi-GPU or --gpu 0 for single GPU",
    )
    # 返回解析的参数
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 设置 GPU 环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU(s): {args.gpu}")

    # 检查组合操作的条件
    if args.condition == "combined":
        if not args.classes_remove or not args.classes_noise:
            raise ValueError(
                "For 'combined' condition, both --classes_remove and --classes_noise must be provided."
            )
    elif args.condition in ["removed_50_percent", "noisy_data"]:
        if not args.classes_remove:
            raise ValueError(
                "For 'removed_50_percent' or 'noisy_data' condition, --classes_remove must be provided."
            )

    # 打印用户输入的配置信息
    print(f"Running experiment with the following configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Condition: {args.condition}")
    if args.classes_remove:
        print(f"  Classes to Remove Samples From: {args.classes_remove}")
    if args.classes_noise:
        print(f"  Classes to Add Noise To: {args.classes_noise}")

    # 运行实验
    run_experiment(
        args.dataset,
        args.model,
        args.classes_remove,
        args.classes_noise,
        args.condition,
    )


if __name__ == "__main__":

    """
    Cifar-10 Example
    """
    # python main.py --dataset cifar-10 --model resnet18 --condition removed_50_percent --classes_remove 0 1 2 3 4
    # python main.py --dataset cifar-10 --model resnet18 --condition noisy_data --classes_remove 0 1 2 3 4
    # python main.py --dataset cifar-10 --model resnet18 --condition combined --classes_remove 0 1 2 3 4 --classes_noise 5 6 7 8 9

    main()
