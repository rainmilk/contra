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
        help="Dataset name, choose from: cifar-10, cifar-100, animals-10, tiny-imagenet-200",
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
        help="Condition for the experiment: original_data, removed_50_percent, noisy_data",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        required=True,
        help="List of selected classes for the experiment, e.g., --classes 0 1 2 3 4",
    )

    # 返回解析的参数
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 打印用户输入的配置信息
    print(f"Running experiment with the following configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Condition: {args.condition}")
    print(f"  Selected Classes: {args.classes}")

    # 运行实验
    run_experiment(args.dataset, args.model, args.classes, args.condition)


if __name__ == "__main__":

    # python main.py --dataset cifar-10 --model resnet18 --condition removed_50_percent --classes 0 1 2 3 4
    # python main.py --dataset cifar-100 --model vgg16 --condition noisy_data --classes 0 1 2 3 4

    main()
