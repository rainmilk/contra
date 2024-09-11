import os
import argparse
from run_experiment import run_experiment


# 解析 classes 参数（支持 0-9 形式）
def parse_class_range(value):
    if "-" in value:
        start, end = map(int, value.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(value)]


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
        help="Condition for the experiment: original_data, remove_data, noisy_data, all_perturbations",
    )

    parser.add_argument(
        "--classes_remove",
        type=parse_class_range,
        nargs="+",
        required=False,
        help="List of classes to remove samples from, e.g., --classes_remove 0 1 2 3 4 or 0-4",
    )

    # 添加 remove_fraction 参数，用于指定删除样本的比例
    parser.add_argument(
        "--remove_fraction",
        type=float,
        default=0.5,
        help="Fraction of samples to remove from the selected classes, e.g., --remove_fraction 0.5 for 50%% removal (default: 0.5)",
    )

    parser.add_argument(
        "--classes_noise",
        type=parse_class_range,
        nargs="+",
        required=False,
        help="List of classes to add noise to, e.g., --classes_noise 5 6 7 8 9 or 5-9",
    )

    # 添加 noise_type 参数，用于指定噪声类型
    parser.add_argument(
        "--noise_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "salt_pepper"],
        help="Type of noise to add to the selected classes, e.g., --noise_type gaussian or --noise_type salt_pepper (default: gaussian)",
    )

    parser.add_argument(
        "--noise_fraction",
        type=float,
        default=0.8,
        help="Fraction of samples in the selected classes to add noise to, e.g., --noise_fraction 0.1 for 10%% noise injection (default: 0.8)",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Specify the GPU(s) to use, e.g., --gpu 0,1 for multi-GPU or --gpu 0 for single GPU",
    )

    # 添加早停开关参数
    parser.add_argument(
        "--use_early_stopping",
        action="store_true",
        help="Enable early stopping if specified, otherwise train for the full number of epochs",
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
    if args.condition == "all_perturbations":
        if not args.classes_remove or not args.classes_noise:
            raise ValueError(
                "For 'all_perturbations' condition, both --classes_remove and --classes_noise must be provided."
            )

    # 打印用户输入的配置信息
    print(f"Running experiment with the following configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Condition: {args.condition}")
    if args.classes_remove:
        print(f"  Classes to Remove Samples From: {args.classes_remove}")
        print(f"  Remove Fraction: {args.remove_fraction}")
    if args.classes_noise:
        print(f"  Classes to Add Noise To: {args.classes_noise}")
        print(f"  Noise Type: {args.noise_type}")
        print(f"  Noise Fraction: {args.noise_fraction}")
    print(f"  Use Early Stopping: {args.use_early_stopping}")

    # 运行实验
    run_experiment(
        args.dataset,
        args.model,
        args.classes_remove,
        args.classes_noise,
        args.condition,
        args.remove_fraction,
        args.noise_type,
        args.noise_fraction,
        args.use_early_stopping,
    )


if __name__ == "__main__":

    """
    Cifar-10 Example
    """
    # python main.py --dataset cifar-10 --model resnet18 --condition remove_data --classes_remove 0 1 2 3 4 --remove_fraction 0.5 --gpu 0
    # python main.py --dataset cifar-10 --model resnet18 --condition noisy_data --classes_remove 0 1 2 3 4 --classes_noise 5 6 7 8 9 --noise_type gaussian --noise_fraction 0.1 --gpu 0
    # python main.py --dataset cifar-10 --model resnet18 --condition all_perturbations --classes_remove 0 1 2 3 4 --remove_fraction 0.5 --classes_noise 5 6 7 8 9 --noise_type salt_pepper --noise_fraction 0.2 --gpu 0

    main()
