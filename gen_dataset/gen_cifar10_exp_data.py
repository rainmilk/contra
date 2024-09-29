import torch
import numpy as np
import os
import argparse
from torchvision import datasets, transforms


def create_cifar10_npy_files(
    data_dir,
    gen_dir,
    noise_type="symmetric",
    noise_ratio=0.2,
    num_versions=3,
    retention_ratios=[0.5, 0.3, 0.1],
):
    transform = transforms.Compose([transforms.ToTensor()])

    # 加载 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_labels = torch.tensor(
        [train_dataset[i][1] for i in range(len(train_dataset))]
    )
    test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    num_samples = len(train_data)
    indices = np.random.permutation(num_samples)
    split_idx = num_samples // 2

    # 划分初始数据集 D_0 和增量数据集 D_inc^(0)
    D_0_indices = indices[:split_idx]
    D_inc_indices = indices[split_idx:]

    D_0_data = train_data[D_0_indices]
    D_0_labels = train_labels[D_0_indices]

    D_inc_data = train_data[D_inc_indices]
    D_inc_labels = train_labels[D_inc_indices]

    # 构建重放数据集 D_a（从 D_0 中随机抽取 10% 的样本）
    num_replay_samples = int(len(D_0_data) * 0.1)
    D_a_indices = np.random.choice(len(D_0_data), num_replay_samples, replace=False)
    D_a_data = D_0_data[D_a_indices]
    D_a_labels = D_0_labels[D_a_indices]

    # 定义遗忘类别和噪声类别
    forget_classes = [1, 3, 5, 7, 9]
    noise_classes = [0, 2, 4, 6, 8]

    # 获取增量数据集中的遗忘类别和噪声类别的索引
    D_inc_forget_indices = [
        i for i in range(len(D_inc_labels)) if D_inc_labels[i] in forget_classes
    ]
    D_inc_noise_indices = [
        i for i in range(len(D_inc_labels)) if D_inc_labels[i] in noise_classes
    ]

    # 定义非对称噪声映射
    asymmetric_mapping = {
        0: 2,
        2: 0,
        4: 6,
        6: 4,
        8: 0,
    }

    subdir = os.path.join(gen_dir, f"nr_{noise_ratio}_nt_{noise_type}")
    os.makedirs(subdir, exist_ok=True)

    # 保存初始数据集、初始增量数据集、重放数据集
    torch.save(D_0_data, os.path.join(subdir, "D_0.npy"))
    torch.save(D_0_labels, os.path.join(subdir, "D_0_labels.npy"))
    
    torch.save(D_inc_data, os.path.join(subdir, "D_inc_0_data.npy"))
    torch.save(D_inc_labels, os.path.join(subdir, "D_inc_0_labels.npy"))
    
    torch.save(D_a_data, os.path.join(subdir, "D_a.npy"))
    torch.save(D_a_labels, os.path.join(subdir, "D_a_labels.npy"))

    # 保存测试数据集
    torch.save(test_data, os.path.join(subdir, "test_data.npy"))
    torch.save(test_labels, os.path.join(subdir, "test_labels.npy"))

    num_classes = 10

    # 生成增量版本数据集
    for t in range(num_versions):
        retention_ratio = retention_ratios[t]

        # 模拟遗忘：根据保留比例抽取遗忘类别的样本
        num_forget_samples = int(len(D_inc_forget_indices) * retention_ratio)
        if num_forget_samples > 0:
            forget_sample_indices = np.random.choice(
                D_inc_forget_indices, num_forget_samples, replace=False
            )
            D_f_data = D_inc_data[forget_sample_indices]
            D_f_labels = D_inc_labels[forget_sample_indices]
        else:
            D_f_data = torch.empty(0, 3, 32, 32)
            D_f_labels = torch.empty(0, dtype=torch.long)

        # 噪声注入：对噪声类别的样本注入噪声
        noise_sample_indices = D_inc_noise_indices
        num_noisy_samples = int(len(noise_sample_indices) * noise_ratio)

        if num_noisy_samples > 0:
            noisy_indices = np.random.choice(
                noise_sample_indices, num_noisy_samples, replace=False
            )
        else:
            noisy_indices = []

        D_n_data = D_inc_data[noise_sample_indices]
        D_n_labels = D_inc_labels[noise_sample_indices].clone()

        # 在 D_n_labels 中注入噪声
        for idx_in_D_n, D_inc_idx in enumerate(noise_sample_indices):
            if D_inc_idx in noisy_indices:
                original_label = D_n_labels[idx_in_D_n].item()
                if noise_type == "symmetric":
                    new_label = original_label
                    while new_label == original_label:
                        new_label = np.random.randint(0, num_classes)
                    D_n_labels[idx_in_D_n] = new_label
                elif noise_type == "asymmetric":
                    if original_label in asymmetric_mapping:
                        D_n_labels[idx_in_D_n] = asymmetric_mapping[original_label]
                    else:
                        new_label = original_label
                        while new_label == original_label:
                            new_label = np.random.randint(0, num_classes)
                        D_n_labels[idx_in_D_n] = new_label
                else:
                    raise ValueError("Invalid noise type.")
            else:
                # 未被选中注入噪声的样本标签保持不变
                pass

        # 组合训练数据集 D_tr^{(t)}
        # D_tr_data = torch.cat([D_a_data, D_f_data, D_n_data], dim=0)
        # D_tr_labels = torch.cat([D_a_labels, D_f_labels, D_n_labels], dim=0)
        D_tr_data = torch.cat([D_f_data, D_n_data], dim=0)
        D_tr_labels = torch.cat([D_f_labels, D_n_labels], dim=0)

        # 打乱训练数据集
        perm = torch.randperm(len(D_tr_data))
        D_tr_data = D_tr_data[perm]
        D_tr_labels = D_tr_labels[perm]

        # 保存训练数据集
        torch.save(D_tr_data, os.path.join(subdir, f"D_tr_data_version_{t+1}.npy"))
        torch.save(D_tr_labels, os.path.join(subdir, f"D_tr_labels_version_{t+1}.npy"))

        print(f"D_tr 版本 {t+1} 已保存到 {subdir}")

    print("所有数据集生成完毕。")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(
        description="Generate CIFAR-10 incremental datasets."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/cifar-10/normal",
        help="原始 CIFAR-10 数据集的目录",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default="./data/cifar-10/gen/",
        help="生成数据集的保存目录",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="标签噪声类型：'symmetric' 或 'asymmetric'",
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=0.2, help="噪声比例（默认 0.2）"
    )
    parser.add_argument(
        "--num_versions", type=int, default=3, help="生成的增量版本数量（默认 3）"
    )
    parser.add_argument(
        "--retention_ratios",
        type=float,
        nargs="+",
        default=[0.5, 0.3, 0.1],
        help="各增量版本的保留比例列表",
    )

    args = parser.parse_args()

    if args.gen_dir is None:
        base_data_dir = os.path.join(os.path.dirname(__file__), "../data/cifar-10/")
        args.gen_dir = os.path.join(base_data_dir, "gen")

    create_cifar10_npy_files(
        data_dir=args.data_dir,
        gen_dir=args.gen_dir,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        num_versions=args.num_versions,
        retention_ratios=args.retention_ratios,
    )


if __name__ == "__main__":
    main()
