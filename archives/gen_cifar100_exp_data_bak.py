import torch
import numpy as np
import os
import argparse
from torchvision import datasets, transforms


def add_noise_labels(labels, noise_type="symmetric", noise_ratio=0.2, num_classes=100):
    noisy_labels = labels.clone()
    n_samples = len(labels)
    n_noisy = int(noise_ratio * n_samples)

    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    if noise_type == "symmetric":
        for idx in noisy_indices:
            original_label = noisy_labels[idx].item()
            new_label = original_label
            while new_label == original_label:
                new_label = np.random.randint(0, num_classes)
            noisy_labels[idx] = new_label

    elif noise_type == "asymmetric":
        # Asymmetric noise mapping (example: adjacent class mapping)
        asymmetric_mapping = {
            i: (i + 1) % 100 for i in range(100)
        }  # 将类别映射到下一个类别，最后一类映射到第一个类别
        for idx in noisy_indices:
            original_label = noisy_labels[idx].item()
            if original_label in asymmetric_mapping:
                noisy_labels[idx] = asymmetric_mapping[original_label]
            else:
                new_label = original_label
                while new_label == original_label:
                    new_label = np.random.randint(0, num_classes)
                noisy_labels[idx] = new_label
    else:
        raise ValueError("Invalid noise type. Choose 'symmetric' or 'asymmetric'.")

    return noisy_labels


def create_cifar100_npy_files(
    data_dir,
    gen_dir,
    noise_type="symmetric",
    noise_ratio=0.2,
    num_versions=3,
    retention_ratios=[0.5, 0.3, 0.1],
):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR100(
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
    D_0_indices = indices[:split_idx]
    D_inc_indices = indices[split_idx:]

    D_0_data = train_data[D_0_indices]
    D_0_labels = train_labels[D_0_indices]

    # 遗忘类别为每10个类别中的一个，非遗忘类别为剩余类别
    forget_classes = list(range(1, 100, 10))
    non_forget_classes = [i for i in range(100) if i not in forget_classes]

    D_0_forget_indices = [
        i for i in range(len(D_0_labels)) if D_0_labels[i] in forget_classes
    ]
    D_0_non_forget_indices = [
        i for i in range(len(D_0_labels)) if D_0_labels[i] in non_forget_classes
    ]

    D_0_forget_data = D_0_data[D_0_forget_indices]
    D_0_forget_labels = D_0_labels[D_0_forget_indices]
    D_0_non_forget_data = D_0_data[D_0_non_forget_indices]
    D_0_non_forget_labels = D_0_labels[D_0_non_forget_indices]

    num_replay_samples = int(len(D_0_data) * 0.1)
    D_a_indices = np.random.choice(len(D_0_data), num_replay_samples, replace=False)
    D_a_data = D_0_data[D_a_indices]
    D_a_labels = D_0_labels[D_a_indices]

    D_inc_versions = []
    for t in range(num_versions):
        retention_ratio = retention_ratios[t]
        num_forget_samples = int(len(D_0_forget_data) * retention_ratio)
        num_non_forget_samples = int(len(D_0_non_forget_data) * retention_ratio)

        D_inc_forget_data, D_inc_forget_labels = (
            (torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long))
            if num_forget_samples == 0
            else (
                D_0_forget_data[
                    np.random.choice(
                        len(D_0_forget_data), num_forget_samples, replace=False
                    )
                ],
                D_0_forget_labels[
                    np.random.choice(
                        len(D_0_forget_labels), num_forget_samples, replace=False
                    )
                ],
            )
        )

        D_inc_non_forget_data, D_inc_non_forget_labels = (
            (torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long))
            if num_non_forget_samples == 0
            else (
                D_0_non_forget_data[
                    np.random.choice(
                        len(D_0_non_forget_data), num_non_forget_samples, replace=False
                    )
                ],
                D_0_non_forget_labels[
                    np.random.choice(
                        len(D_0_non_forget_labels),
                        num_non_forget_samples,
                        replace=False,
                    )
                ],
            )
        )

        D_inc_data = torch.cat([D_inc_forget_data, D_inc_non_forget_data], dim=0)
        D_inc_labels = torch.cat([D_inc_forget_labels, D_inc_non_forget_labels], dim=0)

        # 噪声注入
        num_noisy_samples = int(len(D_inc_non_forget_data) * noise_ratio)
        if num_noisy_samples > 0 and len(D_inc_non_forget_data) > 0:
            D_inc_labels = add_noise_labels(
                D_inc_labels,
                noise_type=noise_type,
                noise_ratio=noise_ratio,
                num_classes=100,
            )

        D_inc_versions.append((D_inc_data, D_inc_labels))

    subdir = os.path.join(gen_dir, f"nr_{noise_ratio}_nt_{noise_type}")
    os.makedirs(subdir, exist_ok=True)

    torch.save(D_0_data, os.path.join(subdir, "D_0.npy"))
    torch.save(D_0_labels, os.path.join(subdir, "D_0_labels.npy"))
    torch.save(D_a_data, os.path.join(subdir, "D_a.npy"))
    torch.save(D_a_labels, os.path.join(subdir, "D_a_labels.npy"))

    for t, (data, labels) in enumerate(D_inc_versions):
        torch.save(data, os.path.join(subdir, f"D_inc_{t+1}.npy"))
        torch.save(labels, os.path.join(subdir, f"D_inc_labels_{t+1}.npy"))

    torch.save(test_data, os.path.join(subdir, "test_data.npy"))
    torch.save(test_labels, os.path.join(subdir, "test_labels.npy"))

    print("CIFAR-100数据集生成完毕。")
    return (
        train_data[D_inc_indices],
        train_labels[D_inc_indices],
        test_data,
        test_labels,
    )


def create_incremental_data_versions(
    D_a_data, D_a_labels, D_inc_versions, save_dir=None
):
    for version_num, (D_inc_data, D_inc_labels) in enumerate(D_inc_versions, start=1):
        D_tr_data = torch.cat([D_a_data, D_inc_data], dim=0)
        D_tr_labels = torch.cat([D_a_labels, D_inc_labels], dim=0)

        if save_dir:
            np.save(
                os.path.join(
                    save_dir, f"cifar-100_D_tr_data_version_{version_num}.npy"
                ),
                D_tr_data.numpy(),
            )
            np.save(
                os.path.join(
                    save_dir, f"cifar-100_D_tr_labels_version_{version_num}.npy"
                ),
                D_tr_labels.numpy(),
            )
            print(f"D_tr 版本 {version_num} 已保存到 {save_dir}")

    print("所有 D_tr 版本数据保存完毕！")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Generate CIFAR-100 noisy datasets.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/cifar-100/normal",
        help="原始CIFAR-100数据集的目录",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default="./data/cifar-100/gen",
        help="生成数据集的保存目录",
    )
    parser.add_argument(
        "--noise_ratio", type=float, default=0.2, help="增量数据集中的噪声比例"
    )
    parser.add_argument(
        "--num_versions", type=int, default=3, help="生成的增量版本数量"
    )
    parser.add_argument(
        "--retention_ratios",
        type=float,
        nargs="+",
        default=[0.5, 0.3, 0.1],
        help="各增量版本的Retention ratio列表",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        choices=["symmetric", "asymmetric"],
        default="symmetric",
        help="标签噪声类型：'symmetric' 或 'asymmetric'",
    )

    args = parser.parse_args()

    if args.gen_dir is None:
        base_data_dir = os.path.join(os.path.dirname(__file__), "../data/cifar-100")
        args.gen_dir = os.path.join(base_data_dir, "noise")

    create_cifar100_npy_files(
        data_dir=args.data_dir,
        gen_dir=args.gen_dir,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        num_versions=args.num_versions,
        retention_ratios=args.retention_ratios,
    )

    subdir = os.path.join(args.gen_dir, f"nr_{args.noise_ratio}_nt_{args.noise_type}")
    print("subdir:", subdir)
    D_a_data = torch.load(os.path.join(subdir, "D_a.npy"))
    D_a_labels = torch.load(os.path.join(subdir, "D_a_labels.npy"))

    D_inc_versions = []
    for t in range(args.num_versions):
        D_inc_data = torch.load(os.path.join(subdir, f"D_inc_{t+1}.npy"))
        D_inc_labels = torch.load(os.path.join(subdir, f"D_inc_labels_{t+1}.npy"))
        D_inc_versions.append((D_inc_data, D_inc_labels))

    create_incremental_data_versions(
        D_a_data,
        D_a_labels,
        D_inc_versions,
        save_dir=subdir,
    )


if __name__ == "__main__":
    main()
