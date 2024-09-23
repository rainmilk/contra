import numpy as np
import os


def reconstruct_D_tr(
    forget_class_data_file,
    forget_class_labels_file,
    other_class_data_file,
    noisy_other_class_labels_file,
    save_dir,
):
    """
    从保存的 .npy 文件中重新构造增量数据集 D_tr，并保存为新的 .npy 文件。
    :param forget_class_data_file: 遗忘类的数据文件路径
    :param forget_class_labels_file: 遗忘类的标签文件路径
    :param other_class_data_file: 非遗忘类的数据文件路径
    :param noisy_other_class_labels_file: 非遗忘类的噪声标签文件路径
    :param save_dir: 保存构造的 D_tr 数据集的路径
    """
    # 加载遗忘类数据和标签
    forget_class_data = np.load(forget_class_data_file)
    forget_class_labels = np.load(forget_class_labels_file)

    # 加载非遗忘类数据和噪声标签
    other_class_data = np.load(other_class_data_file)
    noisy_other_class_labels = np.load(noisy_other_class_labels_file)

    # 合并遗忘类和非遗忘类的数据与标签
    D_tr_data = np.concatenate((forget_class_data, other_class_data), axis=0)
    D_tr_labels = np.concatenate(
        (forget_class_labels, noisy_other_class_labels), axis=0
    )

    # 校验合并前后的数据
    validate_data(
        forget_class_data,
        forget_class_labels,
        other_class_data,
        noisy_other_class_labels,
        D_tr_data,
        D_tr_labels,
    )

    # 保存 D_tr 数据和标签
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/cifar10_D_tr_data.npy", D_tr_data)
    np.save(f"{save_dir}/cifar10_D_tr_labels.npy", D_tr_labels)

    print(f"Saved D_tr data and labels to {save_dir}")


def validate_data(
    forget_class_data,
    forget_class_labels,
    other_class_data,
    noisy_other_class_labels,
    D_tr_data,
    D_tr_labels,
):
    """
    校验合并前的数据和合并后的数据是否一致。
    :param forget_class_data: 遗忘类数据
    :param forget_class_labels: 遗忘类标签
    :param other_class_data: 非遗忘类数据
    :param noisy_other_class_labels: 非遗忘类的噪声标签
    :param D_tr_data: 合并后的 D_tr 数据
    :param D_tr_labels: 合并后的 D_tr 标签
    """
    # 校验数据总数
    total_data_count = forget_class_data.shape[0] + other_class_data.shape[0]
    total_label_count = forget_class_labels.shape[0] + noisy_other_class_labels.shape[0]

    assert (
        D_tr_data.shape[0] == total_data_count
    ), f"数据数量不匹配: {D_tr_data.shape[0]} != {total_data_count}"
    assert (
        D_tr_labels.shape[0] == total_label_count
    ), f"标签数量不匹配: {D_tr_labels.shape[0]} != {total_label_count}"

    print("数据数量校验通过！")

    # 校验合并前后数据的值是否一致（通过拼接校验）
    assert np.array_equal(
        D_tr_data[: forget_class_data.shape[0]], forget_class_data
    ), "遗忘类数据与合并后的数据不一致！"
    assert np.array_equal(
        D_tr_data[forget_class_data.shape[0] :], other_class_data
    ), "非遗忘类数据与合并后的数据不一致！"

    # 校验合并前后标签的值是否一致
    assert np.array_equal(
        D_tr_labels[: forget_class_labels.shape[0]], forget_class_labels
    ), "遗忘类标签与合并后的标签不一致！"
    assert np.array_equal(
        D_tr_labels[forget_class_labels.shape[0] :], noisy_other_class_labels
    ), "非遗忘类标签与合并后的标签不一致！"

    print("数据值校验通过！")


if __name__ == "__main__":
    # 原始文件路径
    forget_class_data_file = "cifar10_forget_class_data.npy"
    forget_class_labels_file = "cifar10_forget_class_labels.npy"
    other_class_data_file = "cifar10_other_class_data.npy"
    noisy_other_class_labels_file = "cifar10_noisy_other_class_labels.npy"

    # 保存构造的D_tr的路径
    save_dir = "./reconstruct_D_tr"

    # 调用构造和校验函数
    reconstruct_D_tr(
        forget_class_data_file,
        forget_class_labels_file,
        other_class_data_file,
        noisy_other_class_labels_file,
        save_dir,
    )
