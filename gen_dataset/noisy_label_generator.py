import numpy as np
import torch
import torch.nn.functional as F
import json
import os
from scipy import stats

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NoisyLabelGenerator:
    def __init__(self, dataset_name="custom", num_classes=10, noise_dir="assets"):
        """
        初始化噪声标签生成器
        :param dataset_name: 数据集名称 (如 'CIFAR-10', 'CIFAR-100', 'Tiny-ImageNet', 'Flower102', 或 'custom')
        :param num_classes: 数据集的类别数量 (默认为10)
        :param noise_dir: 存储噪声规则的目录
        """
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.noise_dir = noise_dir

        # 加载噪声规则
        self.symmetric_noise_rules = self._load_json("noise_symmetric.json")
        self.asymmetric_noise_rules = self._load_json("noise_asymmetric.json")
        self.pair_noise_rules = self._load_json("noise_pair.json")

    def _load_json(self, filename):
        """
        从 assets 目录加载 JSON 文件
        :param filename: 文件名
        :return: JSON 文件内容作为字典返回
        """
        file_path = os.path.join(self.noise_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"JSON file {filename} not found in {self.noise_dir}"
            )

        with open(file_path, "r") as f:
            return json.load(f)

    def _get_noisy_indices(self, y, noise_rate):
        """
        根据噪声率，获取要替换噪声的样本索引
        :param y: 原始标签
        :param noise_rate: 噪声比例
        :return: 随机噪声标签的索引数组
        """
        n_samples = len(y)
        n_noisy = int(noise_rate * n_samples)
        return np.random.choice(n_samples, n_noisy, replace=False)

    def add_symmetric_noise(self, y, noise_rate):
        """
        添加对称噪声 (Symmetric Noise)
        :param y: 原始标签数组
        :param noise_rate: 噪声比例
        :return: 添加噪声后的标签
        """
        noisy_labels = y.copy()
        noisy_idx = self._get_noisy_indices(y, noise_rate)

        for i in noisy_idx:
            noisy_labels[i] = np.random.randint(self.num_classes)  # 随机生成新的标签

        return noisy_labels

    def add_asymmetric_noise(self, y, noise_rate):
        """
        添加非对称噪声 (Asymmetric Noise)
        :param y: 原始标签数组
        :param noise_rate: 噪声比例
        :return: 添加噪声后的标签
        """
        noisy_labels = y.copy()
        noisy_idx = self._get_noisy_indices(y, noise_rate)

        # 从 JSON 文件中获取翻转规则
        flip_pairs = self.asymmetric_noise_rules.get(self.dataset_name, {})

        for i in noisy_idx:
            if noisy_labels[i] in flip_pairs:
                noisy_labels[i] = flip_pairs[noisy_labels[i]]

        return noisy_labels

    def add_pair_noise(self, y, noise_rate):
        """
        添加翻转噪声 (Pair Noise)
        :param y: 原始标签数组
        :param noise_rate: 噪声比例
        :return: 添加噪声后的标签
        """
        noisy_labels = y.copy()
        noisy_idx = self._get_noisy_indices(y, noise_rate)

        # 从 JSON 文件中获取翻转规则
        pair_flip = self.pair_noise_rules.get(self.dataset_name, {})

        for i in noisy_idx:
            if noisy_labels[i] in pair_flip:
                noisy_labels[i] = pair_flip[noisy_labels[i]]

        return noisy_labels

    def add_instance_noise(self, y, X, noise_rate, tau=0.2, std=0.1):
        """
        添加实例依赖噪声 (Instance-dependent Noise)
        :param y: 原始标签数组
        :param X: 输入特征 (如图像数据)
        :param noise_rate: 噪声比例
        :param tau: 噪声分布的均值
        :param std: 噪声分布的标准差
        :return: 添加噪声后的标签
        """
        num_samples = len(y)
        num_classes = self.num_classes
        feature_size = X.shape[1] * X.shape[2] * X.shape[3]  # 输入图像的特征维度

        # 定义截断正态分布
        flip_distribution = stats.truncnorm(
            (0 - tau) / std, (1 - tau) / std, loc=tau, scale=std
        )
        q = flip_distribution.rvs(num_samples)  # 样本的实例依赖翻转率

        # 生成特征权重矩阵
        W = (
            torch.tensor(np.random.randn(num_classes, feature_size, num_classes))
            .float()
            .to(device)
        )
        noisy_labels = y.copy()

        for i in range(num_samples):
            # 对每个样本，计算特征并生成实例依赖噪声
            x = torch.tensor(X[i].reshape(1, -1)).float().to(device)
            y_true = noisy_labels[i]

            p = x.mm(W[y_true]).squeeze(0)
            p[y_true] = float("-inf")  # 保证对角线为负无穷
            p = q[i] * F.softmax(p, dim=0)
            p[y_true] += 1 - q[i]

            # 根据生成的概率分布进行类别选择
            noisy_labels[i] = np.random.choice(
                np.arange(num_classes), p=p.cpu().numpy()
            )

        return noisy_labels

    def generate_noisy_labels(self, y, noise_type="symmetric", noise_rate=0.2, X=None):
        """
        根据指定的噪声类型生成带噪声的标签
        :param y: 原始标签数组
        :param noise_type: 噪声类型 ('symmetric', 'asymmetric', 'pair', 'instance')
        :param noise_rate: 噪声比例
        :param X: 输入数据 (仅在实例依赖噪声下使用)
        :return: 添加噪声后的标签
        """
        if noise_type == "symmetric":
            return self.add_symmetric_noise(y, noise_rate)
        elif noise_type == "asymmetric":
            return self.add_asymmetric_noise(y, noise_rate)
        elif noise_type == "pair":
            return self.add_pair_noise(y, noise_rate)
        elif noise_type == "instance":
            if X is None:
                raise ValueError("实例依赖噪声需要输入特征数据")
            return self.add_instance_noise(y, X, noise_rate)
        else:
            raise ValueError(f"未定义的噪声类型：{noise_type}")


# 示例使用代码
if __name__ == "__main__":
    # 假设我们已经加载了某个数据集的标签数组和特征数组
    # 以CIFAR-10为例，y_train为标签数组，X_train为图像数据
    y_train = np.array(
        [np.random.randint(0, 9) for _ in range(50000)]
    )  # 模拟CIFAR-10标签
    X_train = np.random.rand(50000, 32, 32, 3)  # 模拟CIFAR-10图像数据

    # 初始化生成器，假设我们使用的是CIFAR-10，num_classes为10
    generator = NoisyLabelGenerator(dataset_name="CIFAR-10", num_classes=10)

    # 生成对称噪声标签
    y_symmetric_noisy = generator.generate_noisy_labels(
        y_train, noise_type="symmetric", noise_rate=0.2
    )

    # 生成非对称噪声标签
    y_asymmetric_noisy = generator.generate_noisy_labels(
        y_train, noise_type="asymmetric", noise_rate=0.2
    )

    # 生成翻转噪声标签
    y_pair_noisy = generator.generate_noisy_labels(
        y_train, noise_type="pair", noise_rate=0.2
    )

    # 生成实例依赖噪声标签
    y_instance_noisy = generator.generate_noisy_labels(
        y_train, noise_type="instance", noise_rate=0.2, X=X_train
    )

    # 输出部分噪声标签
    print("Original labels:", y_train[:10])
    print("Symmetric noisy labels:", y_symmetric_noisy[:10])
    print("Asymmetric noisy labels:", y_asymmetric_noisy[:10])
    print("Pair noisy labels:", y_pair_noisy[:10])
    print("Instance noisy labels:", y_instance_noisy[:10])
