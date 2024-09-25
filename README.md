# TTA-MR

## Data and model

TODO 1. 保存每个 stage 的 $D_{tr}$ 的 image 和 label 。
 $D_{tr}$ npy 补充上传到网盘。
        ├── cifar10_noisy_other_class_labels.npy # 经过 20% 标签噪声处理的非遗忘类标签, 52K
        <!-- ├── cifar10_other_class_data.npy # 非遗忘类的数据，包含从 D_inc 中抽取的 50% 非遗忘类样本, 201M
        ├── cifar10_other_class_labels.npy # 非遗忘类的标签，包含非遗忘类数据的类别标签, 52K -->
        ├── cifar10_forget_class_data.npy # 遗忘类的数据，包含从 D_inc 中抽取的 10% 的遗忘类样本, 15M
        ├── cifar10_forget_class_labels.npy # 遗忘类的标签，包含对应遗忘类数据的类别标签, 12K

TODO 2. stage-1的模型保存。
TODO 3. 数据分 stage 保存。
TODO 4. run 3 轮。
TODO 5. input image: 10 images，5 rows 10 cols
TODO 6. recall-precision figures, line plot
TODO 7. www 完成 method.

1. 胡院
   1. 1-2，调通
   2. 3-6, baseline
   3. 同步记录 ablation study 的结果。
2. AILab：1-6, dataset / base model
   1. 实现 2 种噪声, sy, asy
   2. 至少 3 个数据集, CIFAR-10 CIFAR-100 TINY-200。
   3. （Animal-10 备选）从 pytorch 内置。
   4. 1 个模型 resnet 。
3. 国庆期间，所有结果 ready。
4. 分析代码和论文的图表可以在国庆期间同步实现。
5. 国庆期间 abstract ddl 。
6. 国庆期间，full paper 完成。
7. 国庆后第一周，full paper 修改完成。

For later experiment, we should firstly prepare the datasets and models. The logic of generating dataset is as follows:

### 数据集构造逻辑

#### 1. **数据集划分**：
从原始的 `CIFAR-10` 训练集（50000 张图像）中，随机划分为两个子集：
- **$D_0$**：包含 50% 的训练数据，共 25000 个样本。
- **$D_{inc}$**：包含剩余 50% 的训练数据，也为 25000 个样本。

#### 2. **Replay 数据集 $D_a$ 的构建**：
从 **$D_0$** 中随机抽取 10% 的样本（即 2500 个样本），作为 Replay 数据集 **$D_a$**，此数据集用于未来的增量训练过程以缓解遗忘效应。

#### 3. **测试集 $D_{ts}$ 的使用**：
测试集 **$D_{ts}$** 使用 CIFAR-10 原始测试数据集（10000 张图像）进行评估模型的泛化性能。

#### 4. **增量训练数据集 $D_{tr}$ 的构建**：
从 **$D_{inc}$** 中构造增量训练数据集 **$D_{tr}$**，具体步骤如下：
- **遗忘类样本**：对于指定的遗忘类（假设选择类 1、3、5、7、9），仅从 **$D_{inc}$** 中抽取 10% 的样本加入 **$D_{tr}$**。
- **非遗忘类样本**：对于其余非遗忘类，从 **$D_{inc}$** 中抽取 50% 的样本加入 **$D_{tr}$**。
- **噪声样本添加**：对 **$D_{tr}$** 中非遗忘类的 50% 样本中的 20% 进行标签噪声处理，具体是将这些样本的标签随机替换为其他类的标签。

#### 5. **数据集保存**：
生成的数据集被保存为 `.npy` 格式文件，以供后续训练和验证使用，具体为：
- `cifar10_train_data.npy` 和 `cifar10_train_labels.npy`：保存 **$D_0$** 的数据和标签。
- `cifar10_aux_data.npy` 和 `cifar10_aux_labels.npy`：保存 **$D_a$** 的数据和标签。
- `cifar10_test_data.npy` 和 `cifar10_test_labels.npy`：保存 **$D_{ts}$** 的测试数据和标签。

An incremental training dataset $D_{tr}$ with noise is constructed, and part of the original training dataset is retained for Replay. Finally, the model training process based on incremental learning is realized. This construction method can simulate real-world scenarios in machine learning where the model gradually forgets old knowledge, introduces new data, and deals with noise.

### 数据集构造操作

```bash
# 生成对称噪声数据集
python gen_noise/gen_cifar10_noise.py --noise_type symmetric --noise_ratio 0.2 --num_versions 3 --retention_ratios 0.5 0.3 0.1

# 训练初始模型 Mp0
python run_experiment.py --step 0 --noise_ratio 0.2 --noise_type symmetric

# 训练增量模型 Mp1
python run_experiment.py --step 1 --noise_ratio 0.2 --noise_type symmetric

# 训练迭代增量模型 Mp2
python run_experiment.py --step 2 --noise_ratio 0.2 --noise_type symmetric
```

```bash
# 生成非对称噪声数据集
python gen_noise/gen_cifar10_noise.py --noise_type asymmetric --noise_ratio 0.2 --num_versions 3 --retention_ratios 0.5 0.3 0.1

# 训练初始模型 Mp0
python run_experiment.py --step 0 --noise_ratio 0.2 --noise_type asymmetric

# 训练增量模型 Mp1
python run_experiment.py --step 1 --noise_ratio 0.2 --noise_type asymmetric

# 训练迭代增量模型 Mp2
python run_experiment.py --step 2 --noise_ratio 0.2 --noise_type asymmetric

# OR 可以执行模型路径
python run_experiment.py --step 2 --noise_ratio 0.2 --noise_type asymmetric --load_model ./ckpt/nr_0.2_nt_asymmetric/model_p1.pth

```

**验证训练效果。**

```bash

python test_model.py \
    --model_path ./ckpt/nr_0.2_nt_asymmetric/model_p0.pth \
    --data_dir ./data/cifar-10/noise/nr_0.2_nt_asymmetric \
    --batch_size 64 \
    --seed 42

python test_model.py \
    --model_path ./ckpt/nr_0.2_nt_asymmetric/model_p1.pth \
    --data_dir ./data/cifar-10/noise/nr_0.2_nt_asymmetric \
    --batch_size 64 \
    --seed 42

python test_model.py \
    --model_path ./ckpt/nr_0.2_nt_asymmetric/model_p2.pth \
    --data_dir ./data/cifar-10/noise/nr_0.2_nt_asymmetric \
    --batch_size 64 \
    --seed 42

```

You can download the above files from baidu net disk.

```bash
Link: https://pan.baidu.com/s/1iKwmQC94GlpmvY3LteS-7g?pwd=tnpk
Code: tnpk
```

## Preparation

### Download datasets

Currently supported datasets(*9/19 pm*) are:

1. **CIFAR-10**
2. **CIFAR-100**
3. The other datasets (**TinyImageNet**, **WebVision**, **Clothing1M**) are under development.

```bash

$ mkdir -p data

$ ll data/
cifar-10
cifar-100
tiny-imagenet-200
Clothing1M
Mini-WebVision
```

**Reference:**

1. https://github.com/sangamesh-kodge/Mini-WebVision
2. https://github.com/sangamesh-kodge/Clothing1M

### Creating a virtual environment

```bash
conda create -n tta-mr python=3.8
conda activate tta-mr
```

```bash
pip install -r requirements.txt
```

## Forget Class

### Usage

```bash

$ python main.py --help
usage: main.py [-h] --dataset {cifar-10,cifar-100,flowers-102,tiny-imagenet-200} --model {resnet18,vgg16} [--pretrained] --condition
               {original_data,remove_data,noisy_data,all_perturbations} [--classes_remove CLASSES_REMOVE [CLASSES_REMOVE ...]] [--remove_fraction REMOVE_FRACTION]
               [--classes_noise CLASSES_NOISE [CLASSES_NOISE ...]] [--noise_type {gaussian,salt_pepper}] [--noise_fraction NOISE_FRACTION] [--gpu GPU] [--batch_size BATCH_SIZE]
               [--learning_rate LEARNING_RATE] [--optimizer {sgd,adam}] [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] [--num_epochs NUM_EPOCHS]
               [--early_stopping_patience EARLY_STOPPING_PATIENCE] [--early_stopping_accuracy_threshold EARLY_STOPPING_ACCURACY_THRESHOLD] [--use_early_stopping]
               [--kwargs [KWARGS [KWARGS ...]]]

Run experiments with different datasets, models, and conditions.

optional arguments:
  -h, --help            show this help message and exit
  --dataset {cifar-10,cifar-100,flowers-102,tiny-imagenet-200}
                        Dataset name, choose from: cifar-10, cifar-100, flowers-102, tiny-imagenet-200
  --model {resnet18,vgg16}
                        Model name, choose from: resnet18, vgg16
  --pretrained          If specified, use pretrained weights for the model
  --condition {original_data,remove_data,noisy_data,all_perturbations}
                        Condition for the experiment: original_data, remove_data, noisy_data, all_perturbations
  --classes_remove CLASSES_REMOVE [CLASSES_REMOVE ...]
                        List of classes to remove samples from, e.g., --classes_remove 0 1 2 3 4 or 0-4
  --remove_fraction REMOVE_FRACTION
                        Fraction of samples to remove from the selected classes, e.g., --remove_fraction 0.5 for 50% removal (default: 0.5)
  --classes_noise CLASSES_NOISE [CLASSES_NOISE ...]
                        List of classes to add noise to, e.g., --classes_noise 5 6 7 8 9 or 5-9
  --noise_type {gaussian,salt_pepper}
                        Type of noise to add to the selected classes, e.g., --noise_type gaussian or --noise_type salt_pepper (default: gaussian)
  --noise_fraction NOISE_FRACTION
                        Fraction of samples in the selected classes to add noise to, e.g., --noise_fraction 0.1 for 10% noise injection (default: 0.8)
  --gpu GPU             Specify the GPU(s) to use, e.g., --gpu 0,1 for multi-GPU or --gpu 0 for single GPU
  --batch_size BATCH_SIZE
                        Batch size for training (default: 64)
  --learning_rate LEARNING_RATE
                        Learning rate for the optimizer (default: 0.001)
  --optimizer {sgd,adam}
                        Optimizer for training weights
  --momentum MOMENTUM   Momentum for SGD optimizer (default: 0.9). Only used if optimizer is 'sgd'.
  --weight_decay WEIGHT_DECAY
                        Weight decay for the optimizer (default: 0.0001).
  --num_epochs NUM_EPOCHS
                        Number of epochs to train the model (default: 200)
  --early_stopping_patience EARLY_STOPPING_PATIENCE
                        Patience for early stopping (default: 10)
  --early_stopping_accuracy_threshold EARLY_STOPPING_ACCURACY_THRESHOLD
                        Accuracy threshold for early stopping (default: 0.95)
  --use_early_stopping  Enable early stopping if specified, otherwise train for the full number of epochs
  --kwargs [KWARGS [KWARGS ...]]
                        Additional key=value arguments for hyperparameters

```

### Example run (CIFAR10 with 50% removal)

```bash
python main.py --dataset cifar-10 --model resnet18 --condition remove_data --classes_remove 0 1 2 3 4
```

## Add Noise

### Supported Noise Label types

1. **sym**: Symmetric noise
2. **asym**: Asymmetric noise
3. **pair_flip**: Pairwise noise

Explanations of the above types:

**Symmetric noise** is when the probability of a label being flipped is the same for all classes.

**Asymmetric noise** is when the probability of a label being flipped is different for each class.

**Pairwise noise** is when the probability of a label being flipped is different for each pair of classes.

### Usage

### Example Run

## Acknowledgements

- https://github.com/nazmul-karim170/UNICON-Noisy-Label
- https://github.com/LiJunnan1992/DivideMix
- https://github.com/sangamesh-kodge/LabelNoiseRobustness
- https://github.com/sangamesh-kodge/Mini-WebVision
- https://github.com/sangamesh-kodge/Clothing1M
