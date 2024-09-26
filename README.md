# TTA-MR

## TODO LIST

```plain


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


```


## Preparation

### Code Structure

```bash

$pwd
/home/xxx/tta-mr/

# under tta-mr dir
$ ll
data/ # 保存所有数据（原始数据和生成数据）
ckpt/ # 保存训练模型
gen_dataset/ # 代码，生成数据
run_experiment.py # 代码，训练模型
.. # other files

$ python gen_dataset/gen_cifar10_exp_data.py --help

$ python gen_dataset/gen_cifar100_exp_data.py --help

```

### Create virtual environment

```bash
conda create -n tta-mr python=3.8
conda activate tta-mr
```

```bash
pip install -r requirements.txt
```

### Download datasets

Supported datasets are:

1. **CIFAR-10**
2. **CIFAR-100**
3. **FOOD-101**

```bash

$ mkdir -p data

$ ll data/
cifar-10
cifar-100
food-101
```

每个数据集的下面都包括`gen`和`normal`两个目录，分别用于存储生成的数据和原始的数据集。

- 生成的数据会自动存储在`gen`目录下。
- 原始下载的数据集手动放在`normal`目录下。

比如对于`CIFAR-10`数据集：

```bash
$ ls data/cifar-10
gen  normal

$ ls data/cifar-10/gen/
nr_0.2_nt_asymmetric  nr_0.2_nt_symmetric

$ ls data/cifar-10/normal/
batches.meta  cifar-10-batches-py  cifar-10-python.tar.gz  clean_0.5000_sym.npz  data_batch_1  data_batch_2  data_batch_3  data_batch_4  data_batch_5  readme.html  test_batch
```

### Construct Experimental Datasets

```bash

# 基于 CIFAR-10 数据集生成对称噪声
python gen_dataset/gen_cifar10_exp_data.py --data_dir ./data/cifar-10/normal --gen_dir ./data/cifar-10/gen/ --noise_type symmetric --noise_ratio 0.2 --num_versions 3 --retention_ratios 0.5 0.3 0.1

# 基于 CIFAR-10 数据集生成非对称噪声
python gen_dataset/gen_cifar10_exp_data.py --data_dir ./data/cifar-10/normal --gen_dir ./data/cifar-10/gen/ --noise_type asymmetric --noise_ratio 0.2 --num_versions 3 --retention_ratios 0.5 0.3 0.1

$ tree data/cifar-10/gen/
data/cifar-10/gen/
├── nr_0.2_nt_asymmetric
│   ├── D_0_labels.npy
│   ├── D_0.npy
│   ├── D_a_labels.npy
│   ├── D_a.npy
│   ├── D_inc_0_data.npy
│   ├── D_inc_0_labels.npy
│   ├── D_tr_data_version_1.npy
│   ├── D_tr_data_version_2.npy
│   ├── D_tr_data_version_3.npy
│   ├── D_tr_labels_version_1.npy
│   ├── D_tr_labels_version_2.npy
│   ├── D_tr_labels_version_3.npy
│   ├── test_data.npy
│   └── test_labels.npy
└── nr_0.2_nt_symmetric
    ├── D_0_labels.npy
    ├── D_0.npy
    ├── D_a_labels.npy
    ├── D_a.npy
    ├── D_inc_0_data.npy
    ├── D_inc_0_labels.npy
    ├── D_tr_data_version_1.npy
    ├── D_tr_data_version_2.npy
    ├── D_tr_data_version_3.npy
    ├── D_tr_labels_version_1.npy
    ├── D_tr_labels_version_2.npy
    ├── D_tr_labels_version_3.npy
    ├── test_data.npy
    └── test_labels.npy
```

The logic of generating dataset and validation codes to check the rightness fo generated datasts are within the following notebook:
`result_analysis/dataset_analysis.ipynb` https://github.com/data-centric-research/tta-mr/blob/5bdddef032ea8167364ba6f05d55e1c68083314e/result_analysis/dataset_analysis.ipynb

### Train Initial Models on Experimental Datasets

```bash

$ python run_experiment.py --help

# 基于 CIFAR-10 D_0数据集训练初始模型, model_p0.pth
CUDA_VISIBLE_DEVICES=1 python run_experiment.py --step 0 --dataset_type cifar-10 --epochs 100 --batch_size 32 --learning_rate 0.001
# load model_p0.pth
# 基于 CIFAR-10 D_tr_1数据集(D_tr_data_version_1.npy + D_tr_labels_version_1.npy)训练初始模型, , model_p1.pth
CUDA_VISIBLE_DEVICES=2 python run_experiment.py --step 1 --dataset_type cifar-10 --epochs 50 --batch_size 32 --learning_rate 0.001
# load model_p1.pth (当step>=2时，支持从外部load模型而不是上一个训练好的模型)
# 基于 CIFAR-10 D_tr_2数据集训练初始模型, model_p2.pth
CUDA_VISIBLE_DEVICES=3 python run_experiment.py --step 2 --dataset_type cifar-10 --epochs 50 --batch_size 32 --learning_rate 0.001
# load model_p2.pth
# 基于 CIFAR-10 D_tr_3数据集训练初始模型, model_p3.pth
CUDA_VISIBLE_DEVICES=4 python run_experiment.py --step 3 --dataset_type cifar-10 --epochs 50 --batch_size 32 --learning_rate 0.001

$ tree ckpt/
ckpt/
├── cifar-10
   ├── nr_0.2_nt_asymmetric
   │   ├── model_p0.pth
   │   ├── model_p1.pth
   │   └── model_p2.pth
       └── model_p3.pth
   └── nr_0.2_nt_symmetric
       ├── model_p0.pth
       ├── model_p1.pth
       └── model_p2.pth
       └── model_p3.pth

```

## Core Experiment

TODO

## Validation

TODO

## Acknowledgements

- <https://github.com/nazmul-karim170/UNICON-Noisy-Label>
- <https://github.com/LiJunnan1992/DivideMix>
- <https://github.com/sangamesh-kodge/LabelNoiseRobustness>
- <https://github.com/sangamesh-kodge/Mini-WebVision>
- <https://github.com/sangamesh-kodge/Clothing1M>
