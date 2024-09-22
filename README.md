# TTA-MR

## Data and model

For later experiment, we should firstly prepare the datasets and models.

```bash
# under base directory
 $ cd gen_noise/
 $ python gen_cifar10_noise.py
```

This will generate the following data and model files.

```bash

$ tree export_dir
export_dir
└── cifar-10
    ├── models
    │   ├── incremental_model-cifar10.pth
    │   └── original_model-cifar10.pth
    └── noise
        ├── cifar10_aux_data.npy
        ├── cifar10_aux_labels.npy
        ├── cifar10_forget_class_data.npy
        ├── cifar10_forget_class_labels.npy
        ├── cifar10_inc_data.npy
        ├── cifar10_inc_labels.npy
        ├── cifar10_noisy_other_class_labels.npy
        ├── cifar10_other_class_data.npy
        ├── cifar10_other_class_labels.npy
        ├── cifar10_test_data.npy
        ├── cifar10_test_labels.npy
        ├── cifar10_train_data.npy
        └── cifar10_train_labels.npy
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

## Forgetting Class

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

## Add Noises on Classes

### Supported Noise Label types

1. **sym**: Symmetric noise
2. **asym**: Asymmetric noise
3. **pair_flip**: Pairwise noise

Explanations of the above types:

**Symmetric noise** is when the probability of a label being flipped is the same for all classes.

**Asymmetric noise** is when the probability of a label being flipped is different for each class.

**Pairwise noise** is when the probability of a label being flipped is different for each pair of classes.

### Add noises on classes of CIFAR-X Datasets

#### Usage

```bash
$ python Train_cifar.py --help

usage: Train_cifar.py [-h] [--batch_size BATCH_SIZE] [--lr LR] [--noise_mode NOISE_MODE] [--alpha ALPHA] [--lambda_u LAMBDA_U] [--lambda_c LAMBDA_C] [--T T] [--num_epochs NUM_EPOCHS]
                      [--r R] [--d_u D_U] [--tau TAU] [--metric METRIC] [--seed SEED] [--gpuid GPUID] [--resume RESUME] [--num_class NUM_CLASS] [--data_path DATA_PATH]
                      [--dataset DATASET]

PyTorch CIFAR Training

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        train batchsize
  --lr LR, --learning_rate LR
                        initial learning rate
  --noise_mode NOISE_MODE
  --alpha ALPHA         parameter for Beta
  --lambda_u LAMBDA_U   weight for unsupervised loss
  --lambda_c LAMBDA_C   weight for contrastive loss
  --T T                 sharpening temperature
  --num_epochs NUM_EPOCHS
  --r R                 noise ratio
  --d_u D_U
  --tau TAU             filtering coefficient
  --metric METRIC       Comparison Metric
  --seed SEED
  --gpuid GPUID
  --resume RESUME       Resume from the warmup checkpoint
  --num_class NUM_CLASS
  --data_path DATA_PATH
                        path to dataset
  --dataset DATASET
```

#### Example run (CIFAR10 with 50% symmetric noise) 

```bash
python Train_cifar.py --dataset cifar-10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'sym' --r 0.5 
```

#### Example run (CIFAR100 with 90% symmetric noise)

```bash
python Train_cifar.py --dataset cifar-100 --num_class 100 --data_path ./data/cifar100 --noise_mode 'sym' --r 0.9 
```

This will throw an error as downloaded files will not be in the proper folder. That is why they must be manually moved to the "data_path".

### TODO Add noises on classes of TinyImageNet Dataset

#### Usage

```bash
python Train_TinyImageNet.py --ratio 0.5
```

#### Example Run (TinyImageNet with 50% symmetric noise)

```bash
# todo
```

### TODO Add noises on classes of MiniWebVision dataset

#### Usage

```bash
$ python Train_webvision.py --help
```

#### Example Run (MiniWebVision with 50% symmetric noise)

```bash
# todo
```

### TODO Add noises on classes of Clothing1M Dataset

#### Usage

```bash
$ python Train_clothing1M.py --help
```

#### Example Run (Clothing1M with 50% symmetric noise)

```bash
# todo
```

## Acknowledgements

- https://github.com/nazmul-karim170/UNICON-Noisy-Label
- https://github.com/LiJunnan1992/DivideMix
- https://github.com/sangamesh-kodge/LabelNoiseRobustness
- https://github.com/sangamesh-kodge/Mini-WebVision
- https://github.com/sangamesh-kodge/Clothing1M
