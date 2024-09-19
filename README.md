# TTA-MR

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

## Supported Noise Label types

1. **sym**: Symmetric noise
2. **asym**: Asymmetric noise
3. **pair_flip**: Pairwise noise

Explanations of the above types:

**Symmetric noise** is when the probability of a label being flipped is the same for all classes.

**Asymmetric noise** is when the probability of a label being flipped is different for each class.

**Pairwise noise** is when the probability of a label being flipped is different for each pair of classes.

## Commands Run on CIFAR-X

### Usage

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

### Example run (CIFAR10 with 50% symmetric noise) 

```bash
python Train_cifar.py --dataset cifar-10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'sym' --r 0.5 
```

### Example run (CIFAR100 with 90% symmetric noise) 

```bash
python Train_cifar.py --dataset cifar-100 --num_class 100 --data_path ./data/cifar100 --noise_mode 'sym' --r 0.9 
```

This will throw an error as downloaded files will not be in the proper folder. That is why they must be manually moved to the "data_path".

## TODO Commands Run on TinyImageNet

### Usage

### Example Run (TinyImageNet with 50% symmetric noise)

```bash
python Train_TinyImageNet.py --ratio 0.5
```

## TODO Commands Run on MiniWebVision

### Usage

```bash
$ python Train_webvision.py --help
```

### Example Run (MiniWebVision with 50% symmetric noise)

```bash
# todo
```

## TODO Commands Run on Clothing1M

### Usage

```bash
$ python Train_clothing1M.py --help
```

### Example Run (Clothing1M with 50% symmetric noise)

```bash
# todo
```

## Acknowledgements

- https://github.com/nazmul-karim170/UNICON-Noisy-Label
- https://github.com/LiJunnan1992/DivideMix
- https://github.com/sangamesh-kodge/LabelNoiseRobustness
- https://github.com/sangamesh-kodge/Mini-WebVision
- https://github.com/sangamesh-kodge/Clothing1M
