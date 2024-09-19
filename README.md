# tta-mr

### Example Run

#### Creating a virtual environment, run

```bash
pip install -r requirements.txt
```

#### Example run (CIFAR10 with 50% symmetric noise) 

```bash
python Train_cifar.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'sym' --r 0.5 
```

#### Example run (CIFAR100 with 90% symmetric noise) 

```bash
python Train_cifar.py --dataset cifar100 --num_class 100 --data_path ./data/cifar100 --noise_mode 'sym' --r 0.9 
```

This will throw an error as downloaded files will not be in the proper folder. That is why they must be manually moved to the "data_path".

#### Example Run (TinyImageNet with 50% symmetric noise)

```bash
python Train_TinyImageNet.py --ratio 0.5
```