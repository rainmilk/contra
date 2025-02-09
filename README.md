# Code for the submission "CONTRA: A Continual Train, Refinement and Adaptation Framework for Building Ever-robust Web Image Recognition Systems"

## 1. Core code for CONTRA implementation
/core/contra.py: The core code for the implementation of the CONTRA framework

## 2. Reproducing Examples
The following steps show a reproducing example on the Oxford IIIT Pets dataset.

### 2.1 Data Preparation
Generate train and test datasets for Oxford IIIT Pets with asymmetric noise
```bash
python gen_dataset/gen_pet37_exp_data.py \
    --data_dir ./data/pet-37/normal/oxford-pets/ \
    --gen_dir ./data/pet-37/gen/ \
    --noise_type asymmetric \
    --noise_ratio 0.1 \
    --num_versions 3 \
    --retention_ratios 0.5 0.3 0.1 \
    --balanced
```

### 2.2 Run CONTRA Framework
Run the CONTRA framework over the generated datasets for Stage 0 (Pre-training), and Stage 1 to 3 (incremental train and TTA are included for each stage)
```bash
python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --step 0 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --noise_type asymmetric  --uni_name contra

python ./core/train_teacher.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --step 0 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --noise_type asymmetric  --uni_name contra

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --step 1 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --model_suffix worker_raw --noise_type asymmetric  --uni_name contra

python ./core/core.py --model wideresnet50 --dataset pet-37 --num_epochs 1 --step 1 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --noise_type asymmetric  --uni_name contra

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --step 2 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --model_suffix worker_raw --noise_type asymmetric  --uni_name contra

python ./core/core.py --model wideresnet50 --dataset pet-37 --num_epochs 1 --step 2 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --noise_type asymmetric  --uni_name contra

python ./run_experiment.py --model wideresnet50 --dataset pet-37 --num_epochs 10 --step 3 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --model_suffix worker_raw --noise_type asymmetric  --uni_name contra

python ./core/core.py --model wideresnet50 --dataset pet-37 --num_epochs 1 --step 3 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --noise_type asymmetric  --uni_name contra
```


