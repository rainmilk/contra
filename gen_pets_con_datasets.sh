PYTHONPATH=/nvme/szh/code/tta-mr python gen_dataset/gen_pet37_exp_data.py \
    --data_dir ./data/pet-37/normal/oxford-pets/ \
    --gen_dir ./data/pet-37/gen/ \
    --noise_type symmetric \
    --noise_ratio 0.5 \
    --num_versions 3 \
    --retention_ratios 0.5 0.3 0.1 \
    --balanced

PYTHONPATH=/nvme/szh/code/tta-mr python gen_dataset/gen_pet37_exp_data.py \
    --data_dir ./data/pet-37/normal/oxford-pets/ \
    --gen_dir ./data/pet-37/gen/ \
    --noise_type symmetric \
    --noise_ratio 0.3 \
    --num_versions 3 \
    --retention_ratios 0.5 0.3 0.1 \
    --balanced

PYTHONPATH=/nvme/szh/code/tta-mr python gen_dataset/gen_pet37_exp_data.py \
    --data_dir ./data/pet-37/normal/oxford-pets/ \
    --gen_dir ./data/pet-37/gen/ \
    --noise_type symmetric \
    --noise_ratio 0.1 \
    --num_versions 3 \
    --retention_ratios 0.5 0.3 0.1 \
    --balanced

# ___________________

PYTHONPATH=/nvme/szh/code/tta-mr python gen_dataset/gen_pet37_exp_data.py \
    --data_dir ./data/pet-37/normal/oxford-pets/ \
    --gen_dir ./data/pet-37/gen/ \
    --noise_type asymmetric \
    --noise_ratio 0.5 \
    --num_versions 3 \
    --retention_ratios 0.5 0.3 0.1 \
    --balanced

PYTHONPATH=/nvme/szh/code/tta-mr python gen_dataset/gen_pet37_exp_data.py \
    --data_dir ./data/pet-37/normal/oxford-pets/ \
    --gen_dir ./data/pet-37/gen/ \
    --noise_type asymmetric \
    --noise_ratio 0.3 \
    --num_versions 3 \
    --retention_ratios 0.5 0.3 0.1 \
    --balanced

PYTHONPATH=/nvme/szh/code/tta-mr python gen_dataset/gen_pet37_exp_data.py \
    --data_dir ./data/pet-37/normal/oxford-pets/ \
    --gen_dir ./data/pet-37/gen/ \
    --noise_type asymmetric \
    --noise_ratio 0.1 \
    --num_versions 3 \
    --retention_ratios 0.5 0.3 0.1 \
    --balanced
