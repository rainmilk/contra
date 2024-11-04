# 设置指定的 GPU
export CUDA_VISIBLE_DEVICES=$1
echo "Using GPU $1"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python ./run_experiment_cvpr.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 500 --train_mode pretrain --learning_rate 5e-2 --optimizer adam --batch_size 256 --noise_type asymmetric --noise_ratio 0.5 --data_aug