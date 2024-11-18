# 设置指定的 GPU
export CUDA_VISIBLE_DEVICES=$1
echo "Using GPU $1"
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"

# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

# python ./core_model/train_teacher.py --model wideresnet50 --dataset pet-37 --num_epochs 20 --step 0 --learning_rate 0.0001 --optimizer adam --batch_size 64 --balanced

# python ./core_model/train_teacher.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 200 --step 0 --learning_rate 0.015 --optimizer adam --batch_size 256 --balanced

# python ./core_model/train_teacher.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 100 --step 0 --learning_rate 0.0001 --optimizer adam --batch_size 256 --balanced --noise_type asymmetric

# python ./core_model/train_teacher.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 100 --step 0 --learning_rate 0.0001 --optimizer adam --batch_size 128 --balanced --noise_type asymmetric

# python ./core_model/train_teacher.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 100 --step 0 --learning_rate 0.01 --optimizer adam --batch_size 128 --balanced --noise_type asymmetric

# python ./core_model/train_teacher.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 100 --step 0 --learning_rate 0.01 --optimizer adam --batch_size 64 --balanced --noise_type asymmetric

# GPU 4
# can't work, 会在后续掉点
python ./core_model/train_teacher.py --model cifar-wideresnet40 --dataset cifar-100 --num_epochs 100 --step 0 --learning_rate 0.0001 --optimizer adam --batch_size 64 --balanced --noise_type asymmetric