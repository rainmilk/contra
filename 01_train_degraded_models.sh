#!/bin/bash

# 创建 logs 目录（如果不存在）
mkdir -p logs

# 定义每个任务并使用 nohup 后台执行，将输出保存到 logs 文件夹中

# ResNet18 - Original Data
nohup python main.py --dataset cifar-10 --model resnet18 --condition original_data --gpu 0 > logs/cifar10_resnet18_original.log 2>&1 &
nohup python main.py --dataset cifar-100 --model resnet18 --condition original_data --gpu 1 > logs/cifar100_resnet18_original.log 2>&1 &
nohup python main.py --dataset animals-10 --model resnet18 --condition original_data --gpu 2 > logs/animals10_resnet18_original.log 2>&1 &
nohup python main.py --dataset flowers-102 --model resnet18 --condition original_data --gpu 3 > logs/flowers102_resnet18_original.log 2>&1 &

# # VGG16 - Original Data
# nohup python main.py --dataset cifar-10 --model vgg16 --condition original_data --gpu 4 > logs/cifar10_vgg16_original.log 2>&1 &
# nohup python main.py --dataset cifar-100 --model vgg16 --condition original_data --gpu 5 > logs/cifar100_vgg16_original.log 2>&1 &
# nohup python main.py --dataset animals-10 --model vgg16 --condition original_data --gpu 6 > logs/animals10_vgg16_original.log 2>&1 &
# nohup python main.py --dataset flowers-102 --model vgg16 --condition original_data --gpu 7 > logs/flowers102_vgg16_original.log 2>&1 &

# ResNet18 - Combined Condition
nohup python main.py --dataset cifar-10 --model resnet18 --condition combined --classes_remove 0 1 2 3 4 --classes_noise 5 6 7 8 9 --gpu 4 > logs/cifar10_resnet18_combined.log 2>&1 &
nohup python main.py --dataset cifar-100 --model resnet18 --condition combined --classes_remove 0 1 2 3 4 5 6 7 8 9 --classes_noise 10 11 12 13 14 15 16 17 18 19 --gpu 5 > logs/cifar100_resnet18_combined.log 2>&1 &
nohup python main.py --dataset animals-10 --model resnet18 --condition combined --classes_remove 0 1 2 3 4 --classes_noise 5 6 7 8 9 --gpu 6 > logs/animals10_resnet18_combined.log 2>&1 &
nohup python main.py --dataset flowers-102 --model resnet18 --condition combined --classes_remove 0 1 2 3 4 5 6 7 8 9 --classes_noise 10 11 12 13 14 15 16 17 18 19 --gpu 7 > logs/flowers102_resnet18_combined.log 2>&1 &

# VGG16 - Combined Condition
# nohup python main.py --dataset cifar-10 --model vgg16 --condition combined --classes_remove 0 1 2 3 4 --classes_noise 5 6 7 8 9 --gpu 4 > logs/cifar10_vgg16_combined.log 2>&1 &
# nohup python main.py --dataset cifar-100 --model vgg16 --condition combined --classes_remove 0 1 2 3 4 5 6 7 8 9 --classes_noise 10 11 12 13 14 15 16 17 18 19 --gpu 5 > logs/cifar100_vgg16_combined.log 2>&1 &
# nohup python main.py --dataset animals-10 --model vgg16 --condition combined --classes_remove 0 1 2 3 4 --classes_noise 5 6 7 8 9 --gpu 6 > logs/animals10_vgg16_combined.log 2>&1 &
# nohup python main.py --dataset flowers-102 --model vgg16 --condition combined --classes_remove 0 1 2 3 4 5 6 7 8 9 --classes_noise 10 11 12 13 14 15 16 17 18 19 --gpu 7 > logs/flowers102_vgg16_combined.log 2>&1 &

# echo "All tasks have been started and are running in the background. Logs are being saved in the 'logs/' directory."
