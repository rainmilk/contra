# $env:PYTHONPATH += ($pwd).Path  # Powershell
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH is set to: $PYTHONPATH"

python ./run_experiment.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 1 --step 0 --learning_rate 0.01 --optimizer adam --batch_size 256 --balanced

python ./core_model/train_teacher.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 1 --step 0 --learning_rate 0.005 --optimizer adam --batch_size 256 --balanced

python ./core_model/core.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 1 --step 1 --learning_rate 0.001 --optimizer adam --batch_size 256 --balanced

python ./core_model/core.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 1 --step 2 --learning_rate 0.001 --optimizer adam --batch_size 256 --balanced

python ./core_model/core.py --model cifar-resnet18 --dataset cifar-10 --num_epochs 1 --step 3 --learning_rate 0.001 --optimizer adam --batch_size 256 --balanced
