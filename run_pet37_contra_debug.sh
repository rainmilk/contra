$env:PYTHONPATH += ($pwd).Path


python ./run_experiment.py --model resnet18 --dataset pet-37 --num_epochs 10 --step 0 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --uni_name contra

python ./core_model/train_teacher.py --model resnet18 --dataset pet-37 --num_epochs 10 --step 0 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --uni_name contra

python ./run_experiment.py --model resnet18 --dataset pet-37 --num_epochs 10 --step 1 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --model_suffix worker_raw --uni_name contra

python ./core_model/core.py --model resnet18 --dataset pet-37 --num_epochs 1 --step 1 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --uni_name contra

python ./run_experiment.py --model resnet18 --dataset pet-37 --num_epochs 10 --step 2 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --model_suffix worker_raw --uni_name contra

python ./core_model/core.py --model resnet18 --dataset pet-37 --num_epochs 1 --step 2 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --uni_name contra

python ./run_experiment.py --model resnet18 --dataset pet-37 --num_epochs 10 --step 3 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --model_suffix worker_raw --uni_name contra

python ./core_model/core.py --model resnet18 --dataset pet-37 --num_epochs 1 --step 3 --learning_rate 1e-5 --optimizer adam --batch_size 16 --balanced --uni_name contra
