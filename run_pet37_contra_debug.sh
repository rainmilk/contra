$env:PYTHONPATH += ($pwd).Path  # Powershell


python ./run_experiment.py --model resnet18 --dataset pet-37 --num_epochs 10 --step 0 --learning_rate 0.0001 --optimizer adam --batch_size 16 --balanced --uni_name contra

python ./core_model/train_teacher.py --model resnet18 --dataset pet-37 --num_epochs 5 --step 0 --learning_rate 0.0001 --optimizer adam --batch_size 16 --balanced --uni_name contra

python ./run_experiment.py --model resnet18 --dataset pet-37 --num_epochs 5 --step 1 --learning_rate 0.0001 --optimizer adam --batch_size 16 --balanced --model_suffix worker_raw --uni_name contra

python ./core_model/core.py --model resnet18 --dataset pet-37 --num_epochs 2 --step 1 --learning_rate 0.0001 --optimizer adam --batch_size 16 --balanced --uni_name contra

python ./run_experiment.py --model resnet18 --dataset pet-37 --num_epochs 5 --step 2 --learning_rate 0.0001 --optimizer adam --batch_size 16 --balanced --model_suffix worker_raw --uni_name contra

python ./core_model/core.py --model resnet18 --dataset pet-37 --num_epochs 2 --step 2 --learning_rate 0.0001 --optimizer adam --batch_size 16 --balanced --uni_name contra

python ./run_experiment.py --model resnet18 --dataset pet-37 --num_epochs 5 --step 3 --learning_rate 0.0001 --optimizer adam --batch_size 16 --balanced --model_suffix worker_raw --uni_name contra

python ./core_model/core.py --model resnet18 --dataset pet-37 --num_epochs 2 --step 3 --learning_rate 0.0001 --optimizer adam --batch_size 16 --balanced --uni_name contra
