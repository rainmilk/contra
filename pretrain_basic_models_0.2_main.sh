mkdir -p logs

nohup bash -c "CUDA_VISIBLE_DEVICES=4 python run_experiment.py \
    --step 0 \
    --model cifar-resnet18 \
    --dataset cifar-10 \
    --noise_ratio 0.2 \
    --noise_type symmetric \
    --balanced \
    --num_epochs 100 \
    --learning_rate 0.01 \
    --optimizer adam \
    --batch_size 256" >logs/run_experiment_step0_resnet18_cifar10_$(date +'%Y%m%d_%H%M%S').log 2>&1 &

nohup bash -c "CUDA_VISIBLE_DEVICES=5 python run_experiment.py \
    --step 0 \
    --model wideresnet50 \
    --dataset pet-37 \
    --noise_ratio 0.2 \
    --noise_type symmetric \
    --balanced \
    --num_epochs 30 \
    --learning_rate 0.0001 \
    --optimizer adam \
    --batch_size 64" >logs/run_experiment_step0_wideresnet50_pet37_$(date +'%Y%m%d_%H%M%S').log 2>&1 &

nohup bash -c "CUDA_VISIBLE_DEVICES=6 python run_experiment.py \
    --step 0 \
    --model cifar-wideresnet40 \
    --dataset cifar-100 \
    --noise_ratio 0.2 \
    --noise_type asymmetric \
    --balanced \
    --num_epochs 200 \
    --learning_rate 0.02 \
    --optimizer adam \
    --batch_size 256" >logs/run_experiment_step0_wideresnet40_cifar100_$(date +'%Y%m%d_%H%M%S').log 2>&1 &

nohup bash -c "CUDA_VISIBLE_DEVICES=7 python run_experiment.py \
   --step 0 \
   --model wideresnet50 \
   --dataset pet-37 \
   --noise_ratio 0.2 \
   --noise_type asymmetric \
   --balanced \
   --num_epochs 30 \
   --learning_rate 0.0001 \
   --optimizer adam \
   --batch_size 64" >logs/run_experiment_step0_wideresnet50_pet37_asymmetric_$(date +'%Y%m%d_%H%M%S').log 2>&1 &
