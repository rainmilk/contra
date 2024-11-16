# Directory for saving the training outputs like Models and Logs
OUTPUT_BASE_DIR="/nvme/szh/data/3ai/lips/outputs"

# Directory for saving machine unlearning results
SAVE_DATA_BASE_DIR="/nvme/szh/data/3ai/lips/saved_data"

if [ ! -d "$OUTPUT_BASE_DIR" ]; then
    echo "Directory for saving outputs not found: $OUTPUT_BASE_DIR"
    exit 1
fi
if [ ! -d "$SAVE_DATA_BASE_DIR" ]; then
    echo "Directory for saving MU data directory not found: $SAVE_DATA_BASE_DIR"
    exit 1
fi

python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_fmnist/retrain --unlearn retrain --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/fmnist/resnet18/retrain --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_fmnist/FT --unlearn FT --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/fmnist/resnet18/FT --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_fmnist/GA --unlearn GA --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/fmnist/resnet18/GA --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_fmnist/FF --unlearn fisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/fmnist/resnet18/FF --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_fmnist/IU --unlearn wfisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/fmnist/resnet18/IU --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/resnet18_fmnist/FT_prune --unlearn FT_prune --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --dataset fashionMNIST --arch resnet18 --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/fmnist/resnet18/FT_prune --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_fmnist/retrain --unlearn retrain --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/fmnist/vgg16/retrain --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_fmnist/FT --unlearn FT --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/fmnist/vgg16/FT --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_fmnist/GA --unlearn GA --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/fmnist/vgg16/GA --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_fmnist/IU --unlearn wfisher --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/fmnist/vgg16/IU --shuffle

# python -u main_forget.py --save_dir $OUTPUT_BASE_DIR/vgg16_fmnist/FT_prune --unlearn FT_prune --class_to_replace 1,3,5,7,9 --num_indexes_to_replace 2700 --arch vgg16_bn_lth --dataset fashionMNIST --resume --save_data --save_data_path $SAVE_DATA_BASE_DIR/fmnist/vgg16/FT_prune --shuffle
