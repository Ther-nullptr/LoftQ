SAVE_DIR="model_zoo/loftq"

python quantize_save.py \
    --model_name_or_path facebook/opt-6.7b \
    --bits 4 \
    --iter 5 \
    --rank 16 \
    --save_dir $SAVE_DIR