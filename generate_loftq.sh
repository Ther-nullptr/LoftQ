SAVE_DIR="model_zoo/loftq/"

python quantize_save.py \
    --model_name_or_path /home/yujin-wa20/projects/aliendao/dataroot/models/mistralai/Mistral-7B-v0.1 \
    --bits 2 \
    --iter 5 \
    --rank 16 \
    --save_dir $SAVE_DIR