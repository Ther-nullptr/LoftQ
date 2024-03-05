exp_name=none
python -u train_gsm8k_gact_profile.py \
    --model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/loftq/Mistral-7B-v0.1-4bit-16rank \
    --learning_rate 3e-4 \
    --seed 11 \
    --expt_name $exp_name \
    --output_dir exp_results/$exp_name/ \
    --num_train_epochs 6 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --do_train \
    --report_to wandb \
    --linear_mode DCT \
    --nonlinear_mode NAIVE
