# finetune
tag=WALSH_1D_NAIVE
exp_name=gsm8k_mistral_7b_4bit_64rank_loftq_${tag}
python -u train_gsm8k_gact.py \
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
    --linear_quality 75 \
    --nonlinear_mode NAIVE \
    --nonlinear_quantization_shape 16

# # # # # test
python test_gsm8k.py \
    --model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/loftq/Mistral-7B-v0.1-4bit-16rank \
    --adapter_name_or_path /home/yujin-wa20/projects/LoftQ/exp_results/gsm8k_mistral_7b_4bit_64rank_loftq_${tag}/gsm8k_mistral_7b_4bit_64rank_loftq_${tag}/Mistral-7B-v0.1-4bit-16rank/ep_6/lr_0.0003/seed_11 \
    --batch_size 64