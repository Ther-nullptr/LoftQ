# finetune
tag=ALL_8BIT_DCT_WITH_GEMM_HADAMARD
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
    --linear_mode JPEG \
    --linear_quality 75 \
    --silu_mode JPEG \
    --silu_quality 75 \
    --layernorm_mode NAIVE \
    --layernorm_quality 75 \
    --layernorm_quantization_shape 16 \
    --softmax_mode NAIVE \
    --softmax_quantization_shape 16 \
    --softmax_pruning \
    --softmax_pruning_val -100 \
    --softmax_quality 75 \
    --gemm_mode JPEG \
    --gemm_quality 75 \
    --gemm_quantization_shape 64 \
    --hadamard_mode JPEG \
    --hadamard_quality 75 \
    --hadamard_quantization_shape 64