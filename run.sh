# finetune
tag=OPT
exp_name=gsm8k_opt_7b_4bit_64rank_loftq_${tag}
python -u train_gsm8k_gact.py \
    --model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/loftq/opt-6.7b-4bit-16rank \
    --learning_rate 3e-4 \
    --seed 11 \
    --expt_name $exp_name \
    --output_dir exp_results/$exp_name/ \
    --num_train_epochs 6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --do_train \
    --report_to wandb \
    --linear_mode NONE \
    --linear_quality 75 \
    --silu_mode NONE \
    --silu_quality 75 \
    --layernorm_mode NONE \
    --layernorm_quality 75 \
    --layernorm_quantization_shape 16 \
    --softmax_mode NONE \
    --softmax_quantization_shape 16 \
    --softmax_quality 75 \
    --gemm_mode NONE \
    --gemm_quality 75 \
    --gemm_quantization_shape 64 \
    --softmax_pruning \
    --hadamard_mode NONE \
    --hadamard_quality 75 \
    --hadamard_quantization_shape 64