# finetune
tag=ALL_8BIT_NONE_GELU_FWD_RELU_BWD_REPLACED
exp_name=gsm8k_llama-4bit-16rank_loftq_${tag}
model_name=llama-4bit-16rank
python -u train_gsm8k_gact.py \
    --model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/loftq/${model_name} \
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
    --hadamard_mode NONE \
    --hadamard_quality 75 \
    --hadamard_quantization_shape 64 \
    --relu_replace

python test_gsm8k.py \
    --model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/loftq/${model_name} \
    --adapter_name_or_path /home/yujin-wa20/projects/LoftQ/exp_results/${exp_name}/${exp_name}/${model_name}/ep_6/lr_0.0003/seed_11 \
    --batch_size 64