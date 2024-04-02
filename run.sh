# finetune
linear_mode=JPEG
silu_mode=JPEG
layernorm_mode=NAIVE
softmax_mode=NAIVE
gemm_mode=JPEG
hadamard_mode=NAIVE
linear_quality=30
silu_quality=75
layernorm_quality=75
softmax_quality=75
gemm_quality=50
hadamard_quality=75

tag=opt-6.7b-4bit-16rank-${linear_mode}-layernorm_mode-${layernorm_mode}-softmax_mode-${softmax_mode}-gemm_mode-${gemm_mode}-linear_quality-${linear_quality}-layernorm_quality-${layernorm_quality}-softmax_quality-${softmax_quality}-gemm_quality-${gemm_quality}
exp_name=gsm8k_${tag}
model_name=opt-6.7b-4bit-16rank
python -u train_gsm8k_drop.py \
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
    --transform_bp_enable \
    --linear_mode $linear_mode \
    --silu_mode $silu_mode \
    --layernorm_mode $layernorm_mode \
    --softmax_mode $softmax_mode \
    --gemm_mode $gemm_mode \
    --hadamard_mode $hadamard_mode \
    --linear_quality $linear_quality \
    --silu_quality $silu_quality \
    --layernorm_quality $layernorm_quality \
    --softmax_quality $softmax_quality \
    --gemm_quality $gemm_quality \
    --hadamard_quality $hadamard_quality

python test_gsm8k.py \
    --model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/loftq/${model_name} \
    --adapter_name_or_path /home/yujin-wa20/projects/LoftQ/exp_results/${exp_name}/${exp_name}/${model_name}/ep_6/lr_0.0003/seed_11 \
    --batch_size 64

tag=prosparse-llama-7b-16rank-${linear_mode}-layernorm-${layernorm_mode}-softmax-${softmax_mode}-gemm-${gemm_mode}-hadamard-${hadamard_mode}-linear-${linear_quality}-layernorm-${layernorm_quality}-softmax-${softmax_quality}-gemm-${gemm_quality}-hadamard-${hadamard_quality}
exp_name=gsm8k_${tag}
model_name=prosparse-llama-2-7b-4bit-16rank

python -u train_gsm8k_drop.py \
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
    --transform_bp_enable \
    --linear_mode $linear_mode \
    --silu_mode $silu_mode \
    --layernorm_mode $layernorm_mode \
    --softmax_mode $softmax_mode \
    --gemm_mode $gemm_mode \
    --hadamard_mode $hadamard_mode \
    --linear_quality $linear_quality \
    --silu_quality $silu_quality \
    --layernorm_quality $layernorm_quality \
    --softmax_quality $softmax_quality \
    --gemm_quality $gemm_quality \
    --hadamard_quality $hadamard_quality

python test_gsm8k.py \
    --model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/loftq/${model_name} \
    --adapter_name_or_path /home/yujin-wa20/projects/LoftQ/exp_results/${exp_name}/${exp_name}/${model_name}/ep_6/lr_0.0003/seed_11 \
    --batch_size 64