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
gemm_quality=75
hadamard_quality=75

tag=opt-6.7b-4bit-16rank-${linear_mode}-layernorm_mode-${layernorm_mode}-softmax_mode-${softmax_mode}-gemm_mode-${gemm_mode}-linear_quality-${linear_quality}-layernorm_quality-${layernorm_quality}-softmax_quality-${softmax_quality}-gemm_quality-${gemm_quality}
exp_name=gsm8k_${tag}
model_name=opt-6.7b-4bit-16rank
# train 4-bit 64-rank llama-2-7b on wikitext-2 using 1 GPU
python -u train_clm.py \
  --model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/loftq/${model_name} \
  --learning_rate 3e-4 \
  --seed 11 \
  --run_name $exp_name \
  --output_dir exp_results/$exp_name/ \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 64 \
  --save_strategy "epoch" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --do_train --do_eval \
  --report_to wandb \
  --block_size 1024 \
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