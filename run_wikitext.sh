tag=opt-nf4-0.8-0.25-0.5
exp_name=opt-gact_4bit_16rank_loftq_${tag}
model_name=opt-6.7b-4bit-16rank
# train 4-bit 64-rank llama-2-7b on wikitext-2 using 1 GPU
python -u train_clm.py \
  --model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/loftq/${model_name} \
  --learning_rate 3e-4 \
  --seed 11 \
  --expt_name $exp_name \
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
  --drop_bp_enable \
  --maintain_channels 0.8 \
  --dim_sparsity_ratio 0.25 \
  --head_sparsity_ratio 0.5