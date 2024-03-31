# finetune
tag=opt-6.7b-backward-sparse-attention-0.5-ffn-0.1-quant-nf4-2
exp_name=gsm8k_opt-6.7b-backward-sparse_4bit_16rank_loftq_${tag}
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
    --report_to wandb

python test_gsm8k.py \
    --model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/loftq/${model_name} \
    --adapter_name_or_path /home/yujin-wa20/projects/LoftQ/exp_results/${exp_name}/${exp_name}/${model_name}/ep_6/lr_0.0003/seed_11 \
    --batch_size 64