# finetune
python train_gsm8k_gact.py \
    --model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/loftq/llama-4bit-16rank \
    --learning_rate 3e-4 \
    --seed 11 \
    --expt_name gsm8k_llama_7b_4bit_64rank_loftq_L1 \
    --output_dir exp_results/ \
    --num_train_epochs 6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --do_train \
    --report_to tensorboard \
    --gact \
    --gact_level L1

# # # test
# python test_gsm8k.py \
#     --model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/loftq/llama-2bit-16rank \
#     --adapter_name_or_path /home/yujin-wa20/projects/LoftQ/exp_results/gsm8k_llama_7b_2bit_64rank_loftq/llama-2bit-16rank/ep_6/lr_0.0003/seed_11 \
#     --batch_size 64