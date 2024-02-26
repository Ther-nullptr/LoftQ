exp_name=wikitext_llama_7b_4bit_64rank_loftq_L2.2
# train 4-bit 64-rank llama-2-7b on wikitext-2 using 1 GPU
python -u train_clm.py \
--model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/loftq/llama-4bit-16rank \
--output_dir exp_results/$exp_name/ \
--learning_rate 3e-4  \
--seed 11 \
--dataset_name wikitext \
--dataset_config wikitext-2-raw-v1 \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 16 \
--save_strategy "epoch" \
--weight_decay 0.1 \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--do_train --do_eval \
--logging_steps 50 \
--report_to wandb \
--block_size 1024 \
--expt_name $exp_name


# # train 4-bit 64-rank llama-2-7b on wikitext-2 using 8 GPUs
# accelerate launch train_clm.py \
# --full_precision \
# --model_name_or_path LoftQ/Llama-2-7b-hf-4bit-64rank \
# --output_dir exp_results/wikitext-2/ \
# --learning_rate 3e-4  \
# --seed 11 \
# --dataset_name wikitext \
# --dataset_config wikitext-2-raw-v1 \
# --num_train_epochs 3 \
# --per_device_train_batch_size 2 \
# --per_device_eval_batch_size 4 \
# --gradient_accumulation_steps 4 \
# --save_strategy "epoch" \
# --weight_decay 0.1 \
# --warmup_ratio 0.03 \
# --lr_scheduler_type "cosine" \
# --logging_steps 1 \
# --do_train --do_eval \
# --logging_steps 50 \
# --report_to tensorboard \
# --block_size 1024
