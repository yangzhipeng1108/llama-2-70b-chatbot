output_model=/root/llama/llama-2-70b-chatbot/output
if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi
deepspeed --hostfile /root/llama/llama-2-70b-chatbot/hostfile   /root/llama/llama2-lora-fine-tuning/finetune-lora-ds.py \
    --model_name_or_path /root/llama/llama2-lora-fine-tuning/model/models--daryl149--llama-2-7b-chat-hf/snapshots/bbc9b373dacff93e600e4426f2b3d3dd264e90ed  \
    --tokenizer_name /root/llama/llama2-lora-fine-tuning/merged_tokenizer_hf \
    --train_files /root/llama/llama2-lora-fine-tuning/data/alpaca_gpt4_data_zh.json \
    --validation_files  /root/llama/llama2-lora-fine-tuning/data/trans_chinese_alpaca_data.json \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer true \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 800 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --warmup_steps 400 \
    --load_in_bits 8 \
    --lora_r 8 \
    --lora_alpha 16 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 200 \
    --eval_steps 200 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 1024 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --ignore_data_skip true \
    --ddp_timeout 18000000
