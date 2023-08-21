#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=/root/llama/llama-2-70b-chatbot/output
ZERO_STAGE=3


deepspeed --hostfile /root/llama/llama-2-70b-chatbot/hostfile  /root/llama/llama-2-70b-chatbot/main.py \
   --model_name_or_path /root/llama/llama2-lora-fine-tuning/model/models--daryl149--llama-2-7b-chat-hf/snapshots/bbc9b373dacff93e600e4426f2b3d3dd264e90ed \
   --tokenizer_name /root/llama/llama2-lora-fine-tuning/merged_tokenizer_hf \
   --train_files /root/llama/llama2-lora-fine-tuning/data/alpaca_gpt4_data_zh.json \
   --validation_files  /root/llama/llama2-lora-fine-tuning/data/trans_chinese_alpaca_data.json \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --do_train True \
   --do_eval True \
   --max_eval_samples 800 \
   --max_seq_len 1024 \
   --block_size 1024 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --num_train_epochs 5 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
