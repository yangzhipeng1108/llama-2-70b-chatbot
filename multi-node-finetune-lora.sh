#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=/root/nas-share/chat/llama/output
ZERO_STAGE=3


deepspeed --hostfile /root/nas-share/chat/llama/hostfile  /root/nas-share/chat/llama/main.py \
   --model_name_or_path /root/nas-share/chat/llama/model/models--daryl149--llama-2-7b-chat-hf/snapshots/bbc9b373dacff93e600e4426f2b3d3dd264e90ed \
   --tokenizer_name /root/nas-share/chat/llama/merged_tokenizer_hf \
   --train_files /root/nas-share/chat/llama/alpaca_gpt4_data_zh.json \
   --validation_files  /root/nas-share/chat/llama/trans_chinese_alpaca_data.json \
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

