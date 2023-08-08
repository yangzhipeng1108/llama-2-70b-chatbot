#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path Dahoas/rm-static \
   --model_name_or_path /root/llama/llama2-lora-fine-tuning/model/models--daryl149--llama-2-7b-chat-hf/snapshots/bbc9b373dacff93e600e4426f2b3d3dd264e90ed \
   --tokenizer_name  ./merged_tokenizer_hf \
   --train_files ./data/alpaca_gpt4_data_zh.json \
   --validation_files  ./data/trans_chinese_alpaca_data.json \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --do_train \
   --do_eval \
   --max_eval_samples 800 \
   --max_seq_len 1024 \
   --block_size 1024 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --num_train_epochs 5 \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
