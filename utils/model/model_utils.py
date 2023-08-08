# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)

from transformers.deepspeed import HfDeepSpeedConfig



def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    disable_dropout=False):
    if  "chatglm" in model_name_or_path:
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    else:
        model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if "chatglm" in model_name_or_path:
        model = AutoModel.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config)

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    #     torch_dtype=torch.float16,
    #     load_in_8bit=True if model_args.load_in_bits == 8 else False,
    #     quantization_config=bnb_config_4bit if model_args.load_in_bits == 4 else bnb_config_8bit,
    #     device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}
    # )  # .half().cuda()

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model
