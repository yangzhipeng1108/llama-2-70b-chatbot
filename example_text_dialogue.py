# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama
import os
import torch
import torch.distributed as dist

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(global_rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.distributed.barrier()

    generator.model.to(device)


    print("欢迎使用 llama2 模型，输入内容即可进行对话，stop 终止程序")
    while True:
        prompts = []
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank == 0:
            query = input("\nuser:")
            print(query)
            if query.strip() == "stop":
                break

            prompts = [query]
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            for prompt, result in zip(prompts, results):
                print(prompt)
                print(f"> {result['generation']}")
                print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
