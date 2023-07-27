# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs = []
    print("欢迎使用 llama2 模型，输入内容即可进行对话，stop 终止程序")

    while True:

        print('输入系统提示信息,提示AI模型按什么形式对话')
        system_info = input("\nsystem:")
        if system_info.split(':')[1] != '':
            dialogs.append({"role": "system", "content": system_info.split(':')[1]},)


        print('输入问答信息, 每个对话以user: 和 assistant: 开始,每个对话单独起一行，assistant为携带上下文信息也可不需要，输入信息以user：行结束')
        endstr = ""  # 重新定义结束符
        str = ""
        for line in iter(input, endstr):  # 每行接收的东西 用了iter的哨兵模式
            str += line + "\n"  # 换行

        for i in str.split('\n')[1:]:
            if i % 2 == 0:
                dialogs.append({"role": "user", "content": system_info.split(':')[1]}, )
            if i % 2 != 0:
                dialogs.append({"role": "assistant", "content": system_info.split(':')[1]}, )


        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
