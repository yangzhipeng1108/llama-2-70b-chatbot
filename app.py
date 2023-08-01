import gradio as gr
import torch
import transformers
from transformers import (
AutoTokenizer,
BitsAndBytesConfig,
AutoModelForCausalLM,
)

model  = 'inference/model/models--daryl149--llama-2-70b-chat-hf/snapshots/d18da95c7361c87847879c9b281d2566a9ae4242'

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    load_in_4bit = True,
    device_map="auto",
)


def generate_text(text):
    sequences = pipeline(
        text,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1024,
    )

    return sequences[0]['generated_text']


examples = [
    # For these prompts, the expected answer is the natural continuation of the prompt
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
    "Building a website can be done in 10 simple steps:\n",
    # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
    """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
    """Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
plush girafe => girafe peluche
cheese =>""",
    ]

gr.Interface(
    generate_text,
    "textbox",
    "text",
    title="LLama2 70B",
    description="LLama2 70B large language model.",
    examples=examples
).queue().launch(share=True, inbrowser=True)
