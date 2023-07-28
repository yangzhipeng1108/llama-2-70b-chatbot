import gradio as gr
import torch
import transformers
from transformers import (
AutoTokenizer,
BitsAndBytesConfig,
AutoModelForCausalLM,
)

model_name = 'inference/model/models--daryl149--llama-2-70b-chat-hf/snapshots/d18da95c7361c87847879c9b281d2566a9ae4242'

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, load_in_4bit=True, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids.cuda(), max_length=30)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return result


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
