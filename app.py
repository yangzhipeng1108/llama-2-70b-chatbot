import gradio as gr

from llama import text_completion
from llama import Llama


generator = Llama.build(
    ckpt_dir='llama-2-70b',
    tokenizer_path='tokenizer.model',
    max_seq_len=1024,
    max_batch_size=4,
)

model = generator.model
tokenizer = generator.tokenizer
def generate_text(text):
    text = [text]
    results = text_completion(
        model,tokenizer,
        text,
        max_gen_len=1024,
        temperature=0.6,
        top_p=0.9,
    )

    return results['generation']


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
