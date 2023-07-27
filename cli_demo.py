

import gradio as gr


from llama import Llama



generator = Llama.build(
    ckpt_dir='llama-2-70b',
    tokenizer_path='tokenizer.model',
    max_seq_len=1024,
    max_batch_size=4,
)


def predict(input, chatbot, max_length, top_p, temperature,history):
    text = [input]
    chatbot = generator.text_completion(
        text,
        max_gen_len=max_length,
        temperature=top_p,
        top_p=temperature,
    )

    return chatbot,history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">欢迎使用 MOSS 人工智能助手！</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(
                0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.9, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(
                0, 1, value=0.6, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])  # (message, bot_message)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(server_name='0.0.0.0',share=True, inbrowser=True)
