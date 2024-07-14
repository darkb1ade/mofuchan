import time
import gradio as gr
import gradio.themes as themes

from constant.text import *


def temporary_api(input_message):
    output_message = "User said: " + input_message
    time.sleep(5)
    return output_message


def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(history):
    response_message = temporary_api(history[-1][0])
    history[-1][1] = ""
    for character in response_message:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


def reset_chat():
    return [[None, MOFU_CHAN_INIT_PHRASE]]


def main():
    theme = themes.Default()

    # Custom CSS for chatbot background
    custom_css = """
    .chatbot-container {
        background-color: #f0f8ff; /* Light blue color */
    }
    """
    logo_path = "src/app/assets/mofu_logo.png"

    with gr.Blocks(theme=theme, css=custom_css) as demo:
        gr.Markdown(f"<h1 style='text-align: center;'>{MOFU_CHAN_HEADER}</h1>")
        gr.Markdown(f"<p style='text-align: left;'>{MOFU_CHAN_DESCRIPTION}</p>")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Image(value=logo_path, width=300, show_label=False)
            with gr.Column(scale=10):
                chatbot = gr.Chatbot(
                    height=400,
                    show_label=False,
                    value=reset_chat(),
                    elem_classes="chatbot-container",
                )
        with gr.Row():
            user_input = gr.Textbox(
                placeholder=MOFU_CHAN_TEXTBOX_PLACEHOLDER,
                label="",
                container=True,
                scale=7,
                interactive=True,
            )
            submit_btn = gr.Button("Send", variant="primary")

        submit_click = submit_btn.click(
            user,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot],
            queue=False,
        ).then(bot, chatbot, chatbot)
        submit_enter = user_input.submit(
            user,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot],
            queue=False,
        ).then(bot, chatbot, chatbot)

        reset_btn = gr.Button("Reset")
        reset_btn.click(fn=reset_chat, inputs=None, outputs=chatbot)

        demo.launch()


if __name__ == "__main__":
    main()
