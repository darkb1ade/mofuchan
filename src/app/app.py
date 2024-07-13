import time
import gradio as gr
import gradio.themes as themes


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
    return [[None, "Hi, Can you tell me what kind of investment do you like?"]]


def main():
    theme = themes.Default()

    title_text = "Mofu-chan Financial Assistant"
    description_text = "Let us know your investment preference"

    # Custom CSS for chatbot background
    custom_css = """
    .chatbot-container {
        background-color: #f0f8ff; /* Light blue color */
    }
    """

    with gr.Blocks(theme=theme, css=custom_css) as demo:
        gr.Markdown(f"<h1 style='text-align: center;'>{title_text}</h1>")
        gr.Markdown(f"<p style='text-align: left;'>{description_text}</p>")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Image(value="src/assets/mofu_logo.png", width=300, show_label=False)
            with gr.Column(scale=10):
                chatbot = gr.Chatbot(
                    height=400,
                    show_label=False,
                    value=[
                        [
                            None,
                            "Hi, Can you tell me what kind of investment do you like?",
                        ]
                    ],
                    elem_classes="chatbot-container",
                )
        with gr.Row():
            user_input = gr.Textbox(
                placeholder="Reply to Mofu-chan...",
                label="",
                container=True,
                scale=7,
                interactive=True,
            )
            submit_btn = gr.Button("Submit", variant="primary")

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
