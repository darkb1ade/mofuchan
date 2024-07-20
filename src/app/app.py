import time
import gradio as gr
import gradio.themes as themes

from src.app.chatbot import MofuChatBot
from src.constant.gui_text import *
from functools import partial

import matplotlib.pyplot as plt

def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(history, api_func):
    response_message = api_func(history[-1][0])
    history[-1][1] = ""
    
    if isinstance(response_message, dict):
        graph_update = response_message
        response_message = "Graph is updated"
    else:
        graph_update = None
    
    for character in response_message:
        history[-1][1] += character
        time.sleep(0.01)
        yield history, graph_update


# Somehow I need one object from gradio here as argument.
def reset_chat(button_obj, chatbot):
    chatbot.reset()
    # initial_question = mofu_bot.chat(None)
    return [[None, MOFU_CHAN_INIT_PHRASE]]

    

def update_graph(data):
    plt.figure()
    plt.plot(data["income_assets"])
    plt.title(data["commodities"])
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    return plt



def main():
    theme = themes.Default()

    # Custom CSS for chatbot background
    custom_css = """
    .chatbot-container {
        background-color: #f0f8ff; /* Light blue color */
    }
    """
    logo_path = "src/app/assets/mofu_logo.png"

    mofu_bot = MofuChatBot()

    # bot_func = partial(bot, api_func=mofu_bot.chat)

    def bot_wrapper(history):
        bot_generator = bot(history, mofu_bot.chat)
        for h, graph_update in bot_generator:
            if graph_update is not None:
                yield h, update_graph(graph_update)
            else:
                yield h, None

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
                    value=reset_chat(None, mofu_bot),
                    elem_classes="chatbot-container",
                )

        with gr.Row():
            user_input = gr.Textbox(
                placeholder=MOFU_CHAN_TEXTBOX_PLACEHOLDER,
                show_label=False,
                container=True,
                scale=7,
                interactive=True,
            )
            submit_btn = gr.Button("Send", variant="primary")


        reset_btn = gr.Button("Reset")
        mofu_input = gr.Textbox(visible=False, value=mofu_bot)
        debug_btn = gr.Button("DEBUG")
        
        graph_output = gr.Plot(label="Prediction")

        
        submit_click = submit_btn.click(
            user,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot],
            queue=False,
        ).then(bot_wrapper, chatbot, [chatbot, graph_output])
        submit_enter = user_input.submit(
            user,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot],
            queue=False,
        ).then(bot_wrapper, chatbot, [chatbot, graph_output])

        reset_btn.click(
            fn=partial(reset_chat, chatbot=mofu_bot), inputs=mofu_input, outputs=chatbot
        )

        debug_btn = gr.Button("DEBUG")
        debug_btn.click(
            fn=mofu_bot.current_bot.get_history,
            inputs=None,
            outputs=None,
        )
        demo.launch()


if __name__ == "__main__":
    main()
