import time
import gradio as gr
import gradio.themes as themes

from src.app.chatbot import MofuChatBot
from src.constant.gui_text import *
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

def user(user_message, history):
    return "", history + [[user_message, None]]


def temp_chat(input_text):
    if input_text.lower() == "draw graph":
        return {"Yo": 0}
    else:
        return "User said: " + input_text



class MofuInterface:
    def __init__(self, optimizer=None):
        self.optimizer = optimizer        
        self.chatbot = MofuChatBot()
        self.call_chatbot = self.chatbot.chat
        # self.call_chatbot = temp_chat
        self.theme = themes.Default()
        self.logo_path = "src/app/assets/mofu_logo.png"
        # Custom CSS for chatbot background
        self.custom_css = """
        .chatbot-container {
            background-color: #f0f8ff; /* Light blue color */
        }
        """
        
    def launch(self):

        with gr.Blocks(theme=self.theme, css=self.custom_css) as demo:
            gr.Markdown(f"<h1 style='text-align: center;'>{MOFU_CHAN_HEADER}</h1>")
            gr.Markdown(f"<p style='text-align: left;'>{MOFU_CHAN_DESCRIPTION}</p>")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Image(value=self.logo_path, width=300, show_label=False)
                with gr.Column(scale=10):
                    chatbot = gr.Chatbot(
                        height=400,
                        show_label=False,
                        value=self.reset_chat(None),
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
            mofu_input = gr.Textbox(visible=False, value=self.chatbot)
            debug_btn = gr.Button("DEBUG")
            

            
            submit_click = submit_btn.click(
                user,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot],
                queue=False,
            ).then(self.process_response, chatbot, chatbot)
            submit_enter = user_input.submit(
                user,
                inputs=[user_input, chatbot],
                outputs=[user_input, chatbot],
                queue=False,
            ).then(self.process_response, chatbot, chatbot)

            reset_btn.click(
                fn=self.reset_chat, inputs=mofu_input, outputs=chatbot
            )
            debug_btn.click(
                fn=self.chatbot.current_bot.get_history,
                inputs=None,
                outputs=None,
            )
            demo.launch()

    def process_response(self, history):
        response_message = self.call_chatbot(history[-1][0])
        
        if isinstance(response_message, dict):
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_title("Sine Wave")
            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            
            # fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
            # fig.update_layout(
            #     title="Sine Wave",
            #     xaxis_title="X axis",
            #     yaxis_title="Y axis"
            # )
            
            history[-1][1] = gr.Plot(fig)
            return history
        else:
            history[-1][1] = response_message
            return history
            # history[-1][1] = gr.Plot(fig)
        #     yield history
        # else:
        #     history[-1][1] = ""
        #     for character in response_message:
        #         history[-1][1] += character
        #         time.sleep(0.01)
        #         yield history

    # Somehow I need one object from gradio here as argument.
    def reset_chat(self, button_obj):
        self.chatbot.reset()
        # initial_question = mofu_bot.chat(None)
        return [[None, MOFU_CHAN_INIT_PHRASE]]
    
    
def main():
    app = MofuInterface()
    app.launch()


if __name__ == "__main__":
    main()
