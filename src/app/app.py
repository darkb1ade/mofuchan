import gradio as gr

def gradio_chat(input_text, history):
    output_text = "User said: " + input_text
    history.append((input_text, output_text))
    return history, history



def main():
    gr.ChatInterface(fn=gradio_chat, 
                     chatbot=gr.Chatbot(height=300),
                     textbox=gr.Textbox()
                     )
    




if __name__ == "__main__":
    main()