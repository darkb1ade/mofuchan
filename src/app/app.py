from dataclasses import dataclass
import time
import gradio as gr
import gradio.themes as themes
import pandas as pd

from src.app.chatbot import MofuChatBot
from src.constant.gui_text import *
from functools import partial

from src.core.utils import (
    get_rebalance_dt,
    read_input_file,
    plot_backtest_result,
    plot_invest_result,
)
from src.core.config import Config
from src.core.predictor import Predictor
from src.core.portfolio import Portfolio
from src.core.simulator import Simulator

import numpy as np
import matplotlib.pyplot as plt

import yaml


def user(user_message, history):
    return "", history + [[user_message, None]]


def temp_chat(input_text):
    if input_text.lower() == "draw graph":
        return {
            "income_assets": 0.2,
            "commodity": 0.1,
            "currency": 0.1,
            "equity": 0.6,
        }
    else:
        return "User said: " + input_text


def enable_row_visibility():
    return gr.Row(visible=True)


def disable_row_visiblity():
    return gr.Row(visible=False)


@dataclass
class SimulatorClass:
    simulator: Simulator
    portfolio: Portfolio
    data: pd.DataFrame
    rebal_dt: pd.DataFrame
    preds: pd.DataFrame


class MofuInterface:
    def __init__(self, simulators):
        ## These are for backtest and portfolio
        self.simulators = simulators
        self.current_portfolio = None

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
        self.enable_chatbot = True
        self.weights = None
        self.number_row = None
        self.chatbot_box = None
        self.textbox_row = None

    def launch(self):

        with gr.Blocks(theme=self.theme, css=self.custom_css) as demo:
            gr.Markdown(f"<h1 style='text-align: center;'>{MOFU_CHAN_HEADER}</h1>")
            gr.Markdown(f"<p style='text-align: left;'>{MOFU_CHAN_DESCRIPTION}</p>")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Image(
                        value=self.logo_path,
                        width=300,
                        show_label=False,
                        show_download_button=False,
                    )
                with gr.Column(scale=10):
                    self.chatbot_box = gr.Chatbot(
                        height=600,
                        show_label=False,
                        value=[[None, MOFU_CHAN_INIT_PHRASE]],
                        elem_classes="chatbot-container",
                        every=5,
                    )

            with gr.Row(visible=True) as self.textbox_row:
                user_input = gr.Textbox(
                    placeholder=MOFU_CHAN_TEXTBOX_PLACEHOLDER,
                    show_label=False,
                    container=True,
                    scale=7,
                    interactive=True,
                )
                submit_btn = gr.Button("Send", variant="primary")

            with gr.Row(visible=False) as self.number_row:
                start_amount_box = gr.Number(
                    value=1000,
                    precision=2,
                    minimum=1,
                    interactive=True,
                    label="Start amount",
                    info="Initial investment money ($)",
                )
                dca_amount_box = gr.Number(
                    value=50,
                    precision=2,
                    minimum=1,
                    interactive=True,
                    label="Dollar Cost Average",
                    info="Amount of money you invest per month ($)",
                )
                period_amount_box = gr.Number(
                    value=24,
                    precision=0,
                    minimum=1,
                    interactive=True,
                    label="Investment time",
                    info="Total investment period (month)",
                )
                confirm_button = gr.Button("Confirm", variant="primary")

            reset_btn = gr.Button("Reset")
            mofu_input = gr.Textbox(visible=False, value=self.chatbot)
            debug_btn = gr.Button("DEBUG")

            submit_click = submit_btn.click(
                user,
                inputs=[user_input, self.chatbot_box],
                outputs=[user_input, self.chatbot_box],
                queue=False,
            ).then(
                self.process_chat_response,
                self.chatbot_box,
                [self.chatbot_box, self.number_row, self.textbox_row],
            )
            submit_enter = user_input.submit(
                user,
                inputs=[user_input, self.chatbot_box],
                outputs=[user_input, self.chatbot_box],
                queue=False,
            ).then(
                self.process_chat_response,
                self.chatbot_box,
                [self.chatbot_box, self.number_row, self.textbox_row],
            )

            reset_btn.click(
                fn=self.reset_chat,
                inputs=mofu_input,
                outputs=[self.chatbot_box, self.number_row, self.textbox_row],
            )
            confirm_button.click(
                fn=self.process_backtesting,
                inputs=[
                    self.chatbot_box,
                    start_amount_box,
                    dca_amount_box,
                    period_amount_box,
                ],
                outputs=[self.chatbot_box],
            )

            debug_btn.click(
                fn=self.chatbot.get_history,
                inputs=None,
                outputs=None,
            )
            demo.launch()

    def process_chat_response(self, history):
        response_message = self.call_chatbot(history[-1][0])

        if isinstance(response_message, dict):
            self.enable_chatbot = False
            final_risk_budget = {
                "fix_income": response_message["income_assets"],
                "commodity": response_message["commodity"],
                "currency": response_message["currency"],
                "equity": response_message["equity"],
            }
            self.current_portfolio = final_risk_budget

            history[-1][-1] = INPUT_NUMBER_TEXT

            return {
                self.chatbot_box: history,
                self.textbox_row: gr.Row(visible=False),
                self.number_row: gr.Row(visible=True),
            }
        else:
            history[-1][1] = response_message
            return {self.chatbot_box: history}

    def process_backtesting(self, history, start_val, dca_val, time_val):
        print(
            f"Final value: {self.current_portfolio}, {start_val}, {dca_val}, {time_val}"
        )

        ask_for_retry = self.weights is None
        if self.weights is None:
            # Optimize and backtest based on current portfolio
            self.weights = self.simulators.portfolio.optimize(
                preds=self.simulators.preds,
                rebal_dt=self.simulators.rebal_dt,
                risk_budget=self.current_portfolio,
            )

        self.simulators.simulator = Simulator(
            st_amount=start_val, dca_amount=dca_val, invest_period=time_val
        )
        invest, groundtruth_risk, result, index_level = (
            self.simulators.simulator.backtesting(
                dfs=self.simulators.data,
                asset_weights=self.weights[0],
                group_weights=self.weights[1],
            )
        )

        # Init simulator again
        metrics = self.simulators.simulator.get_metrics(index_level=index_level)
        sim_port_value = self.simulators.simulator.sim_monte_carlo(
            avg_monthly_return=metrics["avg_monthly_return"],
            monthly_volatility=metrics["monthly_volatility"],
        )

        fig1 = plot_backtest_result(
            result=result,
            invest=invest,
            groundtruth_risk=groundtruth_risk,
            index_level=index_level,
        )
        # %%
        fig2 = plot_invest_result(sim_port_value=sim_port_value)

        result_summary = self.call_chatbot(
            f"I run some investment simulation and get following result. Can you summarize in a way easy for beginners to understand? Please explain one by one in details: \n"
            f"'annualize sharpe ratio': {metrics['annualize sharpe ratio']}\n 'annualize return': {metrics['annualize return']}\n 'annualize volatility: {metrics['annualize volatility']}\n"
            f"Only give me summarize. No extra response!"
        )

        history.append([None, gr.Plot(fig1)])
        history.append([None, gr.Plot(fig2)])
        history.append([None, result_summary])
        if ask_for_retry:
            history.append([None, BACKTEST_ASK_FOR_REQUEST])
        return history

    # Somehow I need one object from gradio here as argument.
    def reset_chat(self, button_obj):
        self.chatbot.reset()
        self.enable_chatbot = True
        self.weights = None
        # initial_question = mofu_bot.chat(None)
        return {
            self.chatbot_box: [[None, MOFU_CHAN_INIT_PHRASE]],
            self.number_row: gr.Row(visible=False),
            self.textbox_row: gr.Row(visible=True),
        }


def init_model():
    with open("/workdir/script/config.yaml", "r") as file:  # ./script/
        config = yaml.safe_load(file)

    conf = Config(**config)
    backtesting_start = "2023-12"
    risk_budget = {
        "fix_income": 0.1,
        "commodity": 0.1,
        "currency": 0.1,
        "equity": 0.7,
    }
    user_request = {"st_amount": 10000, "dca_amount": 5000, "invest_period": 12 * 5}
    data = read_input_file(path=conf.path_in, drop_assets=["GBTC"])
    dataset_spliter_config = {
        "test_start": None,  # date prediction start
        "offset": conf.data_spliter["offset"],
    }
    predictor = Predictor(
        path_out=conf.path_out,
        preproc_config=conf.preprocessor,
        feature_config=conf.feature,
        dataset_spliter_config=dataset_spliter_config,
        model_path=f"{conf.path_out}/groupmodel.pkl",
    )
    simulator = Simulator(**user_request)
    rebal_dt = get_rebalance_dt(df=data.loc[backtesting_start:], period="m")
    portfolio = Portfolio(predictor=predictor, asset_min_w=0.1, asset_max_w=0.5)
    preds = portfolio.predict(dfs=data, rebal_dt=rebal_dt)
    return SimulatorClass(simulator, portfolio, data, rebal_dt, preds)


def main():
    simulators = init_model()
    app = MofuInterface(simulators)
    app.launch()


if __name__ == "__main__":
    main()
