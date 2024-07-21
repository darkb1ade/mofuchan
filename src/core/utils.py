import pandas as pd
import os
from glob import glob
from typing import Any
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def get_dataset_offset(feature_config: dict[str, dict[str, Any]]):
    max_offset = 0
    for config in feature_config.values():
        offset = 0
        if isinstance(config, dict):
            window_size_list = config.get("window_size_list", None)
            smoother_win = config.get("smoother_win", None)
            if window_size_list is not None:
                offset += max(window_size_list)
            if smoother_win is not None:
                offset += smoother_win
            if offset > max_offset:
                max_offset = offset
    return max_offset


def get_rebalance_dt(df: pd.DataFrame, period: str = "m"):
    dt = pd.DataFrame(list(df.index), index=df.index, columns=["date"])
    rebalance_date = dt.groupby(dt["date"].dt.to_period(period)).idxmax()["date"].values
    return rebalance_date


def read_input_file(path: str, drop_assets: list[str]):
    files = glob(f"{path}/*.csv")
    dfs_dict = {}
    for file in files:
        df = pd.read_csv(file, index_col=0, header=[0, 1])
        df.index = pd.to_datetime(df.index)
        asset_group = os.path.splitext(os.path.basename(file))[0]
        dfs_dict[asset_group] = df
    dfs = pd.concat(dfs_dict, axis=1)
    dfs = dfs.drop(drop_assets, axis=1, level=1)
    return dfs


def plot_backtest_result(
    result: pd.Series,
    invest: pd.Series,
    groundtruth_risk: pd.Series,
    index_level: pd.Series,
    figsize: tuple[int] = (10, 7),
):
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes[0].title.set_text("Backtesting Result")
    axes[0].set_ylabel("Investment Growth")
    axes[0].plot(
        result.index,
        result,
        color="red",
        label="with AI mofu",
    )

    axes[0].plot(
        invest[invest != 0].index,
        invest[invest != 0],
        label="without AI",
        drawstyle="steps-post",
    )
    uncertainty = 3 * (groundtruth_risk * invest).dropna()
    axes[0].fill_between(
        invest.index,
        invest - uncertainty,
        invest + uncertainty,
        alpha=0.3,
        color="orange",
        label="Uncertainty",
    )
    axes[1].set_ylabel("Accumulative Return (%)")
    index_level.plot(ax=axes[1], label="AI Mofu return")
    return fig


def plot_invest_result(sim_port_value: pd.DataFrame, figsize: tuple[int] = (10, 7)):
    fig = plt.figure(figsize=figsize)
    sim_port_uncertainty = sim_port_value["risk"] * 2
    plt.plot(sim_port_value.index, sim_port_value["values"], label="With mofu-AI")
    plt.fill_between(
        sim_port_value.index,
        sim_port_value["values"] - sim_port_uncertainty,
        sim_port_value["values"] + sim_port_uncertainty,
        alpha=0.3,
        color="orange",
        label="Uncertainty",
    )
    plt.plot(
        sim_port_value.index,
        sim_port_value["invest"],
        label="without mofu-AI",
        drawstyle="steps-post",
    )
    plt.legend()
    plt.title("Investment Simulation")
    plt.xlabel("Date")
    plt.ylabel("Investment Growth")
    return fig
