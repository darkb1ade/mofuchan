import pandas as pd
import os
from glob import glob
from typing import Any


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
