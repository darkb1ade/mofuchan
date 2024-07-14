import pandas as pd


def return_(df: pd.DataFrame, columns: list, window_size_list: list, smoother_win: int):
    features = []
    for window_size in window_size_list:
        new_col_names = {
            col: f"{col.lower().replace(' ', '_')}_return({window_size})"
            for col in columns
        }
        feature = (
            df[columns]
            .pct_change(window_size)
            .rolling(smoother_win)
            .mean()
            .rename(columns=new_col_names)
        )
        features.append(feature)
    return pd.concat(features, axis=1)


def vol(df: pd.DataFrame, columns: list, window_size_list: list, smoother_win=None):
    features = []
    for window_size in window_size_list:
        new_col_names = {
            col: f"{col.lower().replace(' ', '_')}_vol({window_size})"
            for col in columns
        }
        feature = df[columns].rolling(window_size).std().rename(columns=new_col_names)
        features.append(feature)
    return pd.concat(features, axis=1)


def daily_return(
    df: pd.DataFrame, columns: list, window_size_list: list, smoother_win=None
):
    features = []
    for window_size in window_size_list:
        new_col_names = {
            col: f"{col.lower().replace(' ', '_')}_daily_return({window_size})"
            for col in columns
        }
        feature = (
            df[columns]
            .pct_change(1)
            .rolling(window_size)
            .mean()
            .rename(columns=new_col_names)
        )
        features.append(feature)
    return pd.concat(features, axis=1)


def moving_average(
    df: pd.DataFrame, columns: list, window_size_list: list, smoother_win=None
):
    features = []
    for window_size in window_size_list:
        new_col_names = {
            col: f"{col.lower().replace(' ', '_')}_ma({window_size})" for col in columns
        }
        feature = df[columns].rolling(window_size).mean().rename(columns=new_col_names)
        features.append(feature)
    return pd.concat(features, axis=1)


def intra_day(
    df: pd.DataFrame,
    columns: list = None,
    window_size_list: list = None,
    smoother_win=None,
):
    assert set(["Close", "Open"]).issubset(
        df.columns
    ), f"Expected 'Close' and 'Open' in dataframe. Found:{(',').join(list(df.columns))}"
    return (df["Close"] - df["Open"]).to_frame("intra_day").rolling(smoother_win).mean()


def prices_range(
    df: pd.DataFrame,
    columns: list = None,
    window_size_list: list = None,
    smoother_win=None,
):
    assert set(["High", "Low"]).issubset(
        df.columns
    ), f"Expected 'High' and 'Low' in dataframe. Found:{(',').join(list(df.columns))}"
    return (df["High"] - df["Low"]).to_frame("range").rolling(smoother_win).mean()
