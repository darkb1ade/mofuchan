import pandas as pd
from src.core.predictor import Predictor
from glob import glob
import os


def _get_dataset_offset(feature_config):
    max_offset = 0
    for _, config in feature_config.items():
        offset = 0
        if "window_size_list" in config:
            offset += max(config["window_size_list"])
        if "smoother_win" in config:
            offset += config["smoother_win"]
        if offset > max_offset:
            max_offset = offset
    return max_offset


PATHOUT = "/workdir/notebook/out_final"
os.makedirs(PATHOUT, exist_ok=True)

PREPROC_CONFIG = {"freq": "d", "remove_weekend": True}
FEATURE_CONFIG = {
    "return": {
        "columns": ["Close", "High", "Low", "Open", "Volume"],
        "window_size_list": [5, 21, 42, 63],
        "smoother_win": 5,
    },
    "volatility": {
        "columns": ["Close", "High", "Low", "Open", "Volume"],
        "window_size_list": [5, 21, 42, 63],
    },
    "ma": {
        "columns": ["Close", "High", "Low", "Open", "Volume"],
        "window_size_list": [5, 21],
    },
    "daily_return": {
        "columns": ["Close", "High", "Low", "Open", "Volume"],
        "window_size_list": [5, 21],
    },
    "range": {
        "smoother_win": 5,
    },
    "intra_day": {
        "smoother_win": 5,
    },
}
DATASET_SPLITER_CONFIG = {
    "train_start": None,
    "test_start": "2024",
    "offset": _get_dataset_offset(FEATURE_CONFIG),
}
TUNER_CONFIG = {
    "offset": _get_dataset_offset(FEATURE_CONFIG),
    "n_trial": 100,
    "valid_ratio": 0.2,
}


def train():
    files = glob("/workdir/notebook/data/*.csv")
    predictor = Predictor(
        pathOut=PATHOUT,
        preproc_config=PREPROC_CONFIG,
        feature_config=FEATURE_CONFIG,
        dataset_spliter_config=DATASET_SPLITER_CONFIG,
        tuner_config=TUNER_CONFIG,
        use_scaler=True,
    )
    data = {}
    for file in files:
        df = pd.read_csv(file, index_col=0, header=[0, 1])
        df.index = pd.to_datetime(df.index)
        asset_group = os.path.splitext(os.path.basename(file))[0]
        data[asset_group] = df
    predictor.train(data)


def predict():
    start_predict = "2024"
    files = glob("/workdir/notebook/data/*.csv")
    dataset_spliter_config = {
        "test_start": start_predict,  # date prediction start
        "offset": _get_dataset_offset(FEATURE_CONFIG),
    }
    predictor = Predictor(
        pathOut=PATHOUT,
        preproc_config=PREPROC_CONFIG,
        feature_config=FEATURE_CONFIG,
        dataset_spliter_config=dataset_spliter_config,
    )
    model_path = f"{PATHOUT}/groupmodel.pkl"

    data = {}
    for file in files:
        df = pd.read_csv(file, index_col=0, header=[0, 1])
        df.index = pd.to_datetime(df.index)
        asset_group = os.path.splitext(os.path.basename(file))[0]
        data[asset_group] = df.loc["2023":]
    result = predictor.predict(data=data, model_path=model_path)
    result.to_csv(
        f"{PATHOUT}/predictions_{result.index[0].strftime('%Y%m%d')}_{result.index[-1].strftime('%Y%m%d')}.csv"
    )


if __name__ == "__main__":
    train()
    predict()
    print("done")
