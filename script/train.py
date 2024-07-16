import pandas as pd
from src.core.predictor import Predictor
from src.core.config import Config
from src.core.utils import get_dataset_offset
from glob import glob
import os
import yaml

with open("./script/config.yaml", "r") as file:
    config = yaml.safe_load(file)

conf = Config(**config)
os.makedirs(conf.path_out, exist_ok=True)


def train():
    files = glob(f"{conf.path_in}/*.csv")
    predictor = Predictor(
        path_out=conf.path_out,
        preproc_config=conf.preprocesor,
        feature_config=conf.feature,
        dataset_spliter_config=conf.data_spliter,
        tuner_config=conf.tuner,
        use_scaler=conf.use_scaler,
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
    files = glob(f"{conf.path_in}/*.csv")
    dataset_spliter_config = {
        "test_start": start_predict,  # date prediction start
        "offset": get_dataset_offset(conf.feature),
    }
    predictor = Predictor(
        path_out=conf.path_out,
        preproc_config=conf.preprocesor,
        feature_config=conf.feature,
        dataset_spliter_config=dataset_spliter_config,
    )
    model_path = f"{conf.path_out}/groupmodel.pkl"

    data = {}
    for file in files:
        df = pd.read_csv(file, index_col=0, header=[0, 1])
        df.index = pd.to_datetime(df.index)
        asset_group = os.path.splitext(os.path.basename(file))[0]
        data[asset_group] = df.loc["2023":]
    result = predictor.predict(data=data, model_path=model_path)
    result.to_csv(
        f"{conf.path_out}/predictions_{result.index[0].strftime('%Y%m%d')}_{result.index[-1].strftime('%Y%m%d')}.csv"
    )


if __name__ == "__main__":
    train()
    predict()
    print("done")
