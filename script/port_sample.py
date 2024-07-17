from src.core.utils import get_rebalance_dt, read_input_file
from src.core.config import Config
from src.core.predictor import Predictor
from src.core.portfolio import Portfolio
import yaml

with open("./script/config.yaml", "r") as file:
    config = yaml.safe_load(file)

conf = Config(**config)
backtesting_start = "2023-12"
risk_budget = {
    "fix_income": 0.7,
    "commodity": 0.1,
    "currency": 0.1,
    "equity": 0.1,
}

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
rebal_dt = get_rebalance_dt(df=data.loc[backtesting_start:], period="m")
portfolio = Portfolio(
    predictor=predictor, asset_min_w=0.05, risk_budget=risk_budget, simulator=None
)
ports, returns = portfolio.backtesting(dfs=data, rebal_dt=rebal_dt)
