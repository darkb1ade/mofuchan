# %%
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
import yaml

with open("config.yaml", "r") as file:  # ./script/
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
# ports, returns = portfolio.backtesting(dfs=data, rebal_dt=rebal_dt)
# 1. Get predicted close price of all assets & optimize
asset_weights, group_weights = portfolio.pred_optimize(
    dfs=data, rebal_dt=rebal_dt, risk_budget=risk_budget
)
# 2. Backtest the result
invest, groundtruth_risk, result, index_level = simulator.backtesting(
    dfs=data, asset_weights=asset_weights, group_weights=group_weights
)
# 3. Get evaluation metrics: Need to summarize to user are sharpe, volatility, max_drawdown
metrics = simulator.get_metrics(index_level=index_level)
print(metrics)
# 4. Whole investment period simulation
sim_port_value = simulator.sim_monte_carlo(
    avg_monthly_return=metrics["avg_monthly_return"],
    monthly_volatility=metrics["monthly_volatility"],
)
# %%
fig1 = plot_backtest_result(
    result=result,
    invest=invest,
    groundtruth_risk=groundtruth_risk,
    index_level=index_level,
)
# %%
fig2 = plot_invest_result(sim_port_value=sim_port_value)

# %%
