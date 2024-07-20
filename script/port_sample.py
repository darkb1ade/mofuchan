# %%
from src.core.utils import get_rebalance_dt, read_input_file
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
portfolio = Portfolio(
    predictor=predictor, asset_min_w=0.05, risk_budget=risk_budget, simulator=simulator
)
# ports, returns = portfolio.backtesting(dfs=data, rebal_dt=rebal_dt)
# 1. Get predicted close price of all assets
preds = portfolio.predict(dfs=data, rebal_dt=rebal_dt)
# TODO: concat groundtruth with prediction to optimize
asset_weights, group_weights = portfolio.optimize(preds=preds, rebal_dt=rebal_dt)
invest, groundtruth_risk, result, index_level = simulator.backtesting(
    dfs=data, asset_weights=asset_weights, group_weights=group_weights
)
metrics = simulator.get_metrics(index_level=index_level)
sim_port_value = simulator.sim_monte_carlo(
    avg_monthly_return=metrics["avg_monthly_return"],
    monthly_volatility=metrics["monthly_volatility"],
)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(
    result.index,
    result,
    color="red",
    label="with AI mofu",
)
axes[0].plot(
    invest.index,
    invest,
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
index_level.plot(ax=axes[1], label="AI Mofu return")
# %%
sim_port_uncertainty = sim_port_value["risk"] * 2
plt.plot(sim_port_value.index, sim_port_value["values"], label="With AI")
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
    label="without AI",
    drawstyle="steps-post",
)
plt.legend()


# %%
