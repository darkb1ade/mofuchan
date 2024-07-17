from src.core.predictor import Predictor
from src.skfolio_lib.optimization import MeanRisk
from src.skfolio_lib.prior import XGBPrediction
from skfolio.optimization import ObjectiveFunction
from skfolio.optimization import RiskBudgeting
from skfolio import RiskMeasure
from src.core.utils import get_dataset_offset
from src.core.simulator import Simulator
import pandas as pd

OBJECTIVE_FUNCTION = {"max_return": "MAXIMIZE_RETURN"}
RISKMEASURE = {"variance": "VARIANCE", "cvar": "CVAR"}


class Portfolio:
    def __init__(
        self,
        predictor: Predictor,
        asset_min_w: float,
        risk_budget: dict,
        # rebal_start:str,
        # rebal_period:str = "m",
        simulator: Simulator,
        group_risk_measure: str = "cvar",
        objective_func: str = "max_return",
        asset_risk_measure: str = "variance",
    ):
        # self.rebal_dt = get_rebalance_dt(rebal_start = rebal_start, rebal_period = rebal_period)
        self._set_asset_model(
            predictor=predictor,
            min_w=asset_min_w,
            objective_func=objective_func,
            risk_measure=asset_risk_measure,
        )
        self._set_group_model(
            risk_budget=risk_budget,
            risk_measure=group_risk_measure,
        )
        self.simulator = simulator

    def _set_asset_model(
        self,
        predictor: Predictor,
        min_w: float,
        objective_func: str = "max_return",
        risk_measure: str = "variance",
    ):
        self.prior_model = XGBPrediction(predictor=predictor)
        self.asset_model = MeanRisk(
            objective_function=getattr(
                ObjectiveFunction, OBJECTIVE_FUNCTION[objective_func]
            ),
            risk_measure=getattr(RiskMeasure, RISKMEASURE[risk_measure]),
            min_weights=min_w,
            prior_estimator=self.prior_model,
        )

    def _set_group_model(self, risk_budget: dict, risk_measure: str):
        self.risk_budget = risk_budget
        self.group_model = RiskBudgeting(
            risk_measure=getattr(RiskMeasure, RISKMEASURE[risk_measure]),
            risk_budget=risk_budget,
            portfolio_params=dict(name="Risk Budgeting - CVaR"),
        )

    def backtesting(self, dfs: pd.DataFrame, rebal_dt: list):
        group_weighted_return = self._optimize_asset(dfs=dfs, rebal_dt=rebal_dt)
        ports, returns = self._optimize_group_asset(
            dfs=dfs, group_weight=group_weighted_return, rebal_dt=rebal_dt
        )
        return ports, returns

    def _optimize_asset(self, dfs: pd.DataFrame, rebal_dt: list):
        """
        dfs (pd.DataFrame): multicolumn (asset group, asset name, OHLC)
        """
        offset = (
            self.prior_model.predictor.dataset_spliter.offset
            + self.prior_model.predictor.feature.label_lookahead
            + 1
        )
        asset_groups = dfs.columns.get_level_values(0).unique()
        weighted_returns_all = {}
        for asset_group in asset_groups:
            data = dfs.xs(asset_group, axis=1)

            weighted_returns = []
            for dt in rebal_dt:
                print(f"TODAY: {dt}, predict for next month")
                idx = data.index.get_loc(dt)
                x = data.iloc[idx - offset : idx + 1]
                # fit to get the optimized weight using prediction result in prior
                self.asset_model.fit(x, asset_group=asset_group)
                # groundtruth data (future)*optimize weight(for future)
                weighted_return = (
                    data.loc[
                        self.asset_model.prior_estimator_.idx[
                            0
                        ] : self.asset_model.prior_estimator_.idx[-1]
                    ]
                ).xs("Close", axis=1, level=1).pct_change(
                    1
                ).dropna() * self.asset_model.weights_
                weighted_returns.append(weighted_return)
                print(
                    f"result from {self.asset_model.prior_estimator_.idx[0]} ~ {self.asset_model.prior_estimator_.idx[-1]}"
                )
            weighted_returns = pd.concat(weighted_returns)

            weighted_returns_all[asset_group] = weighted_returns
        weighted_returns_all = pd.concat(weighted_returns_all, axis=1)
        weighted_returns_all = weighted_returns_all.rename(
            columns={"bond": "fix_income", "real_estate": "fix_income"}
        )
        return weighted_returns_all.dropna().groupby(axis=1, level=0).sum()

    def _optimize_group_asset(
        self, dfs: pd.DataFrame, group_weight: pd.DataFrame, rebal_dt: list
    ):
        dfs = dfs.rename(columns={"bond": "fix_income", "real_estate": "fix_income"})
        ports, returns = [], []
        for st_dt, end_dt in zip(rebal_dt[:-1], rebal_dt[1:]):
            self.group_model.fit(group_weight[st_dt:end_dt])
            port = self.group_model.predict(group_weight[st_dt:end_dt])
            daily_return = (
                dfs[st_dt:end_dt].xs("Close", axis=1, level=2).pct_change().dropna()
            )
            ports.append(port)
            returns.append(daily_return.groupby(axis=1, level=0).sum() * port.weights)
        return ports, pd.concat(returns)
