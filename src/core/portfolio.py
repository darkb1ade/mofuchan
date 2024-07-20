from src.core.predictor import Predictor
from skfolio.optimization import MeanRisk
from skfolio.optimization import ObjectiveFunction
from skfolio.optimization import RiskBudgeting
from skfolio import RiskMeasure
from src.core.simulator import Simulator
import pandas as pd
from tqdm import tqdm

OBJECTIVE_FUNCTION = {"max_return": "MAXIMIZE_RETURN", "max_ratio": "MAXIMIZE_RATIO"}
RISKMEASURE = {"variance": "VARIANCE", "cvar": "CVAR"}


class Portfolio:
    def __init__(
        self,
        predictor: Predictor,
        asset_min_w: float,
        asset_max_w: float,
        risk_budget: dict = None,
        group_risk_measure: str = "cvar",
        objective_func: str = "max_return",
        asset_risk_measure: str = "variance",
    ):
        # self.rebal_dt = get_rebalance_dt(rebal_start = rebal_start, rebal_period = rebal_period)
        self._set_asset_model(
            predictor=predictor,
            min_w=asset_min_w,
            max_w=asset_max_w,
            objective_func=objective_func,
            risk_measure=asset_risk_measure,
        )
        self._set_group_model(
            risk_budget=risk_budget,
            risk_measure=group_risk_measure,
        )
        self.offset = (
            self.predictor.dataset_spliter.offset  # offset for feature calculation
            + self.predictor.feature.label_lookahead  # offset for label
            + 1
        )

    def _set_asset_model(
        self,
        predictor: Predictor,
        min_w: float,
        max_w: float,
        objective_func: str = "max_return",
        risk_measure: str = "variance",
    ):
        self.predictor = predictor
        self.asset_model = MeanRisk(
            objective_function=getattr(
                ObjectiveFunction, OBJECTIVE_FUNCTION[objective_func]
            ),
            risk_measure=getattr(RiskMeasure, RISKMEASURE[risk_measure]),
            min_weights=min_w,
            max_weights=max_w,
        )

    def _set_group_model(self, risk_budget: dict, risk_measure: str):
        self.group_model = RiskBudgeting(
            risk_measure=getattr(RiskMeasure, RISKMEASURE[risk_measure]),
            risk_budget=risk_budget,
            portfolio_params=dict(name="Risk Budgeting - CVaR"),
        )

    def optimize(self, preds: pd.DataFrame, rebal_dt: list, risk_budget: dict = None):
        if risk_budget is not None:
            self.group_model.risk_budget = risk_budget
        # 1. Optimize mean-variance model for each individual asset
        asset_weights = self._optimize_individual_asset(preds=preds, rebal_dt=rebal_dt)
        # 2. Optimize risk-budgeting model for each asset group
        group_weights = self._optimize_group_asset(
            preds=preds, asset_weights=asset_weights
        )
        return asset_weights, group_weights

    def predict(self, dfs: pd.DataFrame, rebal_dt: list):
        asset_groups = dfs.columns.get_level_values(0).unique()
        preds = []
        for asset_group in tqdm(asset_groups, desc="Predicting"):
            data = dfs.xs(asset_group, axis=1)
            # 1. get the required data amount for prediction
            start_idx = data.index.get_loc(rebal_dt[0])
            end_idx = data.index.get_loc(rebal_dt[-1])
            x = data.iloc[start_idx - self.offset : end_idx + 1]

            # 2. predict the asset close price at lookahead
            pred = self.predictor.predict(data={asset_group: x})

            # 3. shift index to label index
            if self.predictor.preprocessor.remove_weekend is True:
                weekend_offset = self.predictor.feature.label_lookahead // 5 * 2
            else:
                weekend_offset = 0
            pred.index = pred.index + pd.Timedelta(
                days=self.predictor.feature.label_lookahead + weekend_offset
            )

            # 4. get prediction result
            pred = pred[rebal_dt[0] :]  # included today

            preds.append(pred)
        preds = pd.concat(preds, axis=1)
        return preds

    def _optimize_individual_asset(self, preds: pd.DataFrame, rebal_dt: list):
        """
        Optimize portfolio of all asset in the same asset group.

        Args:
            preds (pd.DataFrame): Dataframe of predicted result. Columns is (assest group, assest name)
            rebal_dt (list): List of rebalance date.

        Returns:
            weights (pd.DataFrame): Dataframe of optimized weight for each individual asset. Index is (rebal_dt, pred_start, pred_end) and columns is (assest group, assest name)
        """
        # offset for the required data amount to use in prediction

        asset_groups = preds.columns.get_level_values(0).unique()
        weights = {}

        for asset_group in tqdm(asset_groups, desc="Optimizing"):
            pred_single_group = preds.xs(asset_group, axis=1)
            asset_cols = list(pred_single_group.columns.get_level_values(0).unique())
            columns = ["rebal_dt", "pred_start", "pred_end"] + asset_cols
            rows = []
            for st_idx, end_idx in zip(rebal_dt[:-1], rebal_dt[1:]):
                pred = pred_single_group[st_idx:end_idx].pct_change().dropna()

                # 1. fit predicted price to get the optimized weights
                self.asset_model.fit(pred[asset_cols])
                # 2. collect weight
                weight = self.asset_model.weights_.copy()
                row = (st_idx, pred.index[0], pred.index[-1]) + tuple(weight)
                rows.append(row)
            # collect weighted return from all rebalancing date
            weights[asset_group] = pd.DataFrame(rows, columns=columns)
            weights[asset_group] = weights[asset_group].set_index(
                ["rebal_dt", "pred_start", "pred_end"]
            )
        weights = pd.concat(weights, axis=1)
        # 7. rename to re-align the assets' group name with expected group from mofu-chan
        weights = weights.rename(
            columns={"bond": "fix_income", "real_estate": "fix_income"}
        )
        return weights

    def _optimize_group_asset(self, preds: pd.DataFrame, asset_weights: pd.DataFrame):
        """Optimize via risk budgeting for each asset group

        Args:
            preds (pd.DataFrame): Dataframe of predicted result. Columns is (assest group, assest name)
            asset_weights (pd.DataFrame): Dataframe of optimized weight for each individual asset. Index is (rebal_dt, pred_start, pred_end) and columns is (assest group, assest name)

        Returns:
            group_weights (pd.DataFrame): Dataframe of optimized weight for each group asset. Index is (rebal_dt, pred_start, pred_end) and columns is assest group.
        """
        preds = preds.rename(
            columns={"bond": "fix_income", "real_estate": "fix_income"}
        )
        preds_return = preds.pct_change(1).dropna()
        asset_group_col = list(preds_return.columns.get_level_values(0).unique())
        columns = ["rebal_dt", "pred_start", "pred_end"] + asset_group_col
        rows = []
        for idx, asset_weight in asset_weights.iterrows():
            weighted_return = (
                (preds_return * asset_weight)
                .groupby(axis=1, level=0)
                .sum()[idx[1] : idx[2]]
            )
            # 1. Optimize weight for each asset group using estimated return of each group
            self.group_model.fit(weighted_return[asset_group_col])
            group_weight = self.group_model.weights_.copy()
            row = idx + tuple(group_weight)
            rows.append(row)
        group_weights = pd.DataFrame(rows, columns=columns)
        group_weights = group_weights.set_index(["rebal_dt", "pred_start", "pred_end"])
        return group_weights

    def pred_optimize(
        self, dfs: pd.DataFrame, rebal_dt: list, risk_budget: dict = None
    ):
        preds = self.predict(dfs=dfs, rebal_dt=rebal_dt)

        # for check with future model
        # preds = dfs.xs("Close", axis = 1, level = 2)
        # preds = preds.reindex(preds.index).dropna()
        asset_weights, group_weights = self.optimize(
            preds=preds, rebal_dt=rebal_dt, risk_budget=risk_budget
        )
        return asset_weights, group_weights
