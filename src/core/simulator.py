import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm


class Simulator:
    # simulate future return with init_money and monthly_money
    # compute metrics from backtesting
    def __init__(
        self,
        st_amount: int,
        dca_amount: int,
        invest_period: int,
        risk_free_rate: float = 0,
        rolling_win: int = 365 * 3,
        base_period: int = 252,
    ):
        self.st_amount = st_amount
        self.dca_amount = dca_amount  # monthly invest
        self.invest_period = invest_period  # unit month
        self.risk_free_rate = risk_free_rate
        self.rolling_win = rolling_win
        self.base_period = base_period

    def backtesting(
        self,
        dfs: pd.DataFrame,
        asset_weights: pd.DataFrame,
        group_weights: pd.DataFrame,
    ):
        # portfolio return
        port_return = self._get_return(
            dfs=dfs, asset_weights=asset_weights, group_weights=group_weights
        )

        # index level
        index_level = port_return.cumsum() + 1
        index_level = index_level / index_level[0] * 100

        # get rebalancing date
        rebal_dt = list(asset_weights.index.get_level_values(0)) + [
            asset_weights.index.get_level_values(2)[-1]
        ]

        # groundtruth related information
        gt_return = self._get_groundtruth_return(dfs=dfs)
        daily_rolling_std = gt_return.rolling(self.rolling_win).std().dropna()
        groundtruth_risk = daily_rolling_std.mean(axis=1).loc[
            rebal_dt[0] : rebal_dt[-1]
        ]
        groundtruth_risk = groundtruth_risk.reindex(port_return.index).ffill()

        # investment simulation
        invest = (
            self._get_invest_amount(rebal_dt=rebal_dt)
            .reindex(port_return.index)
            .ffill()
        )
        result = self._sim_invest(invest=invest, port_return=port_return)

        invest[rebal_dt[0] - timedelta(days=1)] = 0  # day before invest
        invest = invest.sort_index()
        return invest, groundtruth_risk, result, index_level

    def _sim_portfolio(
        self,
        monthly_invest: pd.Series,
        avg_monthly_return: float,
        monthly_volatility: float,
    ):
        port_return = np.random.normal(
            avg_monthly_return, monthly_volatility, len(monthly_invest)
        )
        port_return = pd.Series(port_return, index=monthly_invest.index)
        result = self._sim_invest(invest=monthly_invest, port_return=port_return)
        return result

    def sim_monte_carlo(
        self, avg_monthly_return: float, monthly_volatility: float, num_sim: int = 100
    ):
        monthly_invest = self._get_invest_amount().groupby(pd.Grouper(freq="1m")).last()
        portfolio_values = []
        for _ in tqdm(range(num_sim), desc="Sim Monte Carlo"):
            portfolio_values.append(
                self._sim_portfolio(
                    monthly_invest=monthly_invest,
                    avg_monthly_return=avg_monthly_return,
                    monthly_volatility=monthly_volatility,
                )
            )
        portfolio_values = np.array(portfolio_values)
        return pd.DataFrame(
            {
                "values": portfolio_values.mean(axis=0),
                "risk": portfolio_values.std(axis=0),
                "invest": monthly_invest,
            },
            index=monthly_invest.index,
        )

    def _sim_invest(self, invest: pd.DataFrame, port_return: pd.DataFrame):
        # 1. get investment amount
        invest = invest.copy()
        # 2. create placeholder for result
        result = pd.Series(0, index=invest.index)
        # 3. reindex portfolio return with user investment amount
        port_return = port_return.reindex(invest.index).ffill().fillna(0)

        # 4. compute expected investment growth during backtesting period
        while (invest.dropna() > 0).any():
            # investment amount at that month
            idx_amount = invest[invest > 0].index[0]  # date that invest
            amount_value = invest[invest > 0].iloc[0]  # amount that invest
            # time-series of multiplier
            mul_series = pd.Series(0, index=invest.index)
            mul_series[invest > 0] = amount_value
            # compute investment growth at that month
            compound_return = (port_return + 1).cumprod()
            compound_return[idx_amount] = 1  # dca location not start compunding yet
            result += (compound_return).mul(mul_series)

            # remove the amount that already compute
            invest -= amount_value
        return result

    def _get_return(
        self,
        dfs: pd.DataFrame,
        asset_weights: pd.DataFrame,
        group_weights: pd.DataFrame,
    ):
        dfs = dfs.rename(columns={"bond": "fix_income", "real_estate": "fix_income"})
        returns = []
        for (idx, asset_weight), (_, group_weight) in zip(
            asset_weights.iterrows(), group_weights.iterrows()
        ):
            asset_return = (
                dfs.loc[idx[0] : idx[2]]
                .xs("Close", axis=1, level=2)
                .pct_change(1)
                .dropna()
            )
            group_return = (
                (asset_return * asset_weight)
                .groupby(axis=1, level=0)
                .sum()[idx[1] : idx[2]]
            )
            returns.append((group_return * group_weight).sum(axis=1))
        return_ = pd.concat(returns)
        return return_

    def _get_groundtruth_return(self, dfs: pd.DataFrame):
        dfs = dfs.rename(columns={"bond": "fix_income", "real_estate": "fix_income"})
        gt_return = (
            dfs.pct_change()
            .dropna()
            .xs("Close", axis=1, level=2)
            .groupby(axis=1, level=0)
            .mean()
        )
        gt_return = gt_return.sort_index()
        return gt_return

    def _get_invest_amount(self, rebal_dt: list = None):
        if rebal_dt is None:
            rebal_dt = pd.period_range(
                datetime.now().date() - timedelta(days=30),
                freq="M",
                periods=self.invest_period + 1,
            ).to_timestamp()

        invest = [self.st_amount + i * self.dca_amount for i in range(len(rebal_dt))]
        invest = pd.Series(invest, index=rebal_dt)
        invest = invest.resample("d").first().ffill()

        return invest.iloc[:-1]  # exclue last rebalancing

    def get_metrics(self, index_level: pd.DataFrame):
        index_level = index_level.resample("d").first().ffill()
        holding_period = len(index_level.resample("d").first())
        annualization = self.base_period / holding_period

        return_ = index_level.iloc[-1] / index_level.iloc[0]
        return_ = return_ ** (annualization) - 1
        vol = index_level.pct_change(1).std() * np.sqrt(self.base_period)
        sharpe = (return_ - self.risk_free_rate) / vol

        cummulative_max = index_level.pct_change(1).cummax()
        drawdowns = (index_level.pct_change(1) - cummulative_max) / cummulative_max
        max_drawdown = drawdowns.min()

        # monthly metrics
        monthly_return = index_level.groupby(pd.Grouper(freq="1m")).apply(
            lambda x: x.iloc[-1] / x.iloc[0]
        )
        monthly_vol = monthly_return.std()
        monthly_return_mean = (monthly_return - 1).mean()

        metrics = {
            "annualize sharpe ratio": sharpe,
            "annualize return": return_,
            "annualize volatility": vol,
            "max_drawdown": max_drawdown,
            "avg_monthly_return": monthly_return_mean,
            "monthly_volatility": monthly_vol,
        }
        metrics = pd.Series(metrics)

        return metrics
