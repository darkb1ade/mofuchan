"""Empirical Prior Model estimator."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

import numpy as np
import numpy.typing as npt
import sklearn.utils.metadata_routing as skm

from skfolio.moments import BaseCovariance, BaseMu, EmpiricalCovariance, EmpiricalMu
from skfolio.prior._base import BasePrior, PriorModel
from skfolio.utils.tools import check_estimator
from src.core.predictor import Predictor
import pandas as pd


class XGBPrediction(BasePrior):
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            mu_estimator=self.predictor,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def fit(self, X: npt.ArrayLike, y=None, **fit_params) -> "XGBPrediction":
        result = self.predictor.predict(data={fit_params["asset_group"]: X})
        result.index = result.index + pd.Timedelta(days=20 + 4 * 2)  # 4*2 for weekend

        # filter prediction
        dt = X.index[-1]  # today date
        idx = result.index.get_loc(dt)
        result = result.iloc[idx:]

        daily_return = result.pct_change().dropna()  # drop first get tomorrow
        mu = daily_return.mean().values
        covariance = daily_return.cov().values

        self.prior_model_ = PriorModel(
            mu=mu,
            covariance=covariance,
            returns=daily_return,
        )
        self.idx = result.index  # include today index
        return self
