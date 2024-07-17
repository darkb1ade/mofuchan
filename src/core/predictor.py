import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .features import (
    return_,
    vol,
    daily_return,
    moving_average,
    intra_day,
    prices_range,
)
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from xgboost import plot_importance
import pickle


class Preprocessor:  # TODO: discard non-business day
    def __init__(self, freq: int = "d", remove_weekend: bool = True):
        self.freq = freq
        self.remove_weekend = remove_weekend

    def transform(self, df: pd.DataFrame):
        df = df.copy().sort_index()
        # 1. resample
        df = df.sort_index().resample("d").first()
        # 2. ffill
        df = df.ffill()
        # 3. remove weekend
        df = df[df.index.dayofweek <= 4]
        return df


class Feature:
    def __init__(self, label_lookahead: int = 1, **feature_config):
        self.feature_funcs = {
            "return": return_,
            "volatility": vol,
            "ma": moving_average,
            "daily_return": daily_return,
            "range": prices_range,
            "intra_day": intra_day,
        }
        assert set(feature_config.keys()).issubset(
            self.feature_funcs.keys()
        ), f"Feature config key not to match with function. Found:{(',').join(feature_config.keys())}. Expected:{(',').join(self.feature_funcs.keys())}"
        self.__dict__.update(feature_config)
        self.label_lookahead = label_lookahead

    def transform(self, df: pd.DataFrame):
        features = []
        str_idx = df.index[df.notna().all(axis=1).values][0]  # first index that not nan
        df = df[str_idx:].copy()
        for feature, func in self.feature_funcs.items():
            if hasattr(self, feature):
                feature = func(df=df, **getattr(self, feature))
            features.append(feature)
        x = pd.concat(features, axis=1).replace([np.inf, -np.inf], 0)
        y = (
            df[["Close"]]
            .shift(-1 * self.label_lookahead)
            .rename(columns={"Close": f"fwd_return({self.label_lookahead})"})
        )
        return x, y


class DatasetSpliter:
    def __init__(
        self, train_start: str = None, test_start: str = "2024", offset: int = 1
    ):
        self.train_start = train_start
        self.test_start = test_start
        self.offset = offset
        self.feature_scaler = None
        self.label_scaler = None

    def transform(self, df: pd.DataFrame):
        if self.test_start is None:  # not split
            return df, df

        test_start_idx = df.index.get_loc(self.test_start).start
        if self.train_start is None:
            train = df.iloc[: test_start_idx - self.offset]
        else:
            train_start_idx = df.index.get_loc(self.train_start).start
            train = df.iloc[train_start_idx : test_start_idx - self.offset]
        test = df.iloc[test_start_idx:]

        return train, test


class Tuner:
    def __init__(self, offset: int = 1, n_trial: int = 100, valid_ratio: float = 0.2):
        self.n_trial = n_trial
        self.offset = offset
        self.valid_ratio = valid_ratio

    def _split_data(self, train_x, train_y):
        train_end_idx = int(train_x.shape[0] * (1 - self.valid_ratio))
        return (
            train_x.iloc[: train_end_idx - self.offset],
            train_y.iloc[: train_end_idx - self.offset],
            train_x.iloc[train_end_idx:],
            train_y.iloc[train_end_idx:],
        )

    def objective(self, trial, x_train, y_train, x_valid, y_valid):
        # 1. Create parameter to be tuned from config
        params = {
            "objective": "reg:squarederror",
            # "sampling_method": "gradient_based",
            "alpha": trial.suggest_float(name="alpha", low=0.001, high=10.0, log=True),
            "lambda": trial.suggest_float(
                name="lambda", low=0.001, high=10.0, log=True
            ),
            "colsample_bytree": trial.suggest_categorical(
                name="colsample_bytree",
                choices=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            ),
            "subsample": trial.suggest_categorical(
                name="subsample", choices=[0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            ),
            "learning_rate": trial.suggest_categorical(
                name="learning_rate",
                choices=[0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.1, 0.3, 0.5],
            ),
            "max_depth": trial.suggest_categorical(
                name="max_depth", choices=[5, 7, 9, 11, 13, 15, 17]
            ),
            "min_child_weight": trial.suggest_int(
                name="min_child_weight", low=1, high=300
            ),
        }

        # 2. Create Model
        model = xgb.XGBRegressor(**params)
        # 3. Train Model
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        # 4. Evaluate
        metrics = mean_squared_error(y_valid, model.predict(x_valid))

        return metrics

    def tune(self, train_x, train_y):
        train_x, train_y, valid_x, valid_y = self._split_data(
            train_x=train_x, train_y=train_y
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.objective(
                trial,
                train_x,
                train_y,
                valid_x,
                valid_y,
            ),
            n_trials=self.n_trial,
        )
        return study.best_params


class GroupModel:
    def __init__(self):
        self._init_model()
        self._init_feature_scaler()
        self._init_label_scaler()

    def _init_model(self):
        self.bond = {}
        self.commodity = {}
        self.currency = {}
        self.real_estate = {}
        self.equity = {}

    def _init_label_scaler(self):
        self.label_scaler_bond = {}
        self.label_scaler_commodity = {}
        self.label_scaler_currency = {}
        self.label_scaler_real_estate = {}
        self.label_scaler_equity = {}

    def _init_feature_scaler(self):
        self.feature_scaler_bond = {}
        self.feature_scaler_commodity = {}
        self.feature_scaler_currency = {}
        self.feature_scaler_real_estate = {}
        self.feature_scaler_equity = {}

    def set_model(self, model, asset_group: str, asset: str, overwrite: bool = True):
        assert hasattr(self, asset_group), "Wrong asset group name"

        if asset in getattr(self, asset_group):
            if overwrite is True:
                getattr(self, asset_group)[asset] = model
            else:
                print(f"Model for {asset} already exists")
        else:
            getattr(self, asset_group)[asset] = model

    def set_label_scaler(
        self, label_scaler, asset_group: str, asset: str, overwrite: bool = True
    ):
        attr_name = f"label_scaler_{asset_group}"
        assert hasattr(self, attr_name), "Wrong asset group name"

        if asset in getattr(self, attr_name):
            if overwrite is True:
                getattr(self, attr_name)[asset] = label_scaler
            else:
                print(f"Label Scaler for {asset} already exists")
        else:
            getattr(self, attr_name)[asset] = label_scaler

    def set_feature_scaler(
        self, feature_scaler, asset_group: str, asset: str, overwrite: bool = True
    ):
        attr_name = f"feature_scaler_{asset_group}"
        assert hasattr(self, attr_name), "Wrong asset group name"

        if asset in getattr(self, attr_name):
            if overwrite is True:
                getattr(self, attr_name)[asset] = feature_scaler
            else:
                print(f"Label Scaler for {asset} already exists")
        else:
            getattr(self, attr_name)[asset] = feature_scaler


class Predictor:
    def __init__(
        self,
        path_out: str,
        preproc_config: dict,
        feature_config: dict,
        tuner_config: dict = {},
        dataset_spliter_config: dict = {},
        model_path: str = None,
        use_scaler: bool = True,
    ):
        self.preprocessor = Preprocessor(**preproc_config)
        self.feature = Feature(**feature_config)
        self.dataset_spliter = DatasetSpliter(**dataset_spliter_config)
        self.tuner = Tuner(**tuner_config)
        self.feature_col = []
        self.label_col = []
        self.path_out = path_out
        self.model_path = model_path
        self.use_scaler = use_scaler
        self._init_model()

    def _init_model(self, **kwargs):
        if self.model_path is None:
            self.model = GroupModel()

    def train(self, data: dict[pd.DataFrame]):
        for asset_group, df in data.items():
            self._train_single_group(df=df, asset_group=asset_group)
        self.save_model()

    def _norm_data(self, train: pd.DataFrame, test: pd.DataFrame):
        feature_scaler = StandardScaler()
        train_x = feature_scaler.fit_transform(train[self.feature_col])
        test_x = feature_scaler.transform(test[self.feature_col])
        train_x = pd.DataFrame(train_x, index=train.index, columns=[self.feature_col])
        test_x = pd.DataFrame(test_x, index=test.index, columns=[self.feature_col])

        label_scaler = StandardScaler()
        train_y = label_scaler.fit_transform(train[self.label_col])
        test_y = label_scaler.transform(test[self.label_col])
        train_y = pd.DataFrame(train_y, index=train.index, columns=[self.label_col])
        test_y = pd.DataFrame(test_y, index=test.index, columns=[self.label_col])
        return feature_scaler, label_scaler, train_x, train_y, test_x, test_y

    def _train_single_group(self, df: pd.DataFrame, asset_group: str):
        df_clean = self.preprocessor.transform(df)
        assets = df_clean.columns.get_level_values(0).unique()

        # for each asset
        for asset in assets:
            x, y = self.feature.transform(df_clean.xs(asset, axis=1))
            self.feature_col = x.columns
            self.label_col = y.columns
            data = pd.concat([x, y], axis=1)
            data = data.dropna().replace([np.inf, -np.inf], 0)

            train, test = self.dataset_spliter.transform(df=data)
            if self.use_scaler is True:
                (
                    feature_scaler,
                    label_scaler,
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                ) = self._norm_data(train=train, test=test)
            else:
                feature_scaler, label_scaler = None, None
                train_x, train_y = train[self.feature_col], train[self.label_col]
                test_x, test_y = test[self.feature_col], test[self.label_col]

            best_params = self.tuner.tune(train_x=train_x, train_y=train_y)
            model = xgb.XGBRegressor(**best_params)
            model.fit(train_x, train_y)

            # save model
            self.model.set_model(model=model, asset_group=asset_group, asset=asset)
            self.model.set_label_scaler(
                label_scaler=label_scaler, asset_group=asset_group, asset=asset
            )
            self.model.set_feature_scaler(
                feature_scaler=feature_scaler, asset_group=asset_group, asset=asset
            )
            self.visualize_result(
                model=model,
                asset_title=f"{asset_group}-{asset}",
                test_x=test_x,
                test_y=test_y,
                label_scaler=label_scaler,
            )

    def visualize_result(
        self,
        label_scaler,
        model,
        asset_title: str,
        test_x: pd.DataFrame,
        test_y: pd.DataFrame,
    ):
        # 1. Create prediction result
        pred = model.predict(test_x)
        y_true = label_scaler.inverse_transform(test_y).reshape(-1)
        y_true = pd.DataFrame(y_true, index=test_y.index, columns=["y_true"])
        y_pred = label_scaler.inverse_transform(
            pd.DataFrame(pred, index=test_y.index, columns=["y_pred"])
        ).reshape(-1)
        y_pred = pd.DataFrame(y_pred, index=test_y.index, columns=["y_pred"])
        result = pd.concat([y_true, y_pred], axis=1)

        # 2. Create confusion matrix
        class_result = (result.pct_change().dropna() >= 0).astype("int")
        con_matrix = confusion_matrix(class_result["y_true"], class_result["y_pred"])

        # 3. Plot
        self._feature_important(
            model=model, asset_title=asset_title
        )  # feature important
        self._time_series_result(
            result=result, asset_title=asset_title
        )  # predicted result
        self._conf_matrix(
            con_matrix=con_matrix, asset_title=asset_title
        )  # confusion matrix

        # 4. Generate evaluation metrics & save
        scores = self._conf_score(con_matrix=con_matrix)
        scores.update({"MSE": mean_squared_error(test_y, pred)})
        score_df = pd.DataFrame(scores, index=["score"])
        score_df.to_csv(f"{self.path_out}/{asset_title}_score.csv")

    def _feature_important(self, model, asset_title: str):
        _, axis = plt.subplots(1, 1, figsize=(15, 15))
        plot_importance(model, ax=axis)
        plt.tight_layout()
        plt.savefig(f"{self.path_out}/{asset_title}_feature_important.jpg")

    def _time_series_result(self, result: pd.DataFrame, asset_title: str):
        _, axis = plt.subplots(2, 1, figsize=(15, 7))
        result.plot(ax=axis[0])
        axis[0].set_title("Predicted Prices")

        return_result = result.pct_change()
        return_result.plot(ax=axis[1])
        axis[1].set_title("Forward Return (1D)")
        axis[1].set_ylabel("Forward Return")

        plt.tight_layout()
        plt.savefig(f"{self.path_out}/{asset_title}_predicted_result.jpg")

    def _conf_matrix(self, con_matrix: np.ndarray, asset_title: str):
        _ = plt.figure(figsize=(10, 10))
        sns.heatmap(
            con_matrix,
            annot=True,
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            linewidths=0.2,
        )
        plt.ylabel("groundtruth")
        plt.xlabel("Prediction")
        plt.title("Signal")
        plt.tight_layout()
        plt.savefig(f"{self.path_out}/{asset_title}_conf_matrix.jpg")

    def _conf_score(self, con_matrix: np.ndarray):
        tn, fp, fn, tp = con_matrix.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fp)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }

    def save_model(self):
        with open(f"{self.path_out}/groupmodel.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, data: dict[pd.DataFrame]):
        self.load_model()
        predictions = {}
        for asset_group, df in data.items():
            prediction = self._predict_single_group(df=df, asset_group=asset_group)
            predictions[asset_group] = prediction
        return pd.concat(predictions, axis=1)

    def _predict_single_group(self, df: pd.DataFrame, asset_group: str):
        # 1. clean data
        df_clean = self.preprocessor.transform(df)
        assets = df_clean.columns.get_level_values(0).unique()

        predictions = []
        # for each asset
        for asset in assets:
            # 2. compute feature
            x, _ = self.feature.transform(df_clean.xs(asset, axis=1))
            # 3. get enough data for prediction
            self.feature_col = x.columns
            _, x = self.dataset_spliter.transform(df=x)
            x = (
                x[self.feature_col].dropna().replace([np.inf, -np.inf], 0)
            )  # remove rows with nan
            idx = x.index
            # 4. predict
            x = getattr(self.model, f"feature_scaler_{asset_group}")[asset].transform(x)
            y_pred = getattr(self.model, asset_group)[asset].predict(x)
            # 5. reverse scaler
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
            y_pred = getattr(self.model, f"label_scaler_{asset_group}")[
                asset
            ].inverse_transform(y_pred)
            y_pred = pd.DataFrame(y_pred, index=idx, columns=[asset])
            predictions.append(y_pred)
        return pd.concat(predictions, axis=1)
