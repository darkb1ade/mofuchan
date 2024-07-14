import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from features import return_, vol, daily_return, moving_average, intra_day, prices_range
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
    def __init__(self, lookahead: int = 1, **feature_config):
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
        self.lookahead = lookahead

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
            .shift(-1 * self.lookahead)
            .rename(columns={"Close": f"fwd_return({self.lookahead})"})
        )
        return x, y


class DatasetSpliter:
    def __init__(
        self, train_start: str, test_start: str, offset: int, use_scaler: bool = True
    ):
        self.train_start = train_start
        self.test_start = test_start
        self.offset = offset
        self.use_scaler = use_scaler
        self.feature_scaler = None
        self.label_scaler = None

    def transform(self, df: pd.DataFrame, feature_columns: list, label_columns: list):
        test_start_idx = df.index.get_loc(self.test_start).start
        if self.train_start is None:
            train = df.iloc[: test_start_idx - self.offset]
        else:
            train_start_idx = df.index.get_loc(self.train_start).start
            train = df.iloc[train_start_idx : test_start_idx - self.offset]
        test = df.iloc[test_start_idx:]

        if self.use_scaler is True:
            self.feature_scaler = StandardScaler()
            train_x = self.feature_scaler.fit_transform(train[feature_columns])
            test_x = self.feature_scaler.transform(test[feature_columns])
            train_x = pd.DataFrame(
                train_x, index=train.index, columns=[feature_columns]
            )
            test_x = pd.DataFrame(test_x, index=test.index, columns=[feature_columns])

            self.label_scaler = StandardScaler()
            train_y = self.label_scaler.fit_transform(train[label_columns])
            test_y = self.label_scaler.transform(test[label_columns])
            train_y = pd.DataFrame(train_y, index=train.index, columns=[label_columns])
            test_y = pd.DataFrame(test_y, index=test.index, columns=[label_columns])
        else:
            train_x, train_y = train[feature_columns], train[label_columns]
            test_x, test_y = test[feature_columns], test[label_columns]
        return train_x, train_y, test_x, test_y


class Tuner:
    def __init__(self, offset: int, n_trial: int = 100, valid_ratio: float = 0.2):
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
        self.bond = {}
        self.commodity = {}
        self.currency = {}
        self.real_estate = {}
        self.equity = {}

    def set_model(self, model, asset_group: str, asset: str, overwrite: bool = True):
        assert hasattr(self, asset_group), "Wrong asset group name"

        if asset in getattr(self, asset_group):
            if overwrite is True:
                getattr(self, asset_group)[asset] = model
            else:
                print(f"Model for {asset} already exists")
        else:
            getattr(self, asset_group)[asset] = model

    def predict(self):
        ...


class Predictor:
    def __init__(
        self,
        pathOut: str,
        preproc_config: dict,
        feature_config: dict,
        dataset_spliter_config: dict,
        tuner_config: dict,
        model_path: str = None,
    ):
        self.preprocessor = Preprocessor(**preproc_config)
        self.feature = Feature(**feature_config)
        self.dataset_spliter = DatasetSpliter(**dataset_spliter_config)
        self.tuner = Tuner(**tuner_config)
        self.feature_col = []
        self.label_col = []
        self.pathOut = pathOut
        self.model_path = model_path
        self._init_model()

    def _init_model(self, **kwargs):
        if self.model_path is None:
            self.model = GroupModel()

    def train(self, df: pd.DataFrame, asset_group: str):
        df_clean = self.preprocessor.transform(df)
        assets = df_clean.columns.get_level_values(0).unique()

        asset = assets[0]
        # TODO: for each asset
        for asset in assets[:2]:
            x, y = self.feature.transform(df_clean.xs(asset, axis=1))
            self.feature_col = x.columns
            self.label_col = y.columns
            data = pd.concat([x, y], axis=1)
            data = data.dropna().replace([np.inf, -np.inf], 0)

            train_x, train_y, test_x, test_y = self.dataset_spliter.transform(
                df=data, feature_columns=self.feature_col, label_columns=self.label_col
            )
            best_params = self.tuner.tune(train_x=train_x, train_y=train_y)
            model = xgb.XGBRegressor(**best_params)
            model.fit(train_x, train_y)
            # save model
            self.model.set_model(model=model, asset_group=asset_group, asset=asset)
            self.visualize_result(
                model=model,
                asset_title=f"{asset_group}-{asset}",
                test_x=test_x,
                test_y=test_y,
            )
        print("done")

    def visualize_result(
        self, model, asset_title: str, test_x: pd.DataFrame, test_y: pd.DataFrame
    ):
        # 1. Create prediction result
        pred = model.predict(test_x)
        y_true = self.dataset_spliter.label_scaler.inverse_transform(test_y).reshape(-1)
        y_true = pd.DataFrame(y_true, index=test_y.index, columns=["y_true"])
        y_pred = self.dataset_spliter.label_scaler.inverse_transform(
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
        score_df.to_csv(f"{self.pathOut}/{asset_title}_score.csv")

    def _feature_important(self, model, asset_title: str):
        _, axis = plt.subplots(1, 1, figsize=(15, 15))
        plot_importance(model, ax=axis)
        plt.tight_layout()
        plt.savefig(f"{self.pathOut}/{asset_title}_feature_important.jpg")

    def _time_series_result(self, result: pd.DataFrame, asset_title: str):
        _, axis = plt.subplots(2, 1, figsize=(15, 7))
        result.plot(ax=axis[0])
        axis[0].set_title("Predicted Prices")

        return_result = result.pct_change()
        return_result.plot(ax=axis[1])
        axis[1].set_title("Forward Return (1D)")
        axis[1].set_ylabel("Forward Return")

        plt.tight_layout()
        plt.savefig(f"{self.pathOut}/{asset_title}_predicted_result.jpg")

    def _conf_matrix(self, con_matrix: np.ndarray, asset_title: str):
        fig = plt.figure(figsize=(10, 10))
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
        plt.savefig(f"{self.pathOut}/{asset_title}_conf_matrix.jpg")

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
        with open(f"{self.pathOut}/groupmodel.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def predict(self, df: pd.DataFrame):
        return self.preprocessor.transform(df)


def _train(df: pd.DataFrame, asset_group: str):
    preproc_config = {"freq": "d", "remove_weekend": True}

    feature_config = {
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
    dataset_spliter_config = {
        "train_start": None,
        "test_start": "2024",
        "offset": _get_dataset_offset(feature_config),
        "use_scaler": True,
    }
    tuner_config = {
        "offset": _get_dataset_offset(feature_config),
        "n_trial": 100,
        "valid_ratio": 0.2,
    }
    predictor = Predictor(
        pathOut="/workdir/notebook/out",
        preproc_config=preproc_config,
        feature_config=feature_config,
        dataset_spliter_config=dataset_spliter_config,
        tuner_config=tuner_config,
    )

    predictor.train(df, asset_group)
    predictor.save_model()


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


def train():
    from glob import glob

    files = glob("/workdir/notebook/data/*.csv")
    dfs = []
    # for file in files:
    #     df = pd.read_csv(file, index_col=0, header=[0, 1])
    #     df.index = pd.to_datetime(df.index)
    df = pd.read_csv(files[0], index_col=0, header=[0, 1])
    df.index = pd.to_datetime(df.index)
    _train(df, asset_group=files[0].split("/")[-1].split(".")[0])


if __name__ == "__main__":
    train()
