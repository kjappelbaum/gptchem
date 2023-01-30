import numpy as np
import pandas as pd
from optuna import create_study
from optuna.integration import XGBoostPruningCallback
from optuna.samplers import TPESampler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .base import BaseLineModel


class XGBClassificationBaseline(BaseLineModel):
    def __init__(self, seed, num_trials=100, timeout=None, njobs: int = -1) -> None:
        self.seed = seed
        self.num_trials = num_trials
        self.model = XGBClassifier()
        self.timeout = timeout
        self.njobs = njobs
        self.label_encoder = LabelEncoder()

    def tune(self, X_train, y_train):
        y_train = self.label_encoder.fit_transform(y_train)

        def objective(
            trial,
            X,
            y,
            random_state=22,
            n_splits=5,
            n_jobs=self.njobs,
        ):
            # XGBoost parameters
            params = {
                "verbosity": 1,  # 0 (silent) - 3 (debug)
                "n_estimators": trial.suggest_int("n_estimators", 4, 10_000),
                "max_depth": trial.suggest_int("max_depth", 4, 50),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.05),
                "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 1),
                "subsample": trial.suggest_loguniform("subsample", 0.00001, 1),
                "alpha": trial.suggest_loguniform("alpha", 1e-8, 10.0),
                "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
                "seed": random_state,
                "n_jobs": n_jobs,
            }

            model = XGBClassifier(**params)
            # pruning_callback = XGBoostPruningCallback(trial, "validation_0-mlogloss")
            kf = KFold(n_splits=n_splits)
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
            y_values = y

            scores = []
            for train_index, test_index in kf.split(X_values):
                X_A, X_B = X_values[train_index, :], X_values[test_index, :]
                y_A, y_B = y_values[train_index], y_values[test_index]

                model.fit(
                    X_A,
                    y_A,
                    eval_set=[(X_B, y_B)],
                    # eval_metric="mlogloss",
                    verbose=0,
                    # callbacks=[pruning_callback],
                )
                y_pred = model.predict(X_B)
                scores.append(f1_score(y_pred, y_B, average="macro"))
            return np.mean(scores)

        sampler = TPESampler(seed=self.seed)
        study = create_study(direction="maximize", sampler=sampler)
        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                y_train,
                random_state=self.seed,
                n_splits=5,
                n_jobs=1,
            ),
            n_trials=self.num_trials,
            n_jobs=1,
            timeout=self.timeout,
        )

        self.model = XGBClassifier(**study.best_params, n_jobs=self.njobs)

    def fit(self, X_train, y_train):
        y_train = self.label_encoder.fit_transform(y_train)
        if isinstance(X_train, pd.DataFrame):
            X_values = X_train.values
        else:
            X_values = X_train
        self.model.fit(X_values, y_train)
        return self.model

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        return self.label_encoder.inverse_transform(self.model.predict(X_values))


class XGBRegressionBaseline(BaseLineModel):
    def __init__(self, seed, num_trials=100, njobs: int = -1) -> None:
        self.seed = seed
        self.num_trials = num_trials
        self.model = XGBRegressor()
        self.njobs = njobs

    def tune(self, X_train, y_train):
        def objective(
            trial,
            X,
            y,
            random_state=22,
            n_splits=5,
            n_jobs=self.njobs,
            early_stopping_rounds=50,
        ):
            # XGBoost parameters
            params = {
                "verbosity": 0,  # 0 (silent) - 3 (debug)
                "objective": "reg:squarederror",
                "n_estimators": 10000,
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
                "colsample_bytree": trial.suggest_uniform(
                    "colsample_bytree", 0.2, 0.6  # note: log uniform was used before
                ),
                "subsample": trial.suggest_uniform(
                    "subsample", 0.4, 0.8
                ),  # note: log uniform was used before
                "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
                "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
                "gamma": trial.suggest_loguniform(
                    "gamma", 1e-8, 10.0
                ),  # note: this was wrong before (lambda was used as name)
                "min_child_weight": trial.suggest_loguniform("min_child_weight", 10, 1000),
                "seed": random_state,
                "n_jobs": n_jobs,
            }

            model = XGBRegressor(**params)

            kf = KFold(n_splits=n_splits)
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
            y_values = y
            scores = []
            for train_index, test_index in kf.split(X_values):
                X_A, X_B = X_values[train_index, :], X_values[test_index, :]
                y_A, y_B = y_values[train_index], y_values[test_index]
                model.fit(
                    X_A,
                    y_A,
                    eval_set=[(X_B, y_B)],
                    eval_metric="rmse",
                    verbose=0,
                    early_stopping_rounds=early_stopping_rounds,
                )
                y_pred = model.predict(X_B)
                scores.append(mean_squared_error(y_pred, y_B))
            return np.mean(scores)

        sampler = TPESampler(seed=self.seed)
        study = create_study(direction="minimize", sampler=sampler)
        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                y_train,
                random_state=self.seed,
                n_splits=5,
                n_jobs=8,
                early_stopping_rounds=100,
            ),
            n_trials=self.num_trials,
            n_jobs=1,
        )

        self.model = XGBRegressor(**study.best_params, n_jobs=self.njobs)

    def fit(self, X_train, y_train):
        if isinstance(X_train, pd.DataFrame):
            X_values = X_train.values
        else:
            X_values = X_train
        self.model.fit(X_values, y_train)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        return self.model.predict(X_values)
