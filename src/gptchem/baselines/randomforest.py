import numpy as np
from optuna import create_study
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..models.base import BaseLineModel


class RFClassificationBaseline(BaseLineModel):
    def __init__(self, seed, num_trials=100, timeout=None, njobs: int = -1) -> None:
        self.seed = seed
        self.num_trials = num_trials
        self.model = RandomForestClassifier()
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
                "n_estimators": trial.suggest_int("n_estimators", 4, 5_000),
                "max_depth": trial.suggest_int("max_depth", 4, 128),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 128),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 128),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "random_state": random_state,
                "n_jobs": n_jobs,
            }

            model = RandomForestClassifier(**params)

            kf = KFold(n_splits=n_splits)
            X_values = X.values
            y_values = y

            scores = []
            for train_index, test_index in kf.split(X_values):
                X_A, X_B = X_values[train_index, :], X_values[test_index, :]
                y_A, y_B = y_values[train_index], y_values[test_index]

                model.fit(
                    X_A,
                    y_A,
                    # eval_metric="mlogloss",
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

        self.model = RandomForestClassifier(**study.best_params, n_jobs=self.njobs)

    def fit(self, X_train, y_train):
        y_train = self.label_encoder.fit_transform(y_train)
        self.model.fit(X_train.values, y_train)
        return self.model

    def predict(self, X):
        return self.label_encoder.inverse_transform(self.model.predict(X.values))
