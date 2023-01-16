from abc import ABC, abstractmethod


class BaseLineModel(ABC):
    @abstractmethod
    def tune(self, X_train, y_train):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, X_train, y_train):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise
