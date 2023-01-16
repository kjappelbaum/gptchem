from numpy.typing import ArrayLike

class GPTClassifier: 
    def __init__(self, formatter, tuner, querier, extractor):
        self.formatter = formatter
        self.tuner = tuner
        self.querier = querier
        self.extractor = extractor

    def fit(X, y):
        pass

    def predict(X):
        pass    