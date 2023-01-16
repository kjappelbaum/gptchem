import gpflow
import numpy as np
import tensorflow as tf
from gpflow.mean_functions import Constant
from gpflow.utilities import positive, print_summary
from gpflow.utilities.ops import broadcasting_elementwise
from pycm import ConfusionMatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .base import BaseLineModel


class Tanimoto(gpflow.kernels.Kernel):
    """Tanimoto kernel.

    Taken from https://github.com/Ryan-Rhys/The-Photoswitch-Dataset/blob/master/property_prediction/kernels.py.
    """

    def __init__(self, **kwargs):
        """
        :param kwargs: accepts `name` and `active_dims`, which is a list or
            slice of indices which controls which columns of X are used (by
            default, all columns are used).
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError("Unknown keyword argument:", kwarg)
        super().__init__(**kwargs)
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))
        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        cross_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -cross_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * cross_product / denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


KERNELS = {"tanimoto": Tanimoto, "rbf": gpflow.kernels.RBF, "linear": gpflow.kernels.Linear}


class GPRBaseline(BaseLineModel):
    """GPR w/ Tanimoto kernel baseline."""

    def __init__(self, kernel="tanimoto") -> None:
        self.model = None
        self.y_scaler = StandardScaler()
        self.kernel = kernel

    def tune(self, X_train: np.ndarray, y_train: np.ndarray):  # N x D features  # N x 1 target
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):  # N x D features  # N x 1 target
        y_train = y_train.reshape(-1, 1)

        def objective_closure():
            return -m.log_marginal_likelihood()

        y_train = self.y_scaler.fit_transform(y_train)

        m = gpflow.models.GPR(
            data=(X_train, y_train),
            mean_function=Constant(np.mean(y_train)),
            kernel=KERNELS[self.kernel](),
            noise_variance=1,
        )

        # Optimise the kernel variance and noise level by the marginal likelihood
        opt = gpflow.optimizers.Scipy()
        opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=10000))
        print_summary(m)
        self.model = m

    def predict(self, X_test: np.ndarray):  # N x D features
        y_pred, y_var = self.model.predict_f(X_test)
        y_pred = self.y_scaler.inverse_transform(y_pred)
        return y_pred
