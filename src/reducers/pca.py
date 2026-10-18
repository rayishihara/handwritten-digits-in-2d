import numpy as np
from .base import BaseReducer

class PCA(BaseReducer):
    def __init__(self, rank):
        self.rank = rank
        self.mean_ = None
        self.principal_components_ = None
        self.explained_variance_ = None

    def fit(self, X):
        """
        X is the data matrix in tabular form, that is,
        every observation is arranged as a row vector.
        
        Shape: (n_samples, n_features)
        """

        X_mean = np.mean(X, axis=0)
        X_centred = X - X_mean

        U, S, VT = np.linalg.svd(X_centred, full_matrices=False)
        
        self.mean_ = X_mean
        self.principal_components_ = VT[:self.rank].T
        self.explained_variance_ = S[:self.rank]**2 / (X.shape[0] - 1)

        return self

    def transform(self, X):
        return (X - self.mean_) @ self.principal_components_