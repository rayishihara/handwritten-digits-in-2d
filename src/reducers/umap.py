import umap
from .base import BaseReducer

class UMAP(BaseReducer):
    def __init__(self, n_components, n_neighbors=15, min_dist=0.1, random_state=None):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.model_ = None
        self.embedding_ = None

    def fit(self, X):
        """
        X is the data matrix in tabular form, that is,
        every observation is arranged as a row vector.

        Shape: (n_samples, n_features)
        """

        self.model_ = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
        ).fit(X)
        self.embedding_ = self.model_.embedding_

        return self

    def transform(self, X):
        return self.model_.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_
