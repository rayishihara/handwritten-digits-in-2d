from abc import ABC, abstractmethod
from typing import Self

import numpy as np

class BaseReducer(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> Self: ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray: ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)