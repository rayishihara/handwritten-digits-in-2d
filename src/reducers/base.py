from abc import ABC, abstractmethod

class BaseReducer(ABC):
    @abstractmethod
    def fit(self, X): ...

    @abstractmethod
    def transform(self, X): ...

    def fit_transform(self, X):
        return self.fit(X).transform(X)