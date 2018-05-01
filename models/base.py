from abc import abstractmethod
from typing import List


class Model:
    @abstractmethod
    def train_and_predict(self, train, valid, weight, categorical_features: List[str], target: str, params: dict):
        raise NotImplementedError

    @abstractmethod
    def train_without_validation(self, train, weight, categorical_features: List[str], target: str, params: dict, best_iteration: int):
        raise NotImplementedError
