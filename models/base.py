from abc import abstractmethod
from typing import List


class Model:
    @abstractmethod
    def train_and_predict(self, train, valid, categorical_features: List[str], target: str, params: dict):
        raise NotImplementedError
