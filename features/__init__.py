import hashlib
import os
import time
from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd


class Feature(ABC):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def create_features(self, train_path: str, test_path: str, random_states: List[Tuple[int, pd.Index]]) -> Tuple[List[str], str]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def categorical_features():
        raise NotImplementedError


class FeatherFeature(Feature):
    def create_features(self, train_path: str, test_path: str, random_states: List[Tuple[int, pd.Index]]) -> (List[str], str):
        train_feature_paths = [self.get_feature_file('train', train_path, test_path, random_state=random_state)
                               for (random_state, _) in random_states]
        train_feature_paths_with_index = [(train_feature_paths[i], index) for i, (_, index) in enumerate(random_states)]
        test_feature_path = self.get_feature_file('test', train_path, test_path, 0)

        is_train_cached = all([os.path.exists(train_feature_path) for train_feature_path in train_feature_paths])
        if is_train_cached and os.path.exists(test_feature_path):
            print("There are cache files for feature [{}] (train_path=[{}], test_path=[{}])"
                  .format(self.name, train_path, test_path))
            return train_feature_paths, test_feature_path

        print("Start computing feature [{}] (train_path=[{}], test_path=[{}])".format(self.name, train_path, test_path))
        start_time = time.time()
        self.create_features_impl(train_path=train_path,
                                  test_path=test_path,
                                  train_feature_paths=train_feature_paths_with_index,
                                  test_feature_path=test_feature_path)

        print("Finished computing feature [{}] (train_path=[{}], test_path=[{}]): {:.3} [s]"
              .format(self.name, train_path, test_path, time.time() - start_time))
        return train_feature_paths, test_feature_path

    def get_feature_file(self, dataset_type: str, train_path: str, test_path: str, random_state: int) -> str:
        feature_cache_suffix = self.get_feature_suffix(train_path, test_path, random_state)
        filename = self.name + '_' + dataset_type + '_' + feature_cache_suffix + '.feather'
        return os.path.join(self.data_dir, filename)

    @staticmethod
    def get_feature_suffix(train_path: str, test_path, random_state: int) -> str:
        return hashlib.md5(str([train_path, test_path]).encode('utf-8')).hexdigest()[:10] + "_{}".format(random_state)

    @abstractmethod
    def create_features_impl(self, train_path: str, test_path: str, train_feature_paths: List[Tuple[str, pd.Index]],
                             test_feature_path: str):
        raise NotImplementedError


class FeatherFeatureDF(FeatherFeature):
    def create_features_impl(self, train_path: str, test_path: str, train_feature_paths: List[Tuple[str, pd.Index]],
                             test_feature_path: str):
        df_train = pd.read_feather(train_path)
        df_test = pd.read_feather(test_path)
        train_feature, test_feature = self.create_features_from_dataframe(df_train, df_test)
        for train_feature_path, index in train_feature_paths:
            train_feature.loc[index].reset_index(drop=True).to_feather(train_feature_path)
        test_feature.to_feather(test_feature_path)

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        raise NotImplementedError
