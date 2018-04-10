import hashlib
import os
import time
from abc import ABC, abstractmethod

import pandas as pd


class Feature(ABC):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def create_features(self, train_path, valid_path, test_path) -> (str, str, str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def categorical_features():
        raise NotImplementedError


class FeatherFeature(Feature):
    def create_features(self, train_path, valid_path, test_path) -> (str, str, str):
        train_feature_file = self.get_feature_file('train', train_path, valid_path, test_path)
        valid_feature_file = self.get_feature_file('valid', train_path, valid_path, test_path)
        test_feature_file = self.get_feature_file('test', train_path, valid_path, test_path)

        if os.path.exists(train_feature_file) and os.path.exists(valid_feature_file) and os.path.exists(
                test_feature_file):
            print("There are cache files for feature [{}] (train_path=[{}], valid_path=[{}], test_path=[{}])"
                  .format(self.name, train_path, valid_path, test_path))
            return train_feature_file, valid_feature_file, test_feature_file

        print("Start computing feature [{}] (train_path=[{}], valid_path=[{}], test_path=[{}])"
              .format(self.name, train_path, valid_path, test_path))
        start_time = time.time()

        self.create_features_impl(train_input=train_path,
                                  valid_input=valid_path,
                                  test_input=test_path,
                                  train_output=train_feature_file,
                                  valid_output=valid_feature_file,
                                  test_output=test_feature_file)

        print("Finished computing feature [{}] (train_path=[{}], valid_path=[{}], test_path=[{}]): {:.3} [s]"
              .format(self.name, train_path, valid_path, test_path, time.time() - start_time))
        return train_feature_file, valid_feature_file, test_feature_file

    def get_feature_file(self, dataset_type, train_path, valid_path, test_path):
        feature_cache_suffix = self.get_feature_suffix(train_path, valid_path, test_path)
        filename = self.name + '_' + dataset_type + '_' + feature_cache_suffix + '.feather'
        return os.path.join(self.data_dir, filename)

    @staticmethod
    def get_feature_suffix(train_path, valid_path, test_path) -> str:
        return hashlib.md5(str([train_path, valid_path, test_path]).encode('utf-8')).hexdigest()[:10]

    @abstractmethod
    def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
        raise NotImplementedError


class FeatherFeatureDF(FeatherFeature):
    def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
        df_train = pd.read_feather(train_input)
        df_valid = pd.read_feather(valid_input)
        df_test = pd.read_feather(test_input)
        train_feature, valid_feature, test_feature = self.create_features_from_dataframe(df_train, df_valid, df_test)
        train_feature.to_feather(train_output)
        valid_feature.to_feather(valid_output)
        test_feature.to_feather(test_output)

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame):
        raise NotImplementedError
