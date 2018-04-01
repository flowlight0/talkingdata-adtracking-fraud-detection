import pandas as pd

from features import FeatherFeatureDF


class Ip(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return ['ip']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame):
        return df_train[['ip']], df_valid[['ip']], df_test[['ip']]


class App(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return ['app']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame):
        return df_train[['app']], df_valid[['app']], df_test[['app']]


class Os(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return ['os']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame):
        return df_train[['os']], df_valid[['os']], df_test[['os']]


class Channel(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return ['channel']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame):
        return df_train[['channel']], df_valid[['channel']], df_test[['channel']]
