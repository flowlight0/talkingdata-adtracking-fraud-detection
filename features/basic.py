import gc

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


class ClickHour(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        # We don't use 'day' information because time span of our dataset is too short
        return ['hour']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame):
        df_train['hour'] = pd.to_datetime(df_train.click_time).dt.hour.astype('uint8')
        df_valid['hour'] = pd.to_datetime(df_valid.click_time).dt.hour.astype('uint8')
        df_test['hour'] = pd.to_datetime(df_test.click_time).dt.hour.astype('uint8')
        return df_train[['hour']], df_valid[['hour']], df_test[['hour']]


class BasicCount(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return []

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame):
        print("Creating new count features: 'n_channels', 'ip_app_count', 'ip_app_os_count'...")
        train: pd.DataFrame = pd.concat([df_train, df_valid, df_test])
        train['hour'] = pd.to_datetime(train.click_time).dt.hour.astype('uint8')
        train['day'] = pd.to_datetime(train.click_time).dt.day.astype('uint8')
        print(train.head(), train.shape)

        print('Computing the number of channels associated with a given IP address within each hour...')
        channel_count = train[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[
            ['channel']].count().reset_index().rename(columns={'channel': 'n_channels'})
        print('Merging the channels data with the main data set...')
        train = train.merge(channel_count, on=['ip', 'day', 'hour'], how='left')
        del channel_count
        gc.collect()

        print('Computing the number of channels associated with a given IP address and app...')
        ip_app_count = train[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[
            ['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})

        print('Merging the channels data with the main data set...')
        train = train.merge(ip_app_count, on=['ip', 'app'], how='left')
        del ip_app_count
        gc.collect()

        print('Computing the number of channels associated with a given IP address, app, and os...')
        ip_app_os_count = train[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[
            ['channel']].count().reset_index().rename(columns={'channel': 'ip_app_os_count'})
        print('Merging the channels data with the main data set...')
        train = train.merge(ip_app_os_count, on=['ip', 'app', 'os'], how='left')
        del ip_app_os_count
        gc.collect()

        features = train[['n_channels', 'ip_app_count', 'ip_app_os_count']]
        return features[:len(df_train)].reset_index(drop=True), \
               features[len(df_train): len(df_train) + len(df_valid)].reset_index(drop=True), \
               features[len(df_train) + len(df_valid):].reset_index(drop=True)
