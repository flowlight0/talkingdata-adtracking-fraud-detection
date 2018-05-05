import gc

import numpy as np
import pandas as pd

from features import FeatherFeatureDF


class Ip(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return ['ip']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        return df_train[['ip']], df_test[['ip']]


class IpForFiltering(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return ['ip_for_filtering']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        column = 'ip_for_filtering'
        df_train[column] = df_train['ip']
        df_test[column] = df_test['ip']
        return df_train[[column]], df_test[[column]]


class App(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return ['app']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        return df_train[['app']], df_test[['app']]


class Os(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return ['os']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        return df_train[['os']], df_test[['os']]


class Channel(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return ['channel']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        return df_train[['channel']], df_test[['channel']]


class ClickHour(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        # We don't use 'day' information because time span of our dataset is too short
        return ['hour']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        df_train['hour'] = pd.to_datetime(df_train.click_time).dt.hour.astype('uint8')
        df_test['hour'] = pd.to_datetime(df_test.click_time).dt.hour.astype('uint8')
        return df_train[['hour']], df_test[['hour']]


class ClickSecond(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        # We don't use 'day' information because time span of our dataset is too short
        return ['second']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        df_train['second'] = pd.to_datetime(df_train.click_time).dt.second.astype('uint8')
        df_test['second'] = pd.to_datetime(df_test.click_time).dt.second.astype('uint8')
        return df_train[['second']], df_test[['second']]


class ClickMinute(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        # We don't use 'day' information because time span of our dataset is too short
        return ['minute']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        df_train['minute'] = pd.to_datetime(df_train.click_time).dt.minute.astype('uint8')
        df_test['minute'] = pd.to_datetime(df_test.click_time).dt.minute.astype('uint8')
        return df_train[['minute']], df_test[['minute']]


class ZeroMinute(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return []

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        df_train['zero-minute'] = pd.to_datetime(df_train.click_time).dt.minute.astype('uint8') == 0
        df_test['zero-minute'] = pd.to_datetime(df_test.click_time).dt.minute.astype('uint8') == 0
        return df_train[['zero-minute']], df_test[['zero-minute']]


# Actually this is used for splitting dataset
class ClickTime(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        # We don't use 'day' information because time span of our dataset is too short
        return []

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        return df_train[['click_time']], df_test[['click_time']]


class BasicCount(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return []

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        print("Creating new count features: 'n_channels', 'ip_app_count', 'ip_app_os_count'...")
        train: pd.DataFrame = pd.concat([df_train, df_test])
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
        return features[:len(df_train)].reset_index(drop=True), features[len(df_train):].reset_index(drop=True)


class Device(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return ['device']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        return df_train[['device']], df_test[['device']]


class IsAttributed(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return ['is_attributed']

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        df_test['is_attributed'] = np.zeros(len(df_test), dtype=np.uint8)
        return df_train[['is_attributed']], df_test[['is_attributed']]


class DuplicatedRowIndexDiff(FeatherFeatureDF):
    @staticmethod
    def categorical_features():
        return []

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        return self.calculate(df_train), self.calculate(df_test)

    @staticmethod
    def calculate(df: pd.DataFrame):
        is_duplicated = df.duplicated(subset=['ip', 'device', 'os', 'channel', 'app', 'click_time'], keep=False)
        features = np.zeros(len(df))
        features[~is_duplicated] = np.nan

        curr_start_index = None
        prev_columns = None
        dup_df = df[is_duplicated]
        dup_rows = zip(dup_df.ip, dup_df.device, dup_df.os, dup_df.channel, dup_df.app, dup_df.click_time)
        for index, curr_columns in zip(dup_df.index, zip(dup_rows)):
            if prev_columns != curr_columns:
                curr_start_index = index
            features[index] = index - curr_start_index
            prev_columns = curr_columns
        df['DuplicateRowIndexDiff'] = features
        return df[['DuplicateRowIndexDiff']]
