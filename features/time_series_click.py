import os
import subprocess
import time

import numpy as np
import pandas as pd

from features import FeatherFeatureDF, FeatherFeature


class IntervalCount(FeatherFeatureDF):
    @staticmethod
    def validate_timestamps(df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame):
        if not df_train.click_time.is_monotonic:
            raise ValueError('Click train in train must be monotonic')

        if not df_valid.click_time.is_monotonic:
            raise ValueError('Click valid in train must be monotonic')

        if not df_test.click_time.is_monotonic:
            raise ValueError('Click test in train must be monotonic')

        if df_train.click_time[len(df_train) - 1] > df_valid.click_time[0]:
            raise ValueError("Train's timestamp must be smaller than valid's ones")

        if df_valid.click_time[len(df_valid) - 1] > df_test.click_time[0]:
            raise ValueError("Valid's timestamp must be smaller than test's ones")


def generate_future_interval_count(window_size):
    class FutureIntervalCountSimple(IntervalCount):
        @property
        def name(self):
            return super().name + '_{}'.format(window_size)

        @staticmethod
        def categorical_features():
            return []

        def create_features_from_dataframe(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame):
            self.validate_timestamps(df_train, df_valid, df_test)
            data: pd.DataFrame = pd.concat([df_train, df_valid, df_test])
            data.reset_index(drop=True, inplace=True)
            data['click_time'] = data.click_time.astype(np.int64) // 10 ** 9

            features = ['ip', 'app', 'os', 'device', 'channel']
            features_df = pd.DataFrame(index=data.index)

            for feature in features:
                feature_values = np.zeros(len(data.index), dtype=np.int32)
                start_time = time.time()
                for _, group in data.groupby(by=feature)['click_time']:
                    cursor = 0
                    group_index = group.index
                    click_times = group.values
                    for i, (index, click_time) in enumerate(zip(group_index, click_times)):
                        while (cursor < len(click_times)) and ((click_times[cursor] - click_time) <= window_size):
                            cursor += 1
                        feature_values[index] = cursor - i
                print('Processed {}: {:.3} [s]'.format(feature, time.time() - start_time))
                assert (feature_values == 0).sum() == 0
                features_df['fic-{}-{}'.format(feature, window_size)] = feature_values
            features_train = features_df[:len(df_train)]
            features_train.index = df_train.index
            features_valid = features_df[len(df_train):len(df_train) + len(df_valid)]
            features_valid.index = df_valid.index
            features_test = features_df[len(df_train) + len(df_valid):]
            features_test.index = df_test.index
            print(features_df.head())
            return features_train, features_valid, features_test

    return FutureIntervalCountSimple


def generate_past_interval_count(window_size):
    class PastIntervalCountSimple(IntervalCount):
        @property
        def name(self):
            return super().name + '_{}'.format(window_size)

        @staticmethod
        def categorical_features():
            return []

        def create_features_from_dataframe(self, df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame):
            self.validate_timestamps(df_train, df_valid, df_test)
            data: pd.DataFrame = pd.concat([df_train, df_valid, df_test])
            data.reset_index(drop=True, inplace=True)
            data['click_time'] = data.click_time.astype(np.int64) // 10 ** 9

            features = ['ip', 'app', 'os', 'device', 'channel']
            features_df = pd.DataFrame(index=data.index)

            for feature in features:
                feature_values = np.zeros(len(data.index), dtype=np.int32)
                start_time = time.time()
                for _, group in data.groupby(by=feature)['click_time']:
                    cursor = 0
                    group_index = group.index
                    click_times = group.values
                    for i, (index, click_time) in enumerate(zip(group_index, click_times)):
                        while (cursor < len(click_times)) and ((click_time - click_times[cursor]) > window_size):
                            cursor += 1
                        feature_values[index] = i - cursor + 1
                print('Processed {}: {:.3} [s]'.format(feature, time.time() - start_time))
                assert (feature_values == 0).sum() == 0
                features_df['pic-{}-{}'.format(feature, window_size)] = feature_values
            features_train = features_df[:len(df_train)]
            features_train.index = df_train.index
            features_valid = features_df[len(df_train):len(df_train) + len(df_valid)]
            features_valid.index = df_valid.index
            features_test = features_df[len(df_train) + len(df_valid):]
            features_test.index = df_test.index
            print(features_df.head())
            return features_train, features_valid, features_test

    return PastIntervalCountSimple


def generate_future_click_count(window_size_in_seconds):
    class FutureClickCount(FeatherFeature):
        def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
            args = [os.path.join(os.path.dirname(__file__), '../cpp/future_click_count_main'), train_input, valid_input,
                    test_input, train_output, valid_output, test_output, str(window_size_in_seconds)]
            subprocess.call(args)

        @staticmethod
        def categorical_features():
            return []

        @property
        def name(self):
            return super().name + '_{}'.format(window_size_in_seconds)

    return FutureClickCount


def generate_future_click_ratio(window_size_in_seconds):
    class FutureClickRatio(FeatherFeature):
        def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
            args = [os.path.join(os.path.dirname(__file__), '../cpp/future_click_ratio_main'), train_input, valid_input,
                    test_input, train_output, valid_output, test_output, str(window_size_in_seconds)]
            subprocess.call(args)

        @staticmethod
        def categorical_features():
            return []

        @property
        def name(self):
            return super().name + '_{}'.format(window_size_in_seconds)

    return FutureClickRatio


def generate_past_click_count(window_size_in_seconds):
    class PastClickCount(FeatherFeature):
        def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
            args = [os.path.join(os.path.dirname(__file__), '../cpp/past_click_count_main'), train_input, valid_input,
                    test_input, train_output, valid_output, test_output, str(window_size_in_seconds)]
            subprocess.call(args)

        @staticmethod
        def categorical_features():
            return []

        @property
        def name(self):
            return super().name + '_{}'.format(window_size_in_seconds)

    return PastClickCount


def generate_past_click_ratio(window_size_in_seconds):
    class PastClickRatio(FeatherFeature):
        def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
            args = [os.path.join(os.path.dirname(__file__), '../cpp/past_click_ratio_main'), train_input, valid_input,
                    test_input, train_output, valid_output, test_output, str(window_size_in_seconds)]
            subprocess.call(args)

        @staticmethod
        def categorical_features():
            return []

        @property
        def name(self):
            return super().name + '_{}'.format(window_size_in_seconds)

    return PastClickRatio


class NextClickTimeDelta(FeatherFeature):
    def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
        args = [os.path.join(os.path.dirname(__file__), '../cpp/next_click_time_delta_main'), train_input, valid_input,
                test_input, train_output, valid_output, test_output]
        subprocess.call(args)

    @staticmethod
    def categorical_features():
        return []


class PrevClickTimeDelta(FeatherFeature):
    def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
        args = [os.path.join(os.path.dirname(__file__), '../cpp/prev_click_time_delta_main'), train_input, valid_input,
                test_input, train_output, valid_output, test_output]
        subprocess.call(args)

    @staticmethod
    def categorical_features():
        return []

class ExactSameClick(FeatherFeature):
    def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
        args = [os.path.join(os.path.dirname(__file__), '../cpp/exact_same_click_main'), train_input, valid_input,
                test_input, train_output, valid_output, test_output]
        subprocess.call(args)

    @staticmethod
    def categorical_features():
        return []


class ExactSameClickId(FeatherFeature):
    def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
        args = [os.path.join(os.path.dirname(__file__), '../cpp/exact_same_click_id_main'), train_input, valid_input,
                test_input, train_output, valid_output, test_output]
        subprocess.call(args)

    @staticmethod
    def categorical_features():
        return []


class AllClickCount(FeatherFeature):
    def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
        args = [os.path.join(os.path.dirname(__file__), '../cpp/all_click_count_main'), train_input, valid_input,
                test_input, train_output, valid_output, test_output]
        subprocess.call(args)

    @staticmethod
    def categorical_features():
        return []


class AverageAttributedRatio(FeatherFeature):
    def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
        args = [os.path.join(os.path.dirname(__file__), '../cpp/average_attributed_ratio_main'), train_input, valid_input,
                test_input, train_output, valid_output, test_output]
        subprocess.call(args)

    @staticmethod
    def categorical_features():
        return []


class CumulativeClickCount(FeatherFeature):
    def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
        args = [os.path.join(os.path.dirname(__file__), '../cpp/cumulative_click_count_main'), train_input,
                valid_input,
                test_input, train_output, valid_output, test_output]
        subprocess.call(args)

    @staticmethod
    def categorical_features():
        return []


class CumulativeClickCountFuture(FeatherFeature):
    def create_features_impl(self, train_input, valid_input, test_input, train_output, valid_output, test_output):
        args = [os.path.join(os.path.dirname(__file__), '../cpp/cumulative_click_count_future_main'), train_input,
                valid_input,
                test_input, train_output, valid_output, test_output]
        subprocess.call(args)

    @staticmethod
    def categorical_features():
        return []


if __name__ == '__main__':
    interval_count = generate_future_interval_count(600)
    data_dir = os.path.join(os.path.dirname(__file__), '../data/input')
    ic = interval_count(data_dir)
    df_train = pd.read_feather(os.path.join(data_dir, 'train_0.feather.small'))
    df_valid = pd.read_feather(os.path.join(data_dir, 'valid_0.feather.small'))
    df_test = pd.read_feather(os.path.join(data_dir, 'old_test.feather.small'))
    ic.create_features_from_dataframe(df_train, df_valid, df_test)
