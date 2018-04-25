import os
import subprocess
from typing import List, Tuple

import pandas as pd

from features import FeatherFeature


class FeatherFeatureCommand(FeatherFeature):
    def create_features_impl(self, train_path: str, test_path: str, train_feature_paths: List[Tuple[str, pd.Index]],
                             test_feature_path: str):
        assert len(train_feature_paths) > 0
        tmp_train_feature_path = train_feature_paths[0][0] + '.tmp'
        args = [os.path.join(os.path.dirname(__file__), '../cpp/{}'.format(self.get_command_name())), train_path,
                test_path, tmp_train_feature_path, test_feature_path]
        args.extend(self.get_parameters())
        subprocess.call(args)
        train_feature = pd.read_feather(tmp_train_feature_path)
        for train_feature_path, index in train_feature_paths:
            train_feature.loc[index].reset_index(drop=True).to_feather(train_feature_path)
        os.remove(tmp_train_feature_path)

    @staticmethod
    def get_command_name() -> str:
        raise NotImplementedError()

    @staticmethod
    def get_parameters() -> List[str]:
        return []


def generate_future_click_count(window_size_in_seconds):
    class FutureClickCount(FeatherFeatureCommand):
        @staticmethod
        def get_command_name() -> str:
            return 'future_click_count_main'

        @staticmethod
        def get_parameters() -> List[str]:
            return [str(window_size_in_seconds)]

        @staticmethod
        def categorical_features():
            return []

        @property
        def name(self):
            return super().name + '_{}'.format(window_size_in_seconds)

    return FutureClickCount


def generate_future_click_ratio(window_size_in_seconds):
    class FutureClickRatio(FeatherFeatureCommand):
        @staticmethod
        def get_command_name() -> str:
            return 'future_click_ratio_main'

        @staticmethod
        def get_parameters() -> List[str]:
            return [str(window_size_in_seconds)]

        @staticmethod
        def categorical_features():
            return []

        @property
        def name(self):
            return super().name + '_{}'.format(window_size_in_seconds)

    return FutureClickRatio


def generate_past_click_count(window_size_in_seconds):
    class PastClickCount(FeatherFeatureCommand):
        @staticmethod
        def get_command_name() -> str:
            return 'past_click_count_main'

        @staticmethod
        def get_parameters() -> List[str]:
            return [str(window_size_in_seconds)]

        @staticmethod
        def categorical_features():
            return []

        @property
        def name(self):
            return super().name + '_{}'.format(window_size_in_seconds)

    return PastClickCount


def generate_past_click_ratio(window_size_in_seconds):
    class PastClickRatio(FeatherFeatureCommand):
        @staticmethod
        def get_command_name() -> str:
            return 'past_click_ratio_main'

        @staticmethod
        def get_parameters() -> List[str]:
            return [str(window_size_in_seconds)]

        @staticmethod
        def categorical_features():
            return []

        @property
        def name(self):
            return super().name + '_{}'.format(window_size_in_seconds)

    return PastClickRatio


class NextClickTimeDelta(FeatherFeatureCommand):
    @staticmethod
    def get_command_name():
        return 'next_click_time_delta_main'

    @staticmethod
    def categorical_features():
        return []


class PrevClickTimeDelta(FeatherFeatureCommand):
    @staticmethod
    def get_command_name():
        return 'prev_click_time_delta_main'

    @staticmethod
    def categorical_features():
        return []


# class NextClickTimeDeltaV2(FeatherFeature):
#     def create_features_impl(self, train_path, valid_input, test_path, train_output, valid_output, test_feature_path):
#         args = [os.path.join(os.path.dirname(__file__), '../cpp/next_click_time_delta_v2_main'), train_path,
#                 valid_input, test_path, train_output, valid_output, test_feature_path]
#         subprocess.call(args)
#
#     @staticmethod
#     def categorical_features():
#         return []
#
#
# class PrevClickTimeDeltaV2(FeatherFeature):
#     def create_features_impl(self, train_path, valid_input, test_path, train_output, valid_output, test_feature_path):
#         args = [os.path.join(os.path.dirname(__file__), '../cpp/prev_click_time_delta_v2_main'), train_path,
#                 valid_input, test_path, train_output, valid_output, test_feature_path]
#         subprocess.call(args)
#
#     @staticmethod
#     def categorical_features():
#         return []
#
#
# class NextClickTimeDeltaV3(FeatherFeature):
#     def create_features_impl(self, train_path, valid_input, test_path, train_output, valid_output, test_feature_path):
#         args = [os.path.join(os.path.dirname(__file__), '../cpp/next_click_time_delta_v3_main'), train_path,
#                 valid_input, test_path, train_output, valid_output, test_feature_path]
#         subprocess.call(args)
#
#     @staticmethod
#     def categorical_features():
#         return []
#
#
# class PrevClickTimeDeltaV3(FeatherFeature):
#     def create_features_impl(self, train_path, valid_input, test_path, train_output, valid_output, test_feature_path):
#         args = [os.path.join(os.path.dirname(__file__), '../cpp/prev_click_time_delta_v3_main'), train_path,
#                 valid_input, test_path, train_output, valid_output, test_feature_path]
#         subprocess.call(args)
#
#     @staticmethod
#     def categorical_features():
#         return []

class ExactSameClick(FeatherFeatureCommand):
    @staticmethod
    def get_command_name():
        return 'exact_same_click_main'

    @staticmethod
    def categorical_features():
        return []


class ExactSameClickId(FeatherFeatureCommand):
    @staticmethod
    def get_command_name():
        return 'exact_same_click_id_main'

    @staticmethod
    def categorical_features():
        return []


class AllClickCount(FeatherFeatureCommand):
    @staticmethod
    def get_command_name():
        return 'all_click_count_main'

    @staticmethod
    def categorical_features():
        return []


class AverageAttributedRatio(FeatherFeatureCommand):
    @staticmethod
    def get_command_name():
        return 'average_attributed_ratio_main'

    @staticmethod
    def categorical_features():
        return []


class CumulativeClickCount(FeatherFeatureCommand):
    @staticmethod
    def get_command_name():
        return 'cumulative_click_count_main'

    @staticmethod
    def categorical_features():
        return []


class CumulativeClickCountFuture(FeatherFeatureCommand):
    @staticmethod
    def get_command_name():
        return 'cumulative_click_count_future_main'

    @staticmethod
    def categorical_features():
        return []


class MedianAttributeTime(FeatherFeatureCommand):
    @staticmethod
    def get_command_name():
        return 'median_attribute_time_main'

    @staticmethod
    def categorical_features():
        return []


class MedianAttributeTimePast(FeatherFeatureCommand):
    @staticmethod
    def get_command_name():
        return 'median_attribute_time_past_main'

    @staticmethod
    def categorical_features():
        return []


class MedianAttributeTimePastV2(FeatherFeatureCommand):
    @staticmethod
    def get_command_name():
        return 'median_attribute_time_past_v2_main'

    @staticmethod
    def categorical_features():
        return []

# class NextApp(FeatherFeature):
#     def create_features_impl(self, train_path, valid_input, test_path, train_output, valid_output, test_feature_path):
#         args = [os.path.join(os.path.dirname(__file__), '../cpp/next_app_main'), train_path,
#                 valid_input,
#                 test_path, train_output, valid_output, test_feature_path]
#         subprocess.call(args)
#
#     @staticmethod
#     def categorical_features():
#         return ['NextApp-ip-dev-os', 'NextApp-ip-dev-os-cha']
#
#
# class PrevApp(FeatherFeature):
#     def create_features_impl(self, train_path, valid_input, test_path, train_output, valid_output, test_feature_path):
#         args = [os.path.join(os.path.dirname(__file__), '../cpp/prev_app_main'), train_path,
#                 valid_input,
#                 test_path, train_output, valid_output, test_feature_path]
#         subprocess.call(args)
#
#     @staticmethod
#     def categorical_features():
#         return ['PrevApp-ip-dev-os', 'PrevApp-ip-dev-os-cha']
#
#
# class NextChannel(FeatherFeature):
#     def create_features_impl(self, train_path, valid_input, test_path, train_output, valid_output, test_feature_path):
#         args = [os.path.join(os.path.dirname(__file__), '../cpp/next_channel_main'), train_path,
#                 valid_input,
#                 test_path, train_output, valid_output, test_feature_path]
#         subprocess.call(args)
#
#     @staticmethod
#     def categorical_features():
#         return ['NextChannel-ip-dev-os', 'NextChannel-ip-app-dev-os']
#
#
# class PrevChannel(FeatherFeature):
#     def create_features_impl(self, train_path, valid_input, test_path, train_output, valid_output, test_feature_path):
#         args = [os.path.join(os.path.dirname(__file__), '../cpp/prev_channel_main'), train_path,
#                 valid_input,
#                 test_path, train_output, valid_output, test_feature_path]
#         subprocess.call(args)
#
#     @staticmethod
#     def categorical_features():
#         return ['PrevChannel-ip-dev-os', 'PrevChannel-ip-app-dev-os']
