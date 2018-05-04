import argparse
import gc
import itertools
import json
import os
import time
from functools import partial
from multiprocessing.pool import Pool
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import pandas.testing

import features.time_series_click
import features.category_vector
from features import Feature
from features.basic import Ip, App, Os, Device, Channel, ClickHour, BasicCount, IsAttributed, ClickSecond, ClickMinute, \
    ClickTime, ZeroMinute, DuplicatedRowIndexDiff
from models import LightGBM, Model
from utils import dump_json_log, simple_timer

parallelizable_feature_map = {
    'ip': Ip,
    'app': App,
    'os': Os,
    'device': Device,
    'channel': Channel,
    'hour': ClickHour,
    'click_time': ClickTime,
    'minute': ClickSecond,
    'second': ClickMinute,
    'count': BasicCount,
    'is_attributed': IsAttributed,
    'zero-minute': ZeroMinute,
    'future_click_count_1': features.time_series_click.generate_future_click_count(60),
    'future_click_count_10': features.time_series_click.generate_future_click_count(600),
    'past_click_count_10': features.time_series_click.generate_past_click_count(600),
    'future_click_count_80': features.time_series_click.generate_future_click_count(4800),
    'past_click_count_80': features.time_series_click.generate_past_click_count(4800),
    'future_click_ratio_10': features.time_series_click.generate_future_click_ratio(600),
    'past_click_ratio_10': features.time_series_click.generate_future_click_ratio(600),
    'future_click_ratio_80': features.time_series_click.generate_future_click_ratio(4800),
    'past_click_ratio_80': features.time_series_click.generate_future_click_ratio(4800),
    'next_click_time_delta': features.time_series_click.NextClickTimeDelta,
    'prev_click_time_delta': features.time_series_click.PrevClickTimeDelta,
    'exact_same_click': features.time_series_click.ExactSameClick,  # It will be duplicated with all id counts
    'exact_same_click_id': features.time_series_click.ExactSameClickId,
    'all_click_count': features.time_series_click.AllClickCount,
    'average_attributed_ratio': features.time_series_click.AverageAttributedRatio,
    'cumulative_click_count': features.time_series_click.CumulativeClickCount,
    'cumulative_click_count_future': features.time_series_click.CumulativeClickCountFuture,
    'median_attribute_time': features.time_series_click.MedianAttributeTime,
    'median_attribute_time_past': features.time_series_click.MedianAttributeTimePast,
    'median_attribute_time_past_v2': features.time_series_click.MedianAttributeTimePastV2,
    'duplicated_row_index_diff': DuplicatedRowIndexDiff
}

unparallelizable_feature_map = {
    'komaki_lda_5': features.category_vector.KomakiLDA5,
    'komaki_lda_5_no_device': features.category_vector.KomakiLDA5NoDevice,
    'komaki_lda_10_no_device_1': features.category_vector.KomakiLDA10NoDevice_1,
    'komaki_lda_10_no_device_2': features.category_vector.KomakiLDA10NoDevice_2,
    'komaki_lda_20_no_device_ip': features.category_vector.KomakiLDA20NoDevice_Ip,
    'komaki_lda_20_no_device_os': features.category_vector.KomakiLDA20NoDevice_Os,
    'komaki_lda_20_no_device_channel': features.category_vector.KomakiLDA20NoDevice_Channel,
    'komaki_lda_20_no_device_app': features.category_vector.KomakiLDA20NoDevice_App,
    'komaki_pca_5': features.category_vector.KomakiPCA5,
    'komaki_pca_5_no_device': features.category_vector.KomakiPCA5NoDevice,
    'komaki_nmf_5': features.category_vector.KomakiNMF5,
    'komaki_nmf_5_no_device': features.category_vector.KomakiNMF5NoDevice,
    'single_pca_count': features.category_vector.SinglePCACount,
    'single_pca_tfidf': features.category_vector.SinglePCATfIdf,
    'komaki_lda_5_mindf_1': features.category_vector.KomakiLDA5MinDF1,
    "user_item_lda_30": features.category_vector.UserItemLDA,
    "item_user_lda_30": features.category_vector.ItemUserLDA,
}

models = {
    'lightgbm': LightGBM
}

output_directory = 'data/output'

target_variable = 'is_attributed'

in_test_hours = {4, 5, 9, 10, 13, 14}


# Now we don't set index when loading training features because they should have been already down-sampled.
def load_dataset(paths, index=None) -> pd.DataFrame:
    assert len(paths) > 0

    feature_datasets = []
    for path in paths:
        if index is None:
            feature_datasets.append(pd.read_feather(path))
        else:
            feature_datasets.append(pd.read_feather(path).loc[index])
        gc.collect()
    # check if all of feature dataset share the same index
    index = feature_datasets[0].index
    for feature_dataset in feature_datasets[1:]:
        pandas.testing.assert_index_equal(index, feature_dataset.index)

    return pd.concat(feature_datasets, axis=1)


def get_dataset_filename(config, dataset_type: str) -> str:
    return os.path.join(config['dataset']['input_directory'], config['dataset']['files'][dataset_type])


class DownSampler(object):
    def __init__(self, config, data_path, random_states):
        self.data_path = data_path
        self.random_states = random_states
        self.config = config
        with simple_timer("Load training dataset in random sampled index calculation"):
            data = pd.read_feather(data_path)
        with simple_timer("Create cache file for indices"):
            for random_state in self.random_states:
                cache_file = self.get_index_cache_file(random_state)
                if not os.path.exists(cache_file):
                    index = self.negative_down_sampling(data, random_state)
                    joblib.dump(index, cache_file)

    def get_index_cache_file(self, random_state: int) -> str:
        cache_dir: str = self.config['dataset']['cache_directory']
        cache_file = os.path.join(cache_dir, os.path.basename(self.data_path) + '.index_' + str(random_state) + '.pkl')
        return cache_file

    @staticmethod
    def negative_down_sampling(data: pd.DataFrame, random_state: int):
        with simple_timer("Get positive data"):
            positive_data = data[data[target_variable] == 1]
            positive_ratio = float(len(positive_data)) / len(data)
        with simple_timer("Get negative data"):
            negative_data = data[data[target_variable] == 0].sample(
                frac=positive_ratio / (1 - positive_ratio), random_state=random_state)
        return positive_data.index.union(negative_data.index).sort_values()

    # prepare must be called before calling this function
    def get_indices(self) -> List[Tuple[int, pd.Index]]:
        rs = []
        for random_state in self.random_states:
            cache_file = self.get_index_cache_file(random_state)
            if not os.path.exists(cache_file):
                raise FileNotFoundError("Cache file must be created before!")
            index = joblib.load(cache_file)
            rs.append((random_state, index))
        return rs


def get_parallelizable_feature_list(config) -> List[str]:
    for feature in config['features']:
        assert feature in parallelizable_feature_map or feature in unparallelizable_feature_map, \
            "Unknown feature {}".format(feature)
    features = [feature for feature in config['features'] if feature in parallelizable_feature_map]
    return [*features, target_variable, 'click_time']


def get_unparallelizable_feature_list(config) -> List[str]:
    for feature in config['features']:
        assert feature in parallelizable_feature_map or feature in unparallelizable_feature_map, \
            "Unknown feature {}".format(feature)
    return [feature for feature in config['features'] if feature in unparallelizable_feature_map]


def get_feature(feature_name: str, config) -> Feature:
    cache_dir: str = config['dataset']['cache_directory']
    if feature_name in parallelizable_feature_map:
        return parallelizable_feature_map[feature_name](cache_dir)
    else:
        return unparallelizable_feature_map[feature_name](cache_dir)


def load_feature(feature_name: str, train_path: str, test_path: str, sampler: DownSampler, config) -> Tuple[
    List[str], str]:
    feature = get_feature(feature_name=feature_name, config=config)
    random_states_ = sampler.get_indices()
    return feature.create_features(train_path, test_path, random_states=random_states_)


def load_features(config, random_states: List[int]) -> Tuple[List[List[str]], List[str]]:
    tr_path = get_dataset_filename(config, 'train')
    te_path = get_dataset_filename(config, 'test')
    train_feature_paths_lists = [[] for _ in random_states]
    test_feature_paths = []

    sampler = DownSampler(config, tr_path, random_states)
    # We fix the number of processes to four because of memory limitation
    parallelizable_feature_list = get_parallelizable_feature_list(config)
    unparallelizable_feature_list = get_unparallelizable_feature_list(config)
    print("Create features in parallel: ", parallelizable_feature_list)
    print("Create features without parallelism: ", unparallelizable_feature_list)

    with Pool(4) as p:
        res = p.map(partial(load_feature, train_path=tr_path, test_path=te_path,
                            sampler=sampler, config=config), parallelizable_feature_list)
        for tr_feature_path_list, te_feature_path in res:
            for i, tr_feature_path in enumerate(tr_feature_path_list):
                train_feature_paths_lists[i].append(tr_feature_path)
            test_feature_paths.append(te_feature_path)

    for feature_name in unparallelizable_feature_list:
        tr_feature_path_list, te_feature_path = load_feature(feature_name, tr_path, te_path, sampler, config)
        for i, tr_feature_path in enumerate(tr_feature_path_list):
            train_feature_paths_lists[i].append(tr_feature_path)
        test_feature_paths.append(te_feature_path)
    return train_feature_paths_lists, test_feature_paths


def load_categorical_features(config) -> List[str]:
    return list(itertools.chain(
        *[get_feature(feature, config).categorical_features() for feature in config['features']]))


def load_train_dataset(train_feature_paths: List[str]) -> pd.DataFrame:
    assert len(train_feature_paths) > 0
    print(train_feature_paths)
    df_train = load_dataset(train_feature_paths)
    assert 'is_attributed' in df_train.columns
    return df_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/lightgbm_0.json')
    parser.add_argument('--train_only', default=False, action='store_true')
    options = parser.parse_args()
    config = json.load(open(options.config))
    assert config['model']['name'] in models  # check model's existence before getting datasets

    model: Model = models[config['model']['name']]()
    train_results = []
    categorical_features = load_categorical_features(config)
    negative_down_sampling_config = config['dataset']['negative_down_sampling']

    if not negative_down_sampling_config['enabled']:
        raise NotImplementedError("We should always downsample")

    with simple_timer("Create features"):
        sampled_train_feature_paths_list, test_feature_paths = \
            load_features(config, list(range(negative_down_sampling_config['bagging_size'])))

    prediction_boosters = []
    predictors = []  # this list must be filled in the following loop.
    for i, sampled_train_feature_paths in enumerate(sampled_train_feature_paths_list):
        start_time = time.time()
        with simple_timer("Load train features"):
            sampled_all_train = load_train_dataset(sampled_train_feature_paths)

        # Hard-coded threshold, last one day of training dataset is used for validation
        threshold = pd.Timestamp('2017-11-08 16:00:00')
        sampled_train_data = sampled_all_train[sampled_all_train.click_time < threshold].drop('click_time', axis=1)
        sampled_valid_data = sampled_all_train[sampled_all_train.click_time >= threshold].drop('click_time', axis=1)

        sampled_train_data_weight = None
        sampled_train_data_all_weight = None
        if config.get('test_hours', {}).get('filter_validation', False):
            assert 'hour' in sampled_valid_data.columns, "This script now assumes we include 'hour' in features. " \
                                                         "Sorry for bad implementation:) "

            weight = config.get('test_hours').get("train_weight", 1.0)
            sampled_valid_data = sampled_valid_data[sampled_valid_data.hour.isin(in_test_hours)]
            sampled_train_data_weight = np.ones(len(sampled_train_data))
            sampled_train_data_weight[sampled_train_data.hour.isin(in_test_hours)] = weight
            sampled_train_data_all_weight = np.ones(len(sampled_all_train))
            sampled_train_data_all_weight[sampled_all_train.hour.isin(in_test_hours)] = weight

        predictors = sampled_train_data.columns.drop(target_variable)
        gc.collect()

        with simple_timer("Train model with validation"):
            booster, result = model.train_and_predict(train=sampled_train_data,
                                                      valid=sampled_valid_data,
                                                      weight=sampled_train_data_weight,
                                                      categorical_features=categorical_features,
                                                      target=target_variable,
                                                      params=config['model'])
        best_iteration = booster.best_iteration
        if not options.train_only:
            with simple_timer("Train model without validation"):
                booster = model.train_without_validation(train=sampled_all_train.drop('click_time', axis=1),
                                                         weight=sampled_train_data_all_weight,
                                                         categorical_features=categorical_features,
                                                         target=target_variable,
                                                         params=config['model'],
                                                         best_iteration=best_iteration)
                prediction_boosters.append(booster)

        # This only works when we are using LightGBM
        train_results.append({
            'train_auc': result['train']['auc'][best_iteration],
            'valid_auc': result['valid']['auc'][best_iteration],
            'best_iteration': best_iteration,
            'train_time': time.time() - start_time,
            'feature_importance': {name: int(score) for name, score in
                                   zip(booster.feature_name(), booster.feature_importance())}
        })
        print("Finished {}-th bag training: {}".format(i, str(train_results[-1])))

    dump_json_log(options, train_results, output_directory)
    if not options.train_only:
        prepare_submission(options, prediction_boosters, predictors, test_feature_paths)


def split_index(df, num_splits=5):
    assert num_splits > 0
    sizes = []
    for i in range(num_splits):
        sizes.append(len(df) * i // num_splits)
    sizes.append(len(df))

    for i in range(num_splits):
        yield df.index[sizes[i]: sizes[i + 1]]


def prepare_submission(options, prediction_boosters: List, predictors: List[str], test_feature_paths: List[str]):
    config = json.load(open(options.config))

    with simple_timer("Load click id mapping"):
        id_mapper_file = 'data/working/id_mapping.feather'
        assert os.path.exists(id_mapper_file), "Please download {} from s3 before running this script".format(
            id_mapper_file)
        id_mapper = pd.read_feather(id_mapper_file)

    with simple_timer("Load test file"):
        required_ids = set(id_mapper['old_click_id'])
        df_test = pd.read_feather(get_dataset_filename(config, 'test'))
        df_test = df_test[df_test.click_id.isin(required_ids)]
        del required_ids
        gc.collect()

    old_bagging_predictions = [[] for _ in prediction_boosters]
    for i, sub_index in enumerate(split_index(df_test)):
        with simple_timer("Load {}-th test features batch".format(i)):
            test_data = load_dataset(test_feature_paths, sub_index)
            test_data['click_id'] = df_test.click_id.loc[sub_index]
            gc.collect()

        with simple_timer("Create prediction on {}-th test features batch".format(i)):
            for j, booster in enumerate(prediction_boosters):
                prediction = booster.predict(test_data[predictors])
                old_bagging_predictions[j].extend(list(prediction))

    old_click_to_predictions = []
    for i, old_bagging_prediction in enumerate(old_bagging_predictions):
        old_click_to_prediction = {cid: pred for (cid, pred) in zip(df_test.click_id, old_bagging_prediction)}
        old_click_to_predictions.append(old_click_to_prediction)

    click_ids = []
    predictions = [[] for _ in range(len(old_click_to_predictions))]
    for (new_click_id, old_click_id) in zip(id_mapper.new_click_id, id_mapper.old_click_id):
        if old_click_id not in old_click_to_predictions[0]:
            continue
        click_ids.append(new_click_id)
        for i, old_click_to_prediction in enumerate(old_click_to_predictions):
            predictions[i].append(old_click_to_prediction[old_click_id])

    submission = pd.DataFrame({'click_id': click_ids})
    pred_columns = []
    rank_columns = []
    for i, prediction in enumerate(predictions):
        pred_column = 'pred-{}'.format(i)
        rank_column = 'rank-{}'.format(i)
        submission[pred_column] = prediction
        submission[rank_column] =  submission[pred_column].rank()
        pred_columns.append(pred_column)
        rank_columns.append(rank_column)

    if config.get('rank_average', False):
        submission[target_variable] = submission[rank_columns].mean(axis=1) / (len(submission))
    else:
        submission[target_variable] = submission[pred_columns].mean(axis=1)

    submission_path = os.path.join(os.path.dirname(__file__), output_directory,
                                   os.path.basename(options.config) + '.submission.csv')
    submission[['click_id', target_variable]].sort_values(by='click_id').to_csv(submission_path, index=False)

if __name__ == "__main__":
    main()
