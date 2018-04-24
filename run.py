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
    ClickTime
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
}

unparallelizable_feature_map = {
    'komaki_lda_5': features.category_vector.KomakiLDA5,
    'komaki_lda_5_no_device': features.category_vector.KomakiLDA5NoDevice,
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
            sampled_train_dataset = load_train_dataset(sampled_train_feature_paths)

        threshold = pd.Timestamp('2017-11-08 16:00:00')
        sampled_train_data = sampled_train_dataset[sampled_train_dataset.click_time < threshold].drop('click_time', axis=1)
        sampled_valid_data = sampled_train_dataset[sampled_train_dataset.click_time >= threshold].drop('click_time', axis=1)
        predictors = sampled_train_data.columns.drop(target_variable)
        gc.collect()

        with simple_timer("Train model with validation"):
            booster, result = model.train_and_predict(train=sampled_train_data,
                                                      valid=sampled_valid_data,
                                                      categorical_features=categorical_features,
                                                      target=target_variable,
                                                      params=config['model'])
        best_iteration = booster.best_iteration
        with simple_timer("Train model without validation"):
            booster = model.train_without_validation(train=sampled_train_dataset.drop('click_time', axis=1),
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

    if not options.train_only:
        prepare_submission(options, prediction_boosters, predictors, test_feature_paths)
    dump_json_log(options, train_results, output_directory)


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

    bagging_predictions = [[] for _ in prediction_boosters]
    for i, sub_index in enumerate(split_index(df_test)):
        with simple_timer("Load {}-th test features batch".format(i)):
            test_data = load_dataset(test_feature_paths, sub_index)
            test_data['click_id'] = df_test.click_id.loc[sub_index]
            gc.collect()

        with simple_timer("Create prediction on {}-th test features batch".format(i)):
            for j, booster in enumerate(prediction_boosters):
                prediction = booster.predict(test_data[predictors])
                bagging_predictions[j].extend(list(prediction))

    df_test['prediction'] = sum(np.array(pred) for pred in bagging_predictions) / len(bagging_predictions)
    old_click_to_prediction = {cid: pred for (cid, pred) in zip(df_test.click_id, df_test.prediction)}

    click_ids = []
    predictions = []
    for (new_click_id, old_click_id) in zip(id_mapper.new_click_id, id_mapper.old_click_id):
        if old_click_id not in old_click_to_prediction:
            continue
        click_ids.append(new_click_id)
        predictions.append(old_click_to_prediction[old_click_id])
    submission = pd.DataFrame({'click_id': click_ids, '{}'.format(target_variable): predictions})
    submission_path = os.path.join(os.path.dirname(__file__), output_directory,
                                   os.path.basename(options.config) + '.submission.csv')
    submission.sort_values(by='click_id').to_csv(submission_path, index=False)


if __name__ == "__main__":
    main()
