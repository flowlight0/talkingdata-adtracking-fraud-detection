import argparse
import itertools
import json
import os
import time
import gc
from typing import List, Tuple

import numpy as np
import pandas as pd
import pandas.testing

import features.time_series_click
from features import Feature
from features.basic import Ip, App, Os, Device, Channel, ClickHour, BasicCount
from models import LightGBM, Model

feature_map = {
    'ip': Ip,
    'app': App,
    'os': Os,
    'device': Device,
    'channel': Channel,
    'hour': ClickHour,
    'count': BasicCount,
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
    'next_click_time_delta_v2': features.time_series_click.NextClickTimeDeltaV2,
    'prev_click_time_delta_v2': features.time_series_click.PrevClickTimeDeltaV2,
    'next_click_time_delta_v3': features.time_series_click.NextClickTimeDeltaV3,
    'prev_click_time_delta_v3': features.time_series_click.PrevClickTimeDeltaV3,
    'exact_same_click': features.time_series_click.ExactSameClick, # It will be duplicated with all id counts
    'exact_same_click_id': features.time_series_click.ExactSameClickId,
    'all_click_count': features.time_series_click.AllClickCount,
    'average_attributed_ratio': features.time_series_click.AverageAttributedRatio,
    'cumulative_click_count': features.time_series_click.CumulativeClickCount,
    'cumulative_click_count_future':  features.time_series_click.CumulativeClickCountFuture,
    'next_channel': features.time_series_click.NextChannel,
    'prev_channel': features.time_series_click.PrevChannel,
    'next_app': features.time_series_click.NextApp,
    'prev_app': features.time_series_click.PrevApp
}

models = {
    'lightgbm': LightGBM
}

output_directory = 'data/output'

target_variable = 'is_attributed'


def get_click_id_(path) -> pd.Series:
    return pd.read_feather(path)[['click_id']].astype('int32')


def get_click_id(config) -> pd.Series:
    return get_click_id_(get_dataset_filename(config, 'test_full'))


def get_target_(path, index=None) -> pd.Series:
    if index is None:
        return pd.read_feather(path)[[target_variable]]
    else:
        return pd.read_feather(path).loc[index, [target_variable]]


def get_target(config, dataset_type: str, index=None) -> pd.Series:
    return get_target_(get_dataset_filename(config, dataset_type), index)


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


def load_datasets(config, random_state) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_list = get_feature_list(config)
    assert len(feature_list) > 0

    train_path = get_dataset_filename(config, 'train')
    valid_path = get_dataset_filename(config, 'valid')
    test_path = get_dataset_filename(config, 'test_full')

    feature_path_lists = [feature.create_features(train_path, valid_path, test_path) for feature in feature_list]
    train_paths = [feature_path_list[0] for feature_path_list in feature_path_lists]
    valid_paths = [feature_path_list[1] for feature_path_list in feature_path_lists]

    train_index= negative_down_sampling(train_path, random_state=random_state)
    valid_index = negative_down_sampling(valid_path, random_state=random_state)
    gc.collect()
    train = load_dataset(train_paths, train_index)
    gc.collect()
    valid = load_dataset(valid_paths, valid_index)
    gc.collect()
    train[target_variable] = get_target(config, 'train', train_index)
    valid[target_variable] = get_target(config, 'valid', valid_index)
    return train, valid


def load_categorical_features(config) -> List[str]:
    return list(itertools.chain(*[feature_map[feature].categorical_features() for feature in config['features']]))


def dump_json_log(options, train_results):
    config = json.load(open(options.config))
    results = {
        'training': {
            'trials': train_results,
            'average_train_auc': np.mean([result['train_auc'] for result in train_results]),
            'average_valid_auc': np.mean([result['valid_auc'] for result in train_results]),
            'train_auc_std': np.std([result['train_auc'] for result in train_results]),
            'valid_auc_std': np.std([result['valid_auc'] for result in train_results]),
            'average_train_time': np.mean([result['train_time'] for result in train_results])
        },
        'config': config,
    }
    log_path = os.path.join(os.path.dirname(__file__), output_directory,
                            os.path.basename(options.config) + '.result.json')
    json.dump(results, open(log_path, 'w'), indent=2)


def negative_down_sampling(data_path, random_state):
    data = pd.read_feather(data_path)
    positive_data = data[data[target_variable] == 1]
    positive_ratio = float(len(positive_data)) / len(data)
    negative_data = data[data[target_variable] == 0].sample(
        frac=positive_ratio / (1 - positive_ratio), random_state=random_state)
    return positive_data.index.union(negative_data.index).sort_values()


def load_test_dataset(config, id_mapper):
    ids_we_need = set(id_mapper['old_click_id'])
    feature_list = get_feature_list(config)
    assert len(feature_list) > 0
    train_path = get_dataset_filename(config, 'train')
    valid_path = get_dataset_filename(config, 'valid')
    test_path = get_dataset_filename(config, 'test_full')
    feature_path_lists = [feature.create_features(train_path, valid_path, test_path) for feature in feature_list]
    test_feature_paths = [feature_path_list[2] for feature_path_list in feature_path_lists]

    df_test = pd.read_feather(test_path)
    df_test = df_test[df_test.click_id.isin(ids_we_need)]
    test = load_dataset(test_feature_paths, df_test.index)
    test['click_id'] = df_test.click_id
    return test


def get_feature_list(config):
    cache_dir: str = config['dataset']['cache_directory']
    for feature in config['features']:
        assert feature in feature_map, "Unknown feature {}".format(feature)
    feature_list: List[Feature] = [feature_map[feature](cache_dir) for feature in config['features']]
    return feature_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/lightgbm_0.json')
    options = parser.parse_args()
    config = json.load(open(options.config))
    assert config['model']['name'] in models  # check model's existence before getting datasets

    id_mapper_file = 'data/working/id_mapping.feather'
    assert os.path.exists(id_mapper_file), "Please download {} from s3 before running this script".format(id_mapper_file)
    id_mapper = pd.read_feather(id_mapper_file)
    test_data = load_test_dataset(config, id_mapper)

    categorical_features = load_categorical_features(config)
    model: Model = models[config['model']['name']]()
    predictions = []
    train_results = []
    negative_down_sampling_config = config['dataset']['negative_down_sampling']

    if not negative_down_sampling_config['enabled']:
        raise NotImplementedError("We should always downsample")

    for i in range(negative_down_sampling_config['bagging_size']):
        start_time = time.time()
        sampled_train_data, sampled_valid_data = load_datasets(config, random_state=i)
        predictors = sampled_train_data.columns.drop(target_variable)
        booster, result = model.train_and_predict(train=sampled_train_data,
                                                  valid=sampled_valid_data,
                                                  categorical_features=categorical_features,
                                                  target=target_variable,
                                                  params=config['model'])
        test_prediction_start_time = time.time()
        prediction = booster.predict(test_data[predictors])
        test_prediction_elapsed_time = time.time() - test_prediction_start_time

        valid_prediction_start_time = time.time()
        valid_prediction_elapsed_time = time.time() - valid_prediction_start_time
        predictions.append(prediction)
        # This only works when we are using LightGBM
        train_results.append({
            'train_auc': result['train']['auc'][booster.best_iteration],
            'valid_auc': result['valid']['auc'][booster.best_iteration],
            'best_iteration': booster.best_iteration,
            'train_time': time.time() - start_time,
            'prediction_time': {
                'test': test_prediction_elapsed_time,
                'valid': valid_prediction_elapsed_time
            },
            'feature_importance': {name: int(score) for name, score in zip(booster.feature_name(), booster.feature_importance())}
        })
        print("Finished processing {}-th bag: {}".format(i, str(train_results[-1])))

    test_data['prediction'] = sum(predictions) / len(predictions)
    old_click_to_prediction = {}
    for (click_id, prediction) in zip(test_data.click_id, test_data.prediction):
        old_click_to_prediction[click_id] = prediction

    click_ids = []
    predictions = []
    for (new_click_id, old_click_id) in zip(id_mapper.new_click_id, id_mapper.old_click_id):
        if old_click_id not in old_click_to_prediction:
            continue
        click_ids.append(new_click_id)
        predictions.append(old_click_to_prediction[old_click_id])
    submission = pd.DataFrame({'click_id': click_ids, '{}'.format(target_variable): predictions})
    submission_path = os.path.join(os.path.dirname(__file__), output_directory, os.path.basename(options.config) + '.submission.csv')
    submission.sort_values(by='click_id').to_csv(submission_path, index=False)
    dump_json_log(options, train_results)


if __name__ == "__main__":
    main()
