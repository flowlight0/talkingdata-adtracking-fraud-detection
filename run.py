import argparse
import itertools
import json
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import pandas.testing
from joblib import Memory
from sklearn.metrics import auc, roc_curve

import features.basic
from features import Feature
from models import LightGBM, Model

memory = Memory(cachedir='.', verbose=0)

feature_map = {
    'ip': features.basic.Ip,
    'app': features.basic.App,
    'os': features.basic.Os,
    'channel': features.basic.Channel
}

models = {
    'lightgbm': LightGBM
}

output_directory = 'data/output'

target_variable = 'is_attributed'


@memory.cache
def get_click_id_(path) -> pd.Series:
    return pd.read_feather(path)[['click_id']].astype('int32')


def get_click_id(config) -> pd.Series:
    return get_click_id_(get_dataset_filename(config, 'test_full'))


@memory.cache
def get_target_(path) -> pd.Series:
    return pd.read_feather(path)[[target_variable]]


def get_target(config, dataset_type: str) -> pd.Series:
    return get_target_(get_dataset_filename(config, dataset_type))


def load_dataset(paths, index=None) -> pd.DataFrame:
    assert len(paths) > 0

    feature_datasets = []
    for path in paths:
        if index is None:
            feature_datasets.append(pd.read_feather(path))
        else:
            feature_datasets.append(pd.read_feather(path).loc(index))

    # check if all of feature dataset share the same index
    index = feature_datasets[0].index
    for feature_dataset in feature_datasets[1:]:
        pandas.testing.assert_index_equal(index, feature_dataset.index)

    return pd.concat(feature_datasets, axis=1)


def get_dataset_filename(config, dataset_type: str) -> str:
    return os.path.join(config['dataset']['input_directory'], config['dataset']['files'][dataset_type])


def load_datasets(config, index=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_dir: str = config['dataset']['cache_directory']
    feature_list: List[Feature] = [feature_map[feature](cache_dir) for feature in config['features']]
    assert len(feature_list) > 0

    train_path = get_dataset_filename(config, 'train')
    valid_path = get_dataset_filename(config, 'valid')
    test_path = get_dataset_filename(config, 'test_full')

    feature_path_lists = [feature.create_features(train_path, valid_path, test_path) for feature in feature_list]
    train_paths = [feature_path_list[0] for feature_path_list in feature_path_lists]
    valid_paths = [feature_path_list[1] for feature_path_list in feature_path_lists]
    test_paths = [feature_path_list[2] for feature_path_list in feature_path_lists]

    train = load_dataset(train_paths, index)
    valid = load_dataset(valid_paths, index)
    test = load_dataset(test_paths, index)
    test['click_id'] = get_click_id(config)
    train[target_variable] = get_target(config, 'train')
    valid[target_variable] = get_target(config, 'valid')
    return train, valid, test


def load_categorical_features(config) -> List[str]:
    return list(itertools.chain(*[feature_map[feature].categorical_features() for feature in config['features']]))


def generate_submission(config, average_prediction):
    pass


def dump_json_log(options, train_results):
    config = json.load(open(options.config))
    results = {
        'training': {
            'trials': train_results,
            'average_train_auc': np.mean([result['train_auc'] for result in train_results]),
            'average_valid_auc': np.mean([result['valid_auc'] for result in train_results]),
            'train_auc_std': np.std([result['train_auc'] for result in train_results]),
            'valid_auc_std': np.std([result['valid_auc'] for result in train_results]),
            'average_time': np.mean([result['time'] for result in train_results])
        },
        'config': config,
    }
    log_path = os.path.join(os.path.dirname(__file__), output_directory,
                            os.path.basename(options.config) + '.result.json')
    json.dump(results, open(log_path, 'w'), indent=2)


def negative_down_sampling(data, random_state):
    positive_data = data[data[target_variable] == 1]
    positive_ratio = float(len(positive_data)) / len(data)
    negative_data = data[data[target_variable] == 0].sample(
        frac=positive_ratio / (1 - positive_ratio), random_state=random_state)
    return pd.concat([positive_data, negative_data])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/lightgbm_0.json')
    options = parser.parse_args()
    config = json.load(open(options.config))
    assert config['model']['name'] in models  # check model's existence before getting datasets

    id_mapper_file = 'data/working/id_mapping.feather'
    assert os.path.exists(id_mapper_file), "Please download {} from s3 before running this script".format(id_mapper_file)
    id_mapper = pd.read_feather(id_mapper_file)

    train_data, valid_data, test_data = load_datasets(config)
    categorical_features = load_categorical_features(config)
    model: Model = models[config['model']['name']]()

    predictions = []
    train_results = []
    negative_down_sampling_config = config['dataset']['negative_down_sampling']
    predictors = train_data.columns.drop(target_variable)

    if not negative_down_sampling_config['enabled']:
        start_time = time.time()
        booster, result = model.train_and_predict(train=train_data,
                                                  valid=valid_data,
                                                  categorical_features=categorical_features,
                                                  target=target_variable,
                                                  params=config['model'])
        prediction = booster.predict(test_data[predictors])
        predictions.append(prediction)
        train_results.append({
            'train_auc': result['train']['auc'][booster.best_iteration],
            'valid_auc': result['valid']['auc'][booster.best_iteration],
            'best_iteration': booster.best_iteration,
            'time': time.time() - start_time
        })
    else:
        for i in range(negative_down_sampling_config['bagging_size']):
            start_time = time.time()
            sampled_train_data: pd.DataFrame = negative_down_sampling(train_data, random_state=i)
            sampled_valid_data: pd.DataFrame = negative_down_sampling(valid_data, random_state=i)
            booster, result = model.train_and_predict(train=sampled_train_data,
                                                      valid=sampled_valid_data,
                                                      categorical_features=categorical_features,
                                                      target=target_variable,
                                                      params=config['model'])
            prediction = booster.predict(test_data[predictors])
            prediction_valid_before_down_sampling = booster.predict(valid_data[predictors])
            valid_fpr, valid_tpr, thresholds = roc_curve(valid_data[target_variable], prediction_valid_before_down_sampling, pos_label=1)
            predictions.append(prediction)
            train_results.append({
                'train_auc': result['train']['auc'][booster.best_iteration],
                'valid_auc': result['valid']['auc'][booster.best_iteration],
                'valid_auc_before_down_sampling': auc(valid_tpr, valid_fpr),
                'best_iteration': booster.best_iteration,
                'time': time.time() - start_time
            })

    print(sum(predictions) / len(predictions))
    test_data['prediction'] = sum(predictions) / len(predictions)
    submission = id_mapper.merge(test_data[['click_id', 'prediction']], how='inner', left_on=['old_click_id'],
                                 right_on='click_id')[['new_click_id', 'prediction']]
    submission_path = os.path.join(os.path.dirname(__file__), output_directory, os.path.basename(options.config) + '.submission.csv')
    submission.rename(columns={'new_click_id': 'click_id'}).sort_values(by='click_id').to_csv(submission_path, index=False)
    generate_submission(config, test_data)
    dump_json_log(options, train_results)


if __name__ == "__main__":
    main()
