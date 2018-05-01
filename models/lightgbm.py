import copy
from typing import List, Tuple

import lightgbm as lgb
import pandas as pd
from lightgbm import Booster

from models.base import Model


class LightGBM(Model):
    def train_and_predict(self, train, valid, weight, categorical_features: List[str], target: str, params: dict) \
            -> Tuple[Booster, dict]:
        if type(train) != pd.DataFrame or type(valid) != pd.DataFrame:
            raise ValueError('Parameter train and valid must be pandas.DataFrame')

        if list(train.columns) != list(valid.columns):
            raise ValueError('Train and valid must have a same column list')

        predictors = train.columns.drop(target)
        if weight is None:
            d_train = lgb.Dataset(train[predictors], label=train[target].values)
        else:
            print(weight)
            d_train = lgb.Dataset(train[predictors], label=train[target].values, weight=weight)
        d_valid = lgb.Dataset(valid[predictors], label=valid[target].values)

        eval_results = {}
        model: Booster = lgb.train(params['model_params'],
                                   d_train,
                                   categorical_feature=categorical_features,
                                   valid_sets=[d_train, d_valid],
                                   valid_names=['train', 'valid'],
                                   evals_result=eval_results,
                                   **params['train_params'])
        return model, eval_results

    def train_without_validation(self, train, weight, categorical_features: List[str], target: str, params: dict, best_iteration: int):
        predictors = train.columns.drop(target)
        if weight is None:
            d_train = lgb.Dataset(train[predictors], label=train[target].values)
        else:
            d_train = lgb.Dataset(train[predictors], label=train[target].values, weight=weight)
        train_params = copy.deepcopy(params['train_params'])
        train_params['num_boost_round'] = best_iteration
        if 'early_stopping_rounds' in train_params:
            del train_params['early_stopping_rounds']
        model = lgb.train(params['model_params'],
                          d_train,
                          categorical_feature = categorical_features,
                          **train_params)
        return model

