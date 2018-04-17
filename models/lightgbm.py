from typing import List, Tuple

import lightgbm as lgb
import pandas as pd
from lightgbm import Booster

from models.base import Model


class LightGBM(Model):
    def train_and_predict(self, train, valid, categorical_features: List[str], target: str, params: dict) \
            -> Tuple[Booster, dict]:
        if type(train) != pd.DataFrame or type(valid) != pd.DataFrame:
            raise ValueError('Parameter train and valid must be pandas.DataFrame')

        if list(train.columns) != list(valid.columns):
            raise ValueError('Train and valid must have a same column list')

        predictors = train.columns.drop(target)
        d_train = lgb.Dataset(train[predictors], label=train[target].values)
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
