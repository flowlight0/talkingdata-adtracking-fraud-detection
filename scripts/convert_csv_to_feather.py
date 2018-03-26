import os

import pandas as pd

data_dir = os.path.join(os.path.dirname(__file__), '../data/input')
train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
test_columns = ['click_id', 'ip', 'app', 'device', 'os', 'channel', 'click_time']
dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'click_time': 'str',
    'attributed_time': 'str',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

train_parse_dates = ['click_time', 'attributed_time']
test_parse_dates = ['click_time']

df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'), dtype=dtypes, usecols=train_columns, parse_dates=train_parse_dates)
df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'), dtype=dtypes, usecols=test_columns, parse_dates=test_parse_dates)
df_old_test = pd.read_csv(os.path.join(data_dir, 'old_test.csv'), dtype=dtypes, usecols=test_columns, parse_dates=test_parse_dates)

df_train.to_feather(os.path.join(data_dir, 'train.feather'))
df_test.to_feather(os.path.join(data_dir, 'test.feather'))
df_old_test.to_feather(os.path.join(data_dir, 'old_test.feather'))

fraction = 0.001
random_state = 114514
df_train.sample(frac=fraction, random_state=random_state).reset_index(drop=True).to_feather(os.path.join(data_dir, 'train.feather.small'))
df_test.sample(frac=fraction, random_state=random_state).reset_index(drop=True).to_feather(os.path.join(data_dir, 'test.feather.small'))
df_old_test.sample(frac=fraction, random_state=random_state).reset_index(drop=True).to_feather(os.path.join(data_dir, 'old_test.feather.small'))
