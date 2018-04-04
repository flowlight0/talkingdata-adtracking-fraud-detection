import argparse

import pandas as pd


# Run the following command when you want to prepare small dataset for training
# python scripts/split_dataset.py \
#     --train data/input/train.feather.small \
#     --train_output data/input/train_0.feather.small  \
#     --validation_output data/input/valid_0.feather.small \
#     --valid_ratio 0.1

# Run the following command when you want to prepare dataset for training
# python scripts/split_dataset.py \
#     --train data/input/train.feather \
#     --train_output data/input/train_0.feather  \
#     --validation_output data/input/valid_0.feather \
#     --valid_ratio 0.1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/input/train.feather.small")
    parser.add_argument("--train_output", default="data/input/train_0.feather.small")
    parser.add_argument("--validation_output", default="data/input/train_0.feather.small")
    parser.add_argument("--valid_ratio", default=0.1, type=float)
    options = parser.parse_args()
    df_train = pd.read_feather(options.train)
    train_size = int(len(df_train) * (1 - options.valid_ratio))
    df_train[:train_size].reset_index(drop=True).to_feather(options.train_output)
    df_train[train_size:].reset_index(drop=True).to_feather(options.validation_output)


if __name__ == '__main__':
    main()
