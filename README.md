# talkingdata-adtracking-fraud-detection
flowlight side of 1st place solution for TalkingData AdTracking Fraud Detection Challenge (https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/)

Disclaimer: I did not care quality/readability of code in this repository, and the following documentation may contain errata.  

### Setup
Since this repository assumes all the experiments are conducted on a Docker container on an AWS docker host created by docker-machine, you need to create a docker host and image before experiments.
* **(Caution: If you run the command in this section, it will automatically create a Docker host in an AWS EC2 spot instance. if you don't want to create it, please skip this section and create your own environment by your self)** 
You can create a docker host by running a script `docker_generate.sh` under `docker/` directory (If you haven't installed docker-machine in your computer, please download before running this script). This scripts also create a docker image `kaggle/flowlight`, which contains some extra libraries in addition to `kaggle/python`.
  * `cd ./docker` 
  * `AWS_VPC_ID=<your own AWS VPC id> ./docker_generate.sh` 
* You are expected to start running a container with this image by running the following commands:
  * `eval $(docker-machine env talkingdata)` 
  * `docker run -it kaggle/flowlight bash`
* After logging into a container, you need to download datasets from Kaggle (If you wonder how you can get these files in a remote server, it's a good opportunity to learn [kaggle-api](https://github.com/Kaggle/kaggle-api])).
  * train.csv
  * test.csv
  * test_supplement.csv
* Then, you have to convert csv format files into feather format ones to accelerate later processes by running `python scripts/convert_csv_to_feather.py`.
During the competition, all the datasets and features are stored in feather files because reading feather files are much faster than csv files. 
* Next, you have to create mapping from full test data (test_supplement.csv) click ids to submission test data (test.csv) click ids by running `python scripts/click_id_mapper.py`. 
* Last, you are expected to build cpp programs used in feature generation:
  * cd ./cpp
  * make -j
### Experiment Configuration
In this repository, all the experiments are configured with a JSON configuration file. 
Please see JSON files under `configs` directory. 
  
```javascript  
{
  "features": [
    "app",
    "hour",
    "count",
    "duplicated_row_index_diff",
    "future_click_count_10",
    "future_click_count_80",
    "next_click_time_delta",
    "prev_click_time_delta",
    "all_click_count",
    "average_attributed_ratio",
    "komaki_lda_10_ip",
    "komaki_lda_20_no_device_ip",
    "komaki_lda_20_no_device_os",
    "komaki_lda_20_no_device_channel",
    "komaki_lda_20_no_device_app",
    "komaki_lda_5_no_device",
    "komaki_nmf_5_no_device",
    "komaki_pca_5_no_device"
  ],
  "model": {
    "name": "lightgbm",
    "model_params": {
      "boosting_type": "gbdt",
      "objective": "binary",
      "metric": "auc",
      "learning_rate": 0.01,
      "num_leaves": 255,
      "max_depth": 8,
      "min_child_samples": 200,
      "subsample": 0.9,
      "subsample_freq": 1,
      "colsample_bytree": 0.5,
      "min_child_weight": 0,
      "subsample_for_bin": 1000000,
      "min_split_gain": 0,
      "reg_lambda": 0,
      "verbose": 0
    },
    "train_params": {
      "num_boost_round": 5000,
      "early_stopping_rounds": 30
    }
  },
  "dataset": {
    "input_directory": "data/input/",
    "cache_directory": "data/working/",
    "files": {
      "train": "train.feather",
      "test": "old_test.feather"
    },
    "negative_down_sampling": {
      "enabled": true,
      "bagging_size": 5
    }
  },
  "rank_average": false,
  "test_hours": {
    "filter_validation": true,
    "higher_train_weight": false
  },
  "note": "100 with min_child_samples = 200"
}
```
* features: You can specify a list of features to be used with this field. Each field and an actual feature generator are tied in `run.py`.
* model: A model name and model definition/training parameters can be specified. Actually, this repository only supports LightGBM (There are lot of hard-coded logic for LightGBM).
* dataset: 
  * input_directory: Our script reads training/test files from this directory.
  * cache_directory: All the created features are cached under this directory.
  * files: You can specify a name of train/test files here. Our usage for this field was to first run an experiment with very small datasets and use full datasets only after features were successfully created from small datasets.
  * negative_down_sampling:
     * enabled: I don't know if you can run experiment when you set this parameter to false. It means you should always set this parameter to true.
     * bagging_size: this parameter specifies the number of datasets created by negative down-sampling. Our script generates each down-sampled dataset with a different random seed. 
* rank_average:
  * This field specifies a way of creating final prediction from predictions for multiple down-sampled datasets. If rank_average = true, Rank averaging (see [KAGGLE ENSEMBLING GUIDE](https://mlwave.com/kaggle-ensembling-guide/)) is used, otherwise the average of prediction values is used.
* test_hours:
  * filter_validation: If filter_validation = true, we filter out validation data based on its click hour because test dataset for submission contains an only small time-range of one day.  
### Experiment Execution
You just need to run a command `python run.py --config <configuration file>`. It will create two files `data/output/<configuration file>.result.json` and `data/output/<configuration file>.submission.csv`. The former file contains statistics of this experiment and the latter file contains prediction for a specified test dataset. Note that you can skip prediction on a test dataset by specifying `--train_only` option.   
