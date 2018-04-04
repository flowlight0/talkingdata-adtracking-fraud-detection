# talkingdata-adtracking-fraud-detection
My solution for TalkingData AdTracking Fraud Detection Challenge (https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/)

### Setup
* First of all, convert csv format files into feather format ones to accelerate later processes by running `python scripts/convert_csv_to_feather.py`. 
* Split train dataset into new train/valid datasets before start ML process
  * Run `python scripts/split_dataset.py --train data/input/train.feather.small --train_output data/input/train_0.feather.small --validation_output data/input/valid_0.feather.small`. 
  * Run `python scripts/split_dataset.py --train data/input/train.feather --train_output data/input/train_0.feather --validation_output data/input/valid_0.feather`. 
* Create mapping from full test data click ids to submission test data click ids by running `python scripts/click_id_mapper.py`. 
