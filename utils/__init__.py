import numpy as np
import json
import os


def dump_json_log(options, train_results, output_directory):
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
    log_path = os.path.join(os.path.dirname(__file__), '../', output_directory,
                            os.path.basename(options.config) + '.result.json')
    json.dump(results, open(log_path, 'w'), indent=2)