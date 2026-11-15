# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import sys, os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

sys.path.extend(['../../'])
from settings.config import Config
from models.Traditional_ML.ML_baselines import ISO_Forest, SVM, Logistic_regression, Decision_Tree, LASSO, AdaBoost, SMOTE_AdaBoost, Random_Forest, XGBoost

from dataset.data_processing import MixedDataset
from dataset.get_dataset_from_config import get_dataset_from_config
from utils.misc import setup_seed
from utils.eval import binary_classification_metrics

def get_results(method_name: str, train_dataset: MixedDataset, val_dataset: MixedDataset, test_dataset: MixedDataset, config: Config, **args):
    if method_name == 'ISO':
        # Isolate 
        y_pred = ISO_Forest(train_dataset, val_dataset, test_dataset, config.seed, **args)
    elif method_name == 'SVM':
        # SVM
        y_pred = SVM(train_dataset, val_dataset, test_dataset)
    elif method_name == 'LR':
        # Logistic_regression
        y_pred = Logistic_regression(train_dataset, val_dataset, test_dataset, **args)
    elif method_name == 'DT':
        y_pred = Decision_Tree(train_dataset, val_dataset, test_dataset, **args)
    elif method_name == 'LASSO':
        y_pred = LASSO(train_dataset, val_dataset, test_dataset)
    elif method_name == 'AdaBoost':
        y_pred = AdaBoost(train_dataset, val_dataset, test_dataset, **args)
    elif method_name == 'SMOTEBoost':
        y_pred = SMOTE_AdaBoost(train_dataset, val_dataset, test_dataset)
    elif method_name == 'RandomForest':
        y_pred = Random_Forest(train_dataset, val_dataset, test_dataset, **args)
    elif method_name == 'XGBoost':
        y_pred = XGBoost(train_dataset, val_dataset, test_dataset)
    else:
        assert False, "Not Found method name: {}".format(method_name)

    return y_pred

def train_Traditioanl_ML(method_name: str, only_eval: bool, config: Config):
    setup_seed(config.seed)
    train_dataset, val_dataset, test_dataset, cat_list = get_dataset_from_config(config, cur_path=os.path.dirname(os.path.abspath(__file__)) + '/../../dataset/')

    if only_eval:
        config.Traditional_ML_settings['return_model'] = True
    y_pred = get_results(method_name, train_dataset, val_dataset, test_dataset, config, **config.Traditional_ML_settings)
    if len(y_pred) == 2:
        y_pred, model = y_pred
        train_y_pred = model.predict(train_dataset._inputs)
        train_metrics = binary_classification_metrics(train_y_pred, train_dataset._targets)

        # save training&testing pred results?
        # save_train_path = f'{config.save_dir}/{config.seed}-{config.training_strategy}-train_pred.csv'
        # save_test_path = f'{config.save_dir}/{config.seed}-{config.training_strategy}-test_pred.csv'
        # save_train_data = pd.DataFrame()
        # save_test_data = pd.DataFrame()
        # save_train_data[f'{config.seed}-train_pred'] = train_y_pred
        # save_test_data[f'{config.seed}-test_pred'] = y_pred
        # save_train_data.to_csv(save_train_path)
        # save_test_data.to_csv(save_test_path)
    else:
        train_metrics = None

    metrics = binary_classification_metrics(y_pred, test_dataset._targets)

    print('pred-unique: ', np.unique(y_pred))
    print('pred-sum: ', np.sum(y_pred))
    print('testing metrics:')
    print(metrics)
    print('training metrics')
    print(train_metrics)
    with open(config.save_dir + '/testing-training-results.txt', 'w+') as f:
        f.write('testing\n')
        for key in metrics.keys():
            f.write('{}: {}\n'.format(key, metrics[key]))
        
        if train_metrics is not None:
            f.write('training\n')
            for key in train_metrics.keys():
                f.write('{}: {}\n'.format(key, train_metrics[key]))

    return metrics
