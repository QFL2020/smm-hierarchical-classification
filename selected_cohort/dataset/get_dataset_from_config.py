# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import sys, torch
sys.path.extend(['../'])
from settings.config import Config
from dataset.data_processing import get_cleaned_data, get_train_val_test, get_train_val_test_NNfeature, MixedDataset_transform

def get_dataset_from_config(config: Config, cur_path: str="./", dtype=torch.float64):
    cleaned_data = get_cleaned_data(cur_path, config.SMM_label, 'default')
    if config.dataset[0] == 'CleanedSMM' and config.dataset[1] in ['Binary_Encoder', 'Label_Encoder'] and config.dataset[2] == 'Standard_Scaler':
        train_dataset, val_dataset, test_dataset, cat_list = get_train_val_test(cleaned_data, config, 'default', dtype)

        if config.NN_model and (config.NN_model[0] == 'DynamicClustering' or config.NN_model[0] == 'HDC'):
            input_dim = [cat_list, train_dataset.cot_length]
            return MixedDataset_transform(train_dataset), MixedDataset_transform(val_dataset), MixedDataset_transform(test_dataset), input_dim
        return train_dataset, val_dataset, test_dataset, cat_list
    elif config.dataset[0] == 'Feature':
        train_dataset, val_dataset, test_dataset, cat_list = get_train_val_test(cleaned_data, config, 'default', dtype)
        train_dataset, val_dataset, test_dataset, input_dim = get_train_val_test_NNfeature(train_dataset, val_dataset, test_dataset, cat_list, 
                                                                                           config.Dynamic_cluster_settings['Feature_model_name'], config)
        return train_dataset, val_dataset, test_dataset, input_dim
    else:
        assert False, 'Not defined this dataset! information: [{}-{}-{}]'.foramt(config.dataset[0], config.dataset[1], config.dataset[2])

    return None
