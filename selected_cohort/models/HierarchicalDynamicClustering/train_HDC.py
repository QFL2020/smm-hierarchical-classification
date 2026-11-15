# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import sys, os
import numpy as np
sys.path.extend(["../../"])
# from Hierarchical_Training.train_hierarchical_clustering import TreeNode, testing_hierarchical
from models.HierarchicalDynamicClustering.HDC_factory import dynamic_hierarchical_clustering_recursive, testing_hierarchical
from utils.misc import setup_seed
from dataset.get_dataset_from_config import get_dataset_from_config
from utils.eval import binary_classification_metrics

from settings.config import Config

def train_HDC(config: Config, only_test: bool=False):
    setup_seed(config.seed)
    DC_settings = config.Dynamic_cluster_settings
    load_previous = DC_settings['load_previous']
    save_dir = config.save_dir

    train_data, val_data, test_data, _input_dim = get_dataset_from_config(config, os.path.dirname(os.path.abspath(__file__)) + r'/../../dataset/')
    root = dynamic_hierarchical_clustering_recursive(train_data, val_data, np.arange(len(train_data)), np.arange(len(test_data)),
                                                    save_path=save_dir, node_id=0, _input_dim=_input_dim, only_test=only_test, load_previous=load_previous,
                                                    config=config)

    # calculate the results from training (testing)
    save_probs_name = 'HDC_probs'
    testing_hierarchical(root, test_data, np.arange(len(test_data)), save_probs_name)
    test_probs = test_data[save_probs_name]
    y_test = test_data['SMM']
    results = binary_classification_metrics(test_probs, y_test)
    print(results)
    
    if True:
        # selected_features..
        test_data.to_csv(save_dir + "/test_Analysis.csv")
    with open(os.path.join(config.save_dir, 'results.txt'), 'w+') as f:
        for key in results.keys():
            f.write("{}:{}".format(key, results[key]) + '\n')

    return results