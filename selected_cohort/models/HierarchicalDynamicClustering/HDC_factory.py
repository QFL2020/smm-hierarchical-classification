# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import os, json
import numpy as np
import pandas as pd

import sys
sys.path.extend(['../../'])
from models.DynamicClustering.dynamic_cluster import DynamicCluster
from models.HierarchicalDynamicClustering.HDC_utils import HDCLogWrite, get_height

class TreeNode:
    def __init__(self, training_indices, testing_indices, node_id: int,
                 model=None, cluster_model=None, save_path: str=None, fixed_node: int=-1):
        self.training_indices = training_indices
        self.testing_indices = testing_indices
        self.node_id = node_id
        self.children = []

        # -1 => no leaf node: clustering it
        # -2 => leaf node: just testing it by SMM_model
        # 0 => defined as Non-SMM
        # 1 => defined as SMM
        self.fixed_node = fixed_node
        # SMM Detection model
        self.model = model
        self.save_model_path = None
        # Clustering model
        self.cluster_model = cluster_model
        self.save_cluster_model_path = None
        
def dynamic_hierarchical_clustering_recursive(data: pd.DataFrame, test_data: pd.DataFrame, training_indices: np.array, testing_indices: np.array,
                                            save_path: str, node_id: int, _input_dim, only_test: bool=False, load_previous: bool=False,
                                            **kwargs):
    config = kwargs['config']
    DC_settings = config.Dynamic_cluster_settings
    HDC_settings = config.HDC_settings
    
    initialize_strategy = DC_settings['initialize_strategy']
    retained_best_name = DC_settings['retained_best_name']
    only_test_load_model_name = DC_settings['only_test_load_model_name']
    load_previous = DC_settings['load_previous']
    cluster_settings = DC_settings['cluster_settings']
    classifier_settings = DC_settings['classifier_settings']

    max_layer = int(HDC_settings['max_layer'])
    min_size = int(HDC_settings['min_size'])
    # if training have only Non-SMM/SMM samples
    # then this cluster is defined as fixed category
    cur_train_data = data.loc[training_indices]
    cur_test_data = test_data.loc[testing_indices]
    training_SMM_count = len(cur_train_data.index[cur_train_data['SMM'] == 1])
    HDCLogWrite(save_path, 'node_id={}, training_size={}/{}, testing_size={}/{}'.format(node_id, sum(cur_train_data['SMM']), len(training_indices), sum(cur_test_data['SMM']), len(testing_indices)))
    if len(training_indices) - training_SMM_count <= 6:
        HDCLogWrite(save_path, 'node_id={} end: all SMM samples'.format(node_id))
        return TreeNode(training_indices, testing_indices, node_id, None, None, save_path, fixed_node=1)
    elif training_SMM_count <= 6:
        HDCLogWrite(save_path, 'node_id={} end: all Non-SMM samples'.format(node_id))
        return TreeNode(training_indices, testing_indices, node_id, None, None, save_path, fixed_node=0)
    elif len(training_indices) <= min_size:
        HDCLogWrite(save_path, 'node_id={} end: indices <= min_size'.format(node_id))
        return TreeNode(training_indices, testing_indices, node_id, None, None, save_path, fixed_node=0 if training_SMM_count / len(training_indices) < 0.5 else 1)
    # 1. dynamic clustering the training dataset
    # 2. if precision > 0.5 and recall > 0.5 => as an leaf node 
    # (save left and right nn_model, and father cluster), otherwise goto 3.
    # 3. clustering it into two sub-train/test dataset for two children

    # training the model
    new_train_data = cur_train_data.copy()

    dynamic_cluster = DynamicCluster(new_train_data, cur_test_data, save_path + f"/dynamic_clustering/node_id_{node_id}/", initialize_strategy,
                                    only_test, only_test_load_model_name, max_iter=10, 
                                    _input_dim=_input_dim, device=config.device, load_previous=load_previous, early_stopping=True,
                                    retained_best_name=retained_best_name, NN_max_iters=config.epoch, 
                                    sub_tree_cls_name=classifier_settings['model_name'], sub_tree_cls_structure=classifier_settings['model_structure'], 
                                    tree_cluster_name=cluster_settings['model_name'], tree_cluster_structure=cluster_settings['model_structure'], 
                                    lr=config.lr, train_id=0, min_size=min_size)

    if dynamic_cluster.is_leaf or (max_layer is not None and get_height(node_id) >= max_layer):
        HDCLogWrite(save_path, 'node_id={} end: leaf node or max_layer'.format(node_id))
        return TreeNode(training_indices, testing_indices, node_id, dynamic_cluster.SMM_Detector, None, save_path, fixed_node=-2)

    X_train = cur_train_data.drop('SMM', axis=1)
    X_test = cur_test_data.drop('SMM', axis=1)
    labels = dynamic_cluster.cluster_model.predict(X_train)
    # split into 2 clusters
    cluster_1 = training_indices[np.where(labels == 0)]
    cluster_2 = training_indices[np.where(labels == 1)]
    root = TreeNode(training_indices, testing_indices, node_id, dynamic_cluster.SMM_Detector, dynamic_cluster.cluster_model, save_path, fixed_node=-1)

    if len(testing_indices) == 0:
        HDCLogWrite(save_path, 'node_id={} end: len(validate)==0'.format(node_id))
        return root
    
    test_labels = dynamic_cluster.cluster_model.predict(X_test)
    # split testing data into 2 clusters
    test_cluster_1 = testing_indices[np.where(test_labels == 0)]
    test_cluster_2 = testing_indices[np.where(test_labels == 1)]

    if dynamic_cluster.is_over:
        HDCLogWrite(save_path, 'node_id={} end: is_over[perfect results]'.format(node_id))
        le_node = TreeNode(cluster_1, test_cluster_1, node_id * 2 + 1, dynamic_cluster.left_model.model, 
                            None, save_path, fixed_node=dynamic_cluster.left_model.fixed_node)
        ri_node = TreeNode(cluster_2, test_cluster_2, node_id * 2 + 2, dynamic_cluster.right_model.model, 
                            None, save_path, fixed_node=dynamic_cluster.right_model.fixed_node)
        root.children.append(le_node)
        root.children.append(ri_node)
        return root
    
    # clustering it, split it into two sub-root [3 step]
    left_child = dynamic_hierarchical_clustering_recursive(data, test_data, cluster_1, test_cluster_1,
                                            save_path, node_id * 2 + 1, _input_dim, only_test, load_previous,
                                            **kwargs)
    right_child = dynamic_hierarchical_clustering_recursive(data, test_data, cluster_2, test_cluster_2,
                                            save_path, node_id * 2 + 2, _input_dim, only_test, load_previous,
                                            **kwargs)
    root.children.append(left_child)
    root.children.append(right_child)
    
    HDCLogWrite(save_path, 'node_id={} child=le_{}, ri_{}'.format(node_id, node_id * 2 + 1, node_id * 2 + 2))
    return root


def testing_hierarchical(root: TreeNode, test_data: pd.DataFrame, testing_indices, save_probs_name: str='Hierarchical_Clustering_Probs'):
    if len(testing_indices) == 0:
        return 
    
    if save_probs_name in test_data.columns:
        X_test = test_data.loc[testing_indices].drop(['SMM', save_probs_name], axis=1)
    else:
        X_test = test_data.loc[testing_indices].drop('SMM', axis=1)

    if root.fixed_node == 0:
        test_data.loc[testing_indices, save_probs_name] = 0
        return 
    elif root.fixed_node == 1:
        test_data.loc[testing_indices, save_probs_name] = 1
        return 
    elif root.fixed_node == -2:
        y_pred = root.model.predict_proba(X_test)
        test_data.loc[testing_indices, save_probs_name] = y_pred[:, 1]
        return
    
    assert root.fixed_node == -1, "ERROR, wrong root.fixed_node={}".format(root.fixed_node)

    # split testing data into 2 clusters
    test_labels = root.cluster_model.predict(X_test)
    test_cluster_1 = testing_indices[np.where(test_labels == 0)]
    test_cluster_2 = testing_indices[np.where(test_labels == 1)]
    
    testing_hierarchical(root.children[0], test_data, test_cluster_1, save_probs_name)
    testing_hierarchical(root.children[1], test_data, test_cluster_2, save_probs_name)
