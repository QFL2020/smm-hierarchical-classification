# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.

import sys, os
import pandas as pd
import numpy as np
import argparse
import json
import torch
sys.path.extend(['../../'])
from models.DynamicClustering.model_training_zoo import get_cluster_model

class SubTreeClassifier:
    def __init__(self, MLP_structure: list=[64, 128, 64], max_iter: int=100, fixed_node: int=-2, 
                 _input_dim: int=512, device: str='cuda:2', **kwargs):
        self.save_dir = kwargs['save_dir']
        self.device = device
        self.MLP_structure = MLP_structure
        self.max_iter = max_iter
        self.model = get_cluster_model(kwargs['sub_tree_cls_name'], MLP_structure, max_iter, _input_dim, device, lr=kwargs['lr'])
        self.fixed_node = fixed_node
    
    def training_model(self, train_data: pd.DataFrame, save_fig_name: str='loss', 
                       test_data=None, min_size=3000, **kwargs):
        SMM_count = len(train_data.index[train_data['SMM'] == 1])
        if SMM_count < 10 or len(train_data) - SMM_count < 10:
            if SMM_count < 10:
                self.fixed_node = 0
            else:
                self.fixed_node = 1
            return 
        elif len(train_data) < min_size:
            if SMM_count <= len(train_data) // 2:
                self.fixed_node = 0
            else:
                self.fixed_node = 1
        else:
            self.fixed_node = -2
        
        X_train = train_data.drop('SMM',axis=1)
        y_train = train_data['SMM']
        if test_data is not None:
            X_test = test_data.drop('SMM', axis=1)
            y_test = test_data['SMM']
        if os.path.exists(kwargs['save_path']):
            self.load(kwargs['save_path'])
        else:
            self.model.fit(X_train, y_train, save_dir=self.save_dir, save_fig_name=save_fig_name,
                           test_X=X_test, test_y=y_test)
            self.save(kwargs['save_path'])

    def predict_proba(self, X_train: pd.DataFrame):
        k1 = np.zeros(shape=(1, len(X_train)))
        k2 = np.ones(shape=(1, len(X_train)))
        if self.fixed_node == 0:
            return np.concatenate((k2, k1)).T
        elif self.fixed_node == 1:
            return np.concatenate((k1, k2)).T
        return self.model.predict_proba(X_train)
    
    def to(self, device: str):
        self.model.to(device=device)

    def load(self, model_path):
        model_state = torch.load(model_path)
        self.MLP_structure = model_state['MLP_structure']
        self.max_iter = model_state['max_iter']
        self.model.model.load_state_dict(model_state['model'])
        self.fixed_node = model_state['fixed_node']
        self.model.to(self.device)

    def save(self, model_path):
        model_state = {
            'model': self.model.model.state_dict(),
            'MLP_structure': self.MLP_structure,
            'max_iter': self.max_iter,
            'fixed_node': self.fixed_node,
        }
        torch.save(model_state, model_path)