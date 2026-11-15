# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import sys, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from typing import Union, List

sys.path.extend(['../../'])
from utils.early_stopping import EarlyStoppingCallback
from utils.training_utils import get_lr_schduler
from dataset.sampler import BalancedSampler
from utils.eval import binary_classification_metrics
from utils.figure_plot import plot_loss, plot_score

from models.DynamicClustering.cls_models import ResidualMLPBlock, ResidualMLP_CatNumBlock, MLP_CatNumBlock
from models.base import Deep_Net

from dataset.data_processing import FeatureDataset
from dataset.selected_features import get_SMM_features

class base_cls_model_training:
    def __init__(self):
        pass

    def init_data(self, X, y):
        if type(X) == pd.DataFrame or type(X) == pd.Series:
            X = X.values
        if type(y) == pd.DataFrame or type(y) == pd.Series:
            y = y.values
        return X, y
    
    def load(self, save_path: str, device: str=None):
        self.model = torch.load(save_path)
        if device is not None:
            self.model.to(device=device)

    def save(self, save_path: str):
        torch.save(self.model, save_path)

    def to(self, device: str):
        self.device = device
        self.model.to(device=device)

# Feature Balanced Training
class BalancedTraining(base_cls_model_training):
    def __init__(self, input_dim: Union[int, List[Union[List, int]]], dim_list: list, p: float = 0.0, act: str = 'ReLu', dtype=torch.float64, 
                learning_rate: float = 1e-2, batch_size: int = 40960, device: str='cpu', max_iter: int=1000,
                selected_model_name: str='Deep_Net'):
        self.selected_model_name = selected_model_name
        self.input_dim = input_dim
        self.dim_list = dim_list
        self.p = p
        self.act = act
        self.dtype = dtype
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.max_iter = max_iter
        
        self._init_NN()
        
    def _init_NN(self):
        if self.selected_model_name == 'DeepNet':
            self.model = Deep_Net(self.input_dim, self.dim_list, self.p, self.act, self.dtype)
        elif self.selected_model_name == 'ResidualMLP':
            self.model = ResidualMLPBlock(self.input_dim, self.dim_list, self.p, self.act, self.dtype)
        elif self.selected_model_name == 'MLP_CleanedSMM':
            self.model = MLP_CatNumBlock(self.input_dim[0], self.input_dim[1], self.dim_list, self.p, self.act, self.dtype)
        elif self.selected_model_name == 'ResidualMLP_CleanedSMM':
            self.model = ResidualMLP_CatNumBlock(self.input_dim[0], self.input_dim[1], self.dim_list, self.p, self.act, self.dtype)
        else:
            assert False, f"ERROR: wrong selected_model_name={self.selected_model_name}"
        
        self.model.layers.append(nn.Linear(self.dim_list[-1], 1, dtype=self.dtype))
        self.model.layers.append(nn.Sigmoid())
        self.model.reset_weights()
        self.model.to(device=self.device)

    def _init_model(self, data_size=40960):
        # Initialize model, loss function, and optimizer
        self._init_NN()
        self.criterion = torch.nn.BCELoss() # For classification tasks

        self.scheduler = None
        if int(self.learning_rate) == -1:
            # constant lr for diff data size
            self.optimizer = optim.Adam(self.model.parameters(), lr=min(1, data_size / self.batch_size) * 1e-3, weight_decay=0.0001)
        elif int(self.learning_rate) == -2:
            # constant lr for diff data size
            self.optimizer = optim.Adam(self.model.parameters(), lr=min(1, data_size / self.batch_size) * 2e-3, weight_decay=0.0001)
            self.scheduler = get_lr_schduler(self.optimizer, lr_schduler_type=["exp",0.95])
        elif int(self.learning_rate) == -3:
            # constant lr for diff data size
            self.optimizer = optim.Adam(self.model.parameters(), lr=min(1, data_size / self.batch_size) * 2e-3, weight_decay=1e-5)
            self.scheduler = get_lr_schduler(self.optimizer, lr_schduler_type=["exp",0.95])
        elif int(self.learning_rate) == -5:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
            self.scheduler = get_lr_schduler(self.optimizer, lr_schduler_type=["exp",0.95])
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            self.scheduler = get_lr_schduler(self.optimizer, lr_schduler_type=["exp",0.95])

        self.early_stopping = EarlyStoppingCallback(patience=30, min_delta=0, verbose=True)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, save_dir: str='./', save_fig_name: str='loss_plot', 
            test_X=None, test_y=None, **kwargs):
        X, y = self.init_data(X, y)
        # previous: data size = self.batch_size
        self._init_model(data_size=len(X))

        if test_X is None or test_y is None:
            # Split the data into training and validation sets (90% train, 10% val)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        else:
            test_X, test_y = self.init_data(test_X, test_y)
            X_train, X_val, y_train, y_val = [X, test_X, y, test_y]

        train_dataset = FeatureDataset(X_train, y_train.reshape(-1, 1), device=self.device, dtype=self.dtype)
        val_dataset = FeatureDataset(X_val, y_val.reshape(-1, 1), device=self.device, dtype=self.dtype)
        sampler = BalancedSampler(train_dataset, self.batch_size)
        train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        train_loss_list = []
        val_loss_list = []
        F1_score_list = []
        # Training loop
        training_loop = tqdm(range(self.max_iter), desc='epoch')
        for epoch in training_loop:
            self.model.train()  # Set the model to training mode
            train_loss = 0.0
            for (batch_inputs, batch_labels) in train_loader:
                # Zero gradients, forward pass, compute loss, backward pass, and optimize
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            if self.scheduler:
                self.scheduler.step()

            # Validation phase
            self.model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            val_outputs = []
            with torch.no_grad():
                for (batch_inputs, batch_labels) in val_loader:
                    outputs = self.model(batch_inputs)
                    val_outputs.extend(outputs.detach().cpu().numpy())
                    loss = self.criterion(outputs, batch_labels)
                    val_loss += loss.item()
            val_outputs = np.array(val_outputs)
            current_results = binary_classification_metrics(val_outputs, y_val)

            # Average loss for the epoch
            val_loss /= len(val_loader)
            train_loss /= len(train_loader)
            
            if self.scheduler:
                cur_lr = self.scheduler.get_last_lr()[0]
            else:
                cur_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            training_loop.set_postfix(lr=cur_lr, train_loss=train_loss, val_loss=val_loss,
                                  Precision=current_results['Precision'], Recall=current_results['Recall'], F1_score=current_results['F1-score'])
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            F1_score_list.append(current_results['F1-score'])
            # Early stopping check
            self.early_stopping(-current_results['F1-score'], cur_name=str(epoch))
            
            if self.early_stopping.early_stop:
                print(f"Stopping early at epoch {epoch + 1} with loss {val_loss:.4f}, F1-score {current_results['F1-score']:.4f}")
                break
        # just load the final model

        # load the best model and delete it?
        # save train/val loss for each epoch
        plot_loss(train_loss_list, val_loss_list, save_dir, save_fig_name)
        plot_score(F1_score_list, save_dir, save_fig_name + '-F1_score')
        with open(save_dir + f'/{save_fig_name}-results.txt', 'w+') as f:
            f.write(str(current_results) + '\n')

    def predict(self, X):
        X, _ = self.init_data(X, None)
        self.model.eval()  # Set the model to evaluation mode

        # Convert input to tensor
        inputs = torch.tensor(X, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            outputs = self.model(inputs).squeeze()
            predicted = torch.zeros(len(outputs))
            predicted[outputs >= 0.5] = 1
        return predicted.detach().cpu().numpy()
    
    def predict_proba(self, X):
        X, _ = self.init_data(X, None)
        self.model.eval()  # Set the model to evaluation mode

        # Convert input to tensor
        inputs = torch.tensor(X, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            positive_probs = outputs
            negative_probs = 1 - positive_probs

            probas = torch.concatenate((negative_probs, positive_probs), dim=-1)
        return probas.detach().cpu().numpy()

def get_cluster_model(
    model_name: str,
    MLP_structure: list = [64, 128, 64],
    max_iter: int = 100,
    _input_dim: Union[int, List[Union[List, int]]] = None,
    device: str = None,
    lr: float = 1e-2
):
    """
    Create and return a specified clustering model based on the model name.

    Args:
        model_name (str): Name of the model, supports 'DeepNet' or 'Residual'.
        MLP_structure (list): Structure of the MLP network, defaults to [64, 128, 64].
        max_iter (int): Maximum number of iterations, defaults to 100.
        _input_dim (int or [list, int]): Input dimension of the model. Must be provided.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        lr (float): Learning rate, defaults to 1e-2.

    Returns:
        model: An instance of the specified clustering model.

    Raises:
        AssertionError: If an invalid model name is provided.
    """
    if model_name == 'MLP':
        model = BalancedTraining(
            _input_dim, MLP_structure, p=0.0, batch_size=4096, device=device,
            max_iter=max_iter, learning_rate=lr, selected_model_name='DeepNet'
        )
    elif model_name == 'Residual':
        model = BalancedTraining(
            _input_dim, MLP_structure, p=0.0, batch_size=4096, device=device,
            max_iter=max_iter, learning_rate=lr, selected_model_name='ResidualMLP'
        )
    elif model_name == 'MLP_CleanedSMM':
        model = BalancedTraining(
            _input_dim, MLP_structure, p=0.0, batch_size=4096, device=device,
            max_iter=max_iter, learning_rate=lr, selected_model_name='MLP_CleanedSMM'
        )
    elif model_name == 'Residual_CleanedSMM':
        model = BalancedTraining(
            _input_dim, MLP_structure, p=0.0, batch_size=4096, device=device,
            max_iter=max_iter, learning_rate=lr, selected_model_name='ResidualMLP_CleanedSMM'
        )
    else:
        raise ValueError(
            f"Error: Invalid model name '{model_name}'. "
            "Must be one of ['DeepNet', 'Residual']."
        )

    return model