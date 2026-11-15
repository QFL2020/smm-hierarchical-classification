"""Neural runtime used by the final sensitivity analysis.
Extracted from the executed runtime; model initialization order, float64 arithmetic,
scalar embeddings, balanced sampling and optimizer settings are preserved.
Training/selection is in modified_hdc.py, not in this module.
"""
import os, random
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, Sampler
from typing import Union, List

class base_NN(nn.Module):
    def __init__(self):
        super(base_NN, self).__init__()

    def reset_weights(self, path=None):
        """ reset model weights to avoid weight leakage """
        if path:
            print('load pretrained model from:', path)
            self.load_state_dict(torch.load(path))

        def _reset_weights(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        self.apply(_reset_weights)

    def save(self, save_file: str):
        torch.save(self.state_dict(), save_file)

def get_MLP_layer(input_dim: int, layer_list: list, p: float, act=nn.ReLU(), dtype=torch.float64) -> nn.Sequential:
    current_layer = nn.Sequential()
    lst_dim = input_dim
    for current_dim in layer_list:
        current_layer.append(nn.Linear(lst_dim, current_dim, bias=True, dtype=dtype))
        current_layer.append(nn.BatchNorm1d(current_dim, dtype=dtype))
        current_layer.append(act)
        current_layer.append(nn.Dropout(p))

        lst_dim = current_dim
    return current_layer

class Embedding_layers(base_NN):
    def __init__(self, cat_length: list, cot_length: int, dtype=torch.float64):
        super(Embedding_layers, self).__init__()
        self.cat_length = cat_length
        self.cot_length = cot_length
        # nn.Embedding Embedding
        cur_sum = 0
        for i in range(len(self.cat_length)):
            cur_sum += int(np.ceil(np.log(self.cat_length[i] + 1) / np.log(2)))
        self.embedding_layers = nn.Embedding(cur_sum, 1, dtype=dtype)

        self.cat_dim = cur_sum
        self.input_dim = cur_sum + cot_length
    
    def forward(self, inputs_cat, inputs_cot):
        C = len(inputs_cat)
        extract_embedding = self.embedding_layers(inputs_cat).reshape(C, self.cat_dim)
        embeddings = torch.cat((extract_embedding, inputs_cot), dim=1)
        return embeddings

    def frozen(self):
        for param in self.parameters():
            param.requires_grad = False

class MLP_CatNumBlock(base_NN):
    def __init__(self, cat_list: list, cot_length: int, 
                 stack_layers: list, p: float=0.3, act: str='ReLu', dtype=torch.float64):
        super(MLP_CatNumBlock,self).__init__()
        self.act = nn.ReLU() if act == 'ReLu' else None

        self.cot_length = cot_length
        self.embedding_layers = Embedding_layers(cat_list, cot_length)
        self.input_dim = self.embedding_layers.input_dim
        self.layers = get_MLP_layer(self.input_dim, stack_layers, p, self.act, dtype)

    def forward(self, inputs):
        inputs_cat = inputs[:, :-self.cot_length].to(torch.int)
        inputs_cot = inputs[:, -self.cot_length:]
        embeddings = self.embedding_layers(inputs_cat, inputs_cot)
        return self.layers(embeddings)

class FeatureDataset(Dataset):
    def __init__(self, inputs, targets, device: str, dtype=torch.float):
        self._inputs = inputs
        self._targets = targets
        
        self.device = device
        self.dtype = dtype
        
        self.inputs = torch.tensor(inputs, dtype=dtype, device=device)
        self.targets = torch.tensor(targets, dtype=dtype, device=device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class BalancedSampler(Sampler):
    def __init__(self, dataset, batch_size, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        self.indices = np.arange(len(dataset))
        self.labels = dataset.targets.cpu().detach().numpy()[self.indices].copy().reshape(-1)

    def __iter__(self):
        positive_indices = np.where(self.labels == 1)[0]
        negative_indices = np.where(self.labels == 0)[0]

        all_batches = []
        negative_replace = False
        if len(negative_indices) < self.batch_size - self.batch_size // 2 - 1:
            negative_replace = True
        for _ in range((len(self.dataset) + self.batch_size - 1) // self.batch_size):
            pos_sample_count = self.batch_size // 2 + 1 # half: positive samples
            neg_sample_count = self.batch_size - pos_sample_count # half: negative samples

            pos_samples = np.random.choice(positive_indices, pos_sample_count, replace=True)
            neg_samples = np.random.choice(negative_indices, neg_sample_count, replace=negative_replace)

            batch = np.concatenate([pos_samples, neg_samples])
            np.random.shuffle(batch)
            all_batches.extend(batch)
        
        return iter(all_batches)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size * self.batch_size

def setup_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_lr_schduler(optimizer, lr_schduler_type):
    if lr_schduler_type[0] != "exp":
        raise ValueError("HDC wrapper requires explicit exponential schedule")
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_schduler_type[1])

class EarlyStoppingCallback:
    def __init__(self, patience=10, min_delta=0, verbose=False, save_best_name: bool=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        if save_best_name:
            self.best_name = None

    def __call__(self, loss, cur_name: str=None):
        score = -loss
        if self.best_score is None:
            self.best_score = score
            self.best_name = cur_name
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_name = cur_name
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping after {self.patience} epochs without improvement.")
                self.early_stop = True


class base_cls_model_training:
    def init_data(self, X, y):
        if type(X) == pd.DataFrame or type(X) == pd.Series:
            X = X.values
        if type(y) == pd.DataFrame or type(y) == pd.Series:
            y = y.values
        return (X, y)

class BalancedTraining(base_cls_model_training):
    def __init__(self, input_dim: Union[int, List[Union[List, int]]], dim_list: list, p: float=0.0, act: str='ReLu', dtype=torch.float64, learning_rate: float=0.01, batch_size: int=40960, device: str='cpu', max_iter: int=1000, selected_model_name: str='Deep_Net'):
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
        assert self.selected_model_name == 'MLP_CleanedSMM'
        self.model = MLP_CatNumBlock(self.input_dim[0], self.input_dim[1], self.dim_list, self.p, self.act, self.dtype)
        self.model.layers.append(nn.Linear(self.dim_list[-1], 1, dtype=self.dtype))
        self.model.layers.append(nn.Sigmoid())
        self.model.reset_weights()
        self.model.to(device=self.device)
    def _init_model(self, data_size=40960):
        self._init_NN()
        self.criterion = torch.nn.BCELoss()
        self.scheduler = None
        if int(self.learning_rate) == -1:
            self.optimizer = optim.Adam(self.model.parameters(), lr=min(1, data_size / self.batch_size) * 0.001, weight_decay=0.0001)
        elif int(self.learning_rate) == -2:
            self.optimizer = optim.Adam(self.model.parameters(), lr=min(1, data_size / self.batch_size) * 0.002, weight_decay=0.0001)
            self.scheduler = get_lr_schduler(self.optimizer, lr_schduler_type=['exp', 0.95])
        elif int(self.learning_rate) == -3:
            self.optimizer = optim.Adam(self.model.parameters(), lr=min(1, data_size / self.batch_size) * 0.002, weight_decay=1e-05)
            self.scheduler = get_lr_schduler(self.optimizer, lr_schduler_type=['exp', 0.95])
        elif int(self.learning_rate) == -5:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-05)
            self.scheduler = get_lr_schduler(self.optimizer, lr_schduler_type=['exp', 0.95])
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-05)
            self.scheduler = get_lr_schduler(self.optimizer, lr_schduler_type=['exp', 0.95])
        self.early_stopping = EarlyStoppingCallback(patience=30, min_delta=0, verbose=True)
    def predict_proba(self, X):
        X, _ = self.init_data(X, None)
        if len(X) == 0:
            return np.empty((0, 2), dtype=float)
        self.model.eval()
        inputs = torch.tensor(X, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            positive_probs = outputs
            negative_probs = 1 - positive_probs
            probas = torch.concatenate((negative_probs, positive_probs), dim=-1)
        return probas.detach().cpu().numpy()

def get_cluster_model(model_name, MLP_structure, max_iter, _input_dim, device, lr):
    assert model_name == "MLP_CleanedSMM"
    return BalancedTraining(_input_dim, MLP_structure, p=0.0, batch_size=4096, device=device, max_iter=max_iter, learning_rate=lr, selected_model_name=model_name)
