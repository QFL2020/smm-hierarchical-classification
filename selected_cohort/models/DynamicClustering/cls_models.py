# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import sys, torch
import torch.nn as nn
sys.path.extend(['../../'])

from models.Residual_factory.residual_block import get_layer
from models.base import base_NN, Embedding_layers, Deep_Net, get_MLP_layer

class ResidualMLPBlock(base_NN):
    def __init__(self, input_dim: int, stack_layers: list, p: float=0.3, act: str='ReLu', dtype=torch.float64):
        super(ResidualMLPBlock,self).__init__()
        self.act = nn.ReLU() if act == 'ReLu' else None

        self.input_dim = input_dim
        self.layers = get_layer(self.input_dim, stack_layers, p, self.act, dtype)

    def forward(self, inputs):
        return self.layers(inputs)

class ResidualMLP_CatNumBlock(base_NN):
    def __init__(self, cat_list: list, cot_length: int, 
                 stack_layers: list, p: float=0.3, act: str='ReLu', dtype=torch.float64):
        super(ResidualMLP_CatNumBlock,self).__init__()
        self.act = nn.ReLU() if act == 'ReLu' else None

        self.cot_length = cot_length
        self.embedding_layers = Embedding_layers(cat_list, cot_length)
        self.input_dim = self.embedding_layers.input_dim
        self.layers = get_layer(self.input_dim, stack_layers, p, self.act, dtype)

    def forward(self, inputs):
        inputs_cat = inputs[:, :-self.cot_length].to(torch.int)
        inputs_cot = inputs[:, -self.cot_length:]
        embeddings = self.embedding_layers(inputs_cat, inputs_cot)
        return self.layers(embeddings)

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
