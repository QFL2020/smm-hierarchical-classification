# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
"""
Code borrowed from
https://github.com/HeartbreakSurvivor/RsAlgorithms/blob/main/,
https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/layers/interaction.py;
deep cross network: https://arxiv.org/abs/1708.05123
"""
import sys, torch
import torch.nn as nn
sys.path.extend(['../'])
from models.base import Embedding_layers, base_NN

# DCN version 1
class Cross(nn.Module):
    """
    the operation is this module is x_0 * x_l^T * w_l + x_l + b_l for each layer, and x_0 is the init input
    """
    def __init__(self, input_dim, num_cross_layers, dtype=torch.float64, p: float=0.3):
        super(Cross, self).__init__()
        
        self.num_cross_layers = num_cross_layers
        weight_w = []
        weight_b = []
        batchnorm = []
        for i in range(num_cross_layers):
            weight_w.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_dim, dtype=dtype))))
            weight_b.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_dim, dtype=dtype))))
            batchnorm.append(nn.BatchNorm1d(input_dim, affine=False, dtype=dtype))
            batchnorm.append(nn.Dropout(p))


        self.weight_w = nn.ParameterList(weight_w)
        self.weight_b = nn.ParameterList(weight_b)
        self.bn = nn.ModuleList(batchnorm)

    def forward(self, x):
        out = x
        x = x.reshape(x.shape[0], -1, 1)
        for i in range(self.num_cross_layers):
            # this compute mode is time-consuming.
            # out = torch.matmul(torch.bmm(x, torch.transpose(out.reshape(out.shape[0], -1, 1), 1, 2)), self.weight_w[i]) + self.weight_b[i] + out
            # use this compute mode to speed up calculation
            xxTw = torch.matmul(x, torch.matmul(torch.transpose(out.reshape(out.shape[0], -1, 1), 1, 2), self.weight_w[i].reshape(1, -1, 1)))
            xxTw = xxTw.reshape(xxTw.shape[0], -1)
            out = xxTw + self.weight_b[i] + out

            out = self.bn[i](out)
        return out

# DCN version 2
# https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/layers/interaction.py
class CrossMetrix(nn.Module):
    def __init__(self, input_dim, num_cross_layers, dtype=torch.float64, p: float=0.3):
        super(CrossMetrix, self).__init__()
        print('CrossMetrix')
        self.num_cross_layers = num_cross_layers
        self.cross_metrix_layers = nn.ModuleList()
        self.input_dim = input_dim
        for i in range(num_cross_layers):
            self.cross_metrix_layers.append(nn.Linear(input_dim, input_dim, dtype=dtype))
            self.cross_metrix_layers.append(nn.BatchNorm1d(input_dim, affine=False, dtype=dtype))
            self.cross_metrix_layers.append(nn.Dropout(p))

    def forward(self, x):
        x_0 = x
        x_l = x_0
        for i in range(self.num_cross_layers):
            dot_ = self.cross_metrix_layers[i](x_l)
            x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
        return x_l


class Deep(nn.Module):
    def __init__(self, input_dim, deep_layers, dtype=torch.float64, p: float=0.3):
        super(Deep, self).__init__()

        deep_layers.insert(0, input_dim)
        deep_layer_list = []
        for layer in list(zip(deep_layers[:-1], deep_layers[1:])):
            deep_layer_list.append(nn.Linear(layer[0], layer[1], dtype=dtype))
            deep_layer_list.append(nn.BatchNorm1d(layer[1], affine=False, dtype=dtype))
            deep_layer_list.append(nn.ReLU(inplace=True))
            deep_layer_list.append(nn.Dropout(p))
        self._deep = nn.Sequential(*deep_layer_list)

    def forward(self, x):
        out = self._deep(x)
        return out

class DeepCross(base_NN):
    def __init__(self, cat_length: list, cot_length: int, 
                 Deep_NN: list, Cross_NN_layers: int, p: float=0.3, cross_type: str='vec'):
        super(DeepCross, self).__init__()
        
        self.embedding_layers = Embedding_layers(cat_length, cot_length)
        self._input_dim = self.embedding_layers.input_dim

        self._deepNet = Deep(self._input_dim, Deep_NN, p=p)
        if cross_type == 'vec':
            self._crossNet = Cross(self._input_dim, Cross_NN_layers, p=p)
        elif cross_type == 'mtx':
            self._crossNet = CrossMetrix(self._input_dim, Cross_NN_layers, p=p)

        self._final_dim = self._input_dim + Deep_NN[-1]
        self._final_linear = nn.Linear(self._final_dim, 1, dtype=torch.float64)

    def forward(self, inputs_cat, inputs_cot):
        embedding = self.embedding_layers(inputs_cat, inputs_cot)
        deep_out = self._deepNet(embedding)
        cross_out = self._crossNet(embedding)

        final_input = torch.cat([deep_out, cross_out], dim=1)
        output = self._final_linear(final_input)
        output = torch.sigmoid(output)
        return output
