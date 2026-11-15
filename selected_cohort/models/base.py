# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import torch
from torch import nn
import numpy as np

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

class Deep_Net(base_NN):
    def __init__(self, input_dim, dim_list: list, p: float=0.1, act: str='ReLu', type=torch.float64):
        super(Deep_Net, self).__init__()
        self.layers = nn.ModuleList([])
        self.activate = nn.ReLU(True) if act == 'ReLu' else nn.Tanh()

        for i in range(len(dim_list)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, dim_list[i], dtype=type))
            else:
                self.layers.append(nn.Linear(dim_list[i - 1], dim_list[i], dtype=type))
            
            self.layers.append(nn.BatchNorm1d(dim_list[i], dtype=type))
            self.layers.append(self.activate)
            self.layers.append(nn.Dropout(p))

    def forward(self, x, penultimate_layer: bool=False):
        for layer in self.layers:
            y = x
            x = layer(x)
        if penultimate_layer:
            return y, x
        return x
    
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
