# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import torch
import sys
sys.path.extend(['../../'])
from models.Residual_factory.residual_block import ResidualNN

def ResMLP3(cat_list, cot_length, NN_model_pretrained_model_path: str):
    model = ResidualNN(cat_list, cot_length, [192,256], 'ReLu', 0.0)
    pretrained_model_path = NN_model_pretrained_model_path + '/ResMLP3.pth'
    model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
    return model
