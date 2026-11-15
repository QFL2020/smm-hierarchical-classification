# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
from torch import nn
import torch, sys
sys.path.extend(['../../'])
from models.SimSiam.model_zoo import ResMLP3
from models.base import base_NN

class MultiSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

# Copy From: https://github.com/liyi01827/noisy-contrastive/blob/main/simsiam/model_factory.py
class SimSiam(base_NN):
    def __init__(self, cat_list, cot_length, emam, backbone_model_name: str, pretrained_model_path: str):
        super(SimSiam, self).__init__()
        self.emam = emam

        self.backbone = SimSiam.get_backbone(cat_list, cot_length, backbone_model_name, pretrained_model_path)
        self.backbone_k = SimSiam.get_backbone(cat_list, cot_length, backbone_model_name, pretrained_model_path)
        dim_out, dim_in = self.backbone.deep_layers[-1].fc2[0].weight.shape
        dim_mlp = 768
        self.backbone.fc = nn.Identity()
        self.backbone_k.fc = nn.Identity()

        # print('dim in', dim_in)
        # print('dim out', dim_out)
        self.projector = nn.Sequential(nn.Linear(dim_in, dim_mlp, dtype=torch.float64), nn.BatchNorm1d(dim_mlp, dtype=torch.float64), nn.ReLU(), nn.Linear(dim_mlp, dim_out, dtype=torch.float64))
        self.projector_k = nn.Sequential(nn.Linear(dim_in, dim_mlp, dtype=torch.float64), nn.BatchNorm1d(dim_mlp, dtype=torch.float64), nn.ReLU(),
                                       nn.Linear(dim_mlp, dim_out, dtype=torch.float64))
        self.projector.apply(self._reset_weights)
        self.projector_k.apply(self._reset_weights)

        # predictor
        self.predictor = nn.Sequential(nn.Linear(dim_out, 512, dtype=torch.float64), nn.BatchNorm1d(512, dtype=torch.float64), nn.ReLU(), nn.Linear(512, dim_out, dtype=torch.float64))
        self.predictor.apply(self._reset_weights)

        self.encoder = MultiSequential(
            self.backbone,
            self.projector
        )

        self.encoder_k = MultiSequential(
            self.backbone_k,
            self.projector_k
        )

        # Binary Classification
        self.linear = nn.Linear(dim_in, 1, dtype=torch.float64)
        self.linear.apply(self._reset_weights)
        self.probability = MultiSequential(
            self.backbone,
            self.linear,
            nn.Sigmoid()
        )

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False


    @staticmethod
    def get_backbone(cat_list, cot_length, backbone_model_name, pretrained_model_path):
        if backbone_model_name == 'ResMLP3':
            model = ResMLP3(cat_list, cot_length, pretrained_model_path)
        else:
            assert False, 'Error: wrong model name={}'.format(pretrained_model_path)
        return model
    
    def forward(self, inputs_cat_aug1, inputs_cot_aug1, inputs_cat_aug2, inputs_cot_aug2, 
                inputs_cat_weak, inputs_cot_weak):
        output = self.probability(inputs_cat_weak, inputs_cot_weak)

        z1 = self.encoder(inputs_cat_aug1, inputs_cot_aug1)
        p1 = self.predictor(z1)
        p1 = nn.functional.normalize(p1, dim=1)

        with torch.no_grad():  # no gradient to keys
            m = self.emam
            for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)

        z2 = self.encoder_k(inputs_cat_aug2, inputs_cot_aug2)
        z2 = nn.functional.normalize(z2, dim=1)
        return p1,z2,output

    def forward_test(self, inputs_cat, inputs_cot):
        return self.probability(inputs_cat, inputs_cot)
    
    def reset_weights(self):
        """ reset model weights to avoid weight leakage """
        pass
    
    @staticmethod
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