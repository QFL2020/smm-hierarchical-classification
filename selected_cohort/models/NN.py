# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import sys, torch
import torch.nn as nn
sys.path.extend(["../"])

from models.base import base_NN, Embedding_layers, Deep_Net

class Multi_MLP_Cat_Num(base_NN):
    def __init__(self, cat_length: list, cot_length: int, 
                 Stacking_NN: list, MLP_NN: list, p: float=0.3):
        super(Multi_MLP_Cat_Num, self).__init__()
        self.cat_length = cat_length
        self.cot_length = cot_length

        self.embedding_layers = Embedding_layers(cat_length, cot_length)
        self.input_dim = self.embedding_layers.input_dim
        self.stacking_layers = Deep_Net(self.input_dim, Stacking_NN, p)
        self.MLP_layers = Deep_Net(Stacking_NN[-1], MLP_NN, p)

        self.fc = nn.Sequential(
            nn.Linear(MLP_NN[-1], 1, dtype=torch.float64),
            nn.Sigmoid()
        )

    def forward(self, inputs_cat, inputs_cot):
        embeddings = self.embedding_layers(inputs_cat, inputs_cot)
        embeddings = self.stacking_layers(embeddings)
        embeddings = self.MLP_layers(embeddings)
        return self.fc(embeddings)
