# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
"""
Auto Encoder Architecture
"""
import torch.nn as nn
import torch, sys
sys.path.extend(["../"])
from models.base import base_NN, Deep_Net, Embedding_layers

class AutoEncoder(base_NN):
    def __init__(self, cat_length: list, cot_length: int,
                 Encoder_list: list, Decoder_list: list, p: float=0.3):
        super(AutoEncoder, self).__init__()
        self.cat_length = cat_length
        self.cot_length = cot_length
        
        self.embedding_layers = Embedding_layers(cat_length, cot_length)
        self.input_dim = self.embedding_layers.input_dim
        self.cat_dim = self.embedding_layers.cat_dim
        self.encoder_layers = Deep_Net(self.input_dim, Encoder_list, p)
        self.decoder_layers = Deep_Net(Encoder_list[-1], Decoder_list, p)

        self.fc = nn.Sequential(
            nn.Linear(Decoder_list[-1], self.cat_dim + self.cot_length, dtype=torch.float64),
            nn.Sigmoid()
        )

    def forward(self, inputs_cat, inputs_cot):
        embeddings = self.embedding_layers(inputs_cat, inputs_cot)

        embeddings = self.encoder_layers(embeddings)
        embeddings = self.decoder_layers(embeddings)
        embeddings = self.fc(embeddings)

        return embeddings
