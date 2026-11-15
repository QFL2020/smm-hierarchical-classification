# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
# https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md#usage
import torch.optim, torch, sys
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
sys.path.extend(['../../'])
from models.base import base_NN

class FT_Transformer(base_NN):
    def __init__(self, n_cont_features, cat_cardinalities, **args):
        '''
        n_cont_features: the number of continuous features
        cat_cardinalities: cats length
        '''
        super(FT_Transformer, self).__init__()
        default_kwargs = FTTransformer.get_default_kwargs()
        self.model = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            d_out=1,
            **default_kwargs,
        )
     
    def forward(self, inputs_cat, inputs_cot):
        output = self.model(inputs_cot, inputs_cat)
        output = torch.sigmoid(output)
        return output
    
    def make_default_optimizer(self):
        return self.model.make_default_optimizer()

class Tabular_FT_ResNet(base_NN):
    def __init__(self, n_cont_features, cat_cardinalities, **args):
        '''
        n_cont_features: the number of continuous features
        cat_cardinalities: cats length
        '''
        super(Tabular_FT_ResNet, self).__init__()
        self.model = ResNet(
            d_in=n_cont_features + sum(cat_cardinalities),
            d_out=1,
            n_blocks=2,
            d_block=192,
            d_hidden=None,
            d_hidden_multiplier=2.0,
            dropout1=0.15,
            dropout2=0.0,
        )
     
    def forward(self, inputs_cat, inputs_cot):
        '''
        x_cat_ohe = [
            F.one_hot(cat_column, c)
            for cat_column, c in zip(x_cat.T, cat_cardinalities)
        ]
        x = torch.column_stack([x_cont] + x_cat_ohe)
        '''
        x = torch.cat((inputs_cot, inputs_cat), dim=-1)
        output = self.model(x)
        output = torch.sigmoid(output)
        return output
    

class Tabular_FT_MLP(base_NN):
    def __init__(self, n_cont_features, cat_cardinalities, **args):
        '''
        n_cont_features: the number of continuous features
        cat_cardinalities: cats length
        '''
        super(Tabular_FT_MLP, self).__init__()
        self.model = MLP(
            d_in=n_cont_features + sum(cat_cardinalities),
            d_out=1,
            n_blocks=2,
            d_block=384,
            dropout=0.1,
        )
     
    def forward(self, inputs_cat, inputs_cot):
        '''
        x_cat_ohe = [
            F.one_hot(cat_column, c)
            for cat_column, c in zip(x_cat.T, cat_cardinalities)
        ]
        x = torch.column_stack([x_cont] + x_cat_ohe)
        '''
        x = torch.cat((inputs_cot, inputs_cat), dim=-1)
        output = self.model(x)
        output = torch.sigmoid(output)
        return output
    
