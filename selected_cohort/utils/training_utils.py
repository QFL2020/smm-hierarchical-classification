# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import sys, torch
import numpy as np
sys.path.extend(["../"])
from settings.config import Config
from models.Loss.loss import FocalLoss, TverskyLoss, F1_beta_loss, Sensitivity_Specificity_Loss, CombinedLoss, MFE_loss

from models.NN import Multi_MLP_Cat_Num
from models.DCN import DeepCross
from models.Residual_factory.residual_block import ResidualNN
from models.Residual_factory.SE_Residual import SE_ResidualNN
from models.AutoEncoder import AutoEncoder
from models.SimSiam.model_factory import SimSiam
from models.Tabular_model.FT_Transformer import FT_Transformer



def lr_lambda(epoch):
    if epoch < 200:
        # warm up
        return epoch / 200
    else:
        # decrease
        return 1 / (1 + 0.2 * (epoch - 200))

def lr_factor_lambda(epoch):
    return 1 / (1 + 0.2 * (epoch))

def get_lr_schduler(optimizer, lr_schduler_type):
    if lr_schduler_type[0] == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_schduler_type[1], last_epoch=-1)
    elif lr_schduler_type[0] == 'stepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=lr_schduler_type[1], step_size=lr_schduler_type[2])
    elif lr_schduler_type[0] == 'LAMBDA':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif lr_schduler_type[0] == 'LAMBDA2':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor_lambda)
    elif lr_schduler_type[0] == 'none':
        scheduler = None
    return scheduler

def get_loss(config: Config):
    # training utils
    if config.loss_func[0] == 'default':
        loss = torch.nn.BCELoss()
    elif config.loss_func[0] == 'focal_loss':
        loss = FocalLoss(alpha=config.loss_func[1], gamma=config.loss_func[2], reduction='mean')
    elif config.loss_func[0] == 'TverskyLoss':
        # alpha[FP], beta[FN], smooth
        loss = TverskyLoss(alpha=config.loss_func[1], beta=config.loss_func[2], smooth=config.loss_func[3])
    elif config.loss_func[0] == 'Sensitivity_Specificity_Loss':
        # lambda[TP], smooth
        loss = Sensitivity_Specificity_Loss(_lambda=config.loss_func[1], smooth=config.loss_func[2])
    elif config.loss_func[0] == 'CombinedLoss':
        loss = CombinedLoss(config.loss_func[1])
    elif config.loss_func[0] == 'MSE':
        loss = torch.nn.MSELoss()
    elif config.loss_func[0] == 'F1_beta_loss':
        # beta larger [more recall]
        # beta smaller [more precision]
        loss = F1_beta_loss(config.loss_func[1])
    elif config.loss_func[0] == 'MFE_loss':
        loss = MFE_loss(config.loss_func[1])
    elif config.loss_func[0] == 'focal_loss+TverskyLoss':
        from models.Loss.loss import focal_loss_TverskyLoss
        loss = focal_loss_TverskyLoss(config.loss_func[1], config.loss_func[2])

    elif config.loss_func[0] == 'GCE':
        from models.Loss.loss import GCE_loss
        loss = GCE_loss(config.loss_func[1])

    return loss

def get_threshold(model, train_loader):
    model.eval()
    with torch.no_grad():
        loss_func = torch.nn.MSELoss(reduction='none')
        cur_train_loss = []
        for i, data in enumerate(train_loader):
            (inputs_cat, inputs_cot, target) = data
            outputs = model(inputs_cat, inputs_cot)
            loss = loss_func(outputs, torch.cat((inputs_cat, inputs_cot), dim=-1)).mean(dim=-1)

            cur_train_loss.extend(loss.detach().cpu().numpy())
    return np.percentile(cur_train_loss, 95)

def get_model(config: Config, cat_list, cot_length):
    # model init
    if config.NN_model[0] == 'NN':
        model = Multi_MLP_Cat_Num(cat_list, cot_length, 
                                  Stacking_NN=config.NN_model[1], MLP_NN=config.NN_model[2], p=config.dropout)
    elif config.NN_model[0] == 'DCN':
        cross_type_mapping = {'v1': 'vec', 'v2': 'mtx'}
        cross_type = cross_type_mapping.get(config.NN_model[3], None)
        assert cross_type is not None, f'Not found DCN version with {config.NN_model[3]}, please modify with [\'v1\', \'v2\']'
        model = DeepCross(cat_list, cot_length, 
                 Deep_NN=config.NN_model[1], Cross_NN_layers=config.NN_model[2], 
                 p=config.dropout, cross_type=cross_type)
    elif config.NN_model[0] == 'residual_MLP':
        model = ResidualNN(cat_list, cot_length, stack_layers=config.NN_model[1], 
                           act=config.NN_model[2], p=config.dropout)
    elif config.NN_model[0] == 'SE_residual_MLP':
        model = SE_ResidualNN(cat_list, cot_length, stack_layers=config.NN_model[1], 
                              act=config.NN_model[2], p=config.dropout)
    elif config.NN_model[0] == 'AutoEncoder':
        model = AutoEncoder(cat_list, cot_length, config.NN_model[1], config.NN_model[2], 
                            config.dropout)
    elif config.NN_model[0] == 'SimSiam':
        model = SimSiam(cat_list, cot_length, emam=config.NN_model[1], 
                        backbone_model_name=config.NN_model[4], pretrained_model_path=config.NN_model[5])

    elif config.NN_model[0] == 'FT_Transformer':
        model = FT_Transformer(cot_length, cat_list)
    else:
        assert False, "Not found model name: {}".format(config.NN_model[0])

    return model
