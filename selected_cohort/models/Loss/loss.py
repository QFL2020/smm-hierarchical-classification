# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
"""
loss definition:
1. focal loss
2. TverskyLoss
3. Sensitivity_Specificity_Loss
4. CombinedLoss
5. F1_beta_loss
6. MFE_loss: 2 types
7. focal_loss + Tversky Loss
"""
# focal loss
# https://github.com/qingyujean/Classification-on-imbalanced-data/blob/master/Classification/src/imbalanced/tune_params_for_focal_loss.py
# Tversky loss
# https://arxiv.org/pdf/1706.05721.pdf

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, reduction='mean', weight: float=None):
        """
        :param alpha: for target = 1
        :param gamma: 
        :param reduction: 'none', 'mean', 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        """
        compute Focal Loss
        :param inputs:  [batch_size, *(sigmod)]
        :param targets: truth_label [batch_size, *]
        :return: Focal Loss
        """

        # compute BCE Loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # print(inputs)
        p_t = (1 - inputs) * targets + inputs * (1 - targets)
        # compute focal term
        focal_term = p_t ** self.gamma

        # compute alpha term
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal Loss
        focal_loss = alpha_t * focal_term * bce_loss

        if self.reduction == 'mean':
            if self.weight:
                return self.weight * focal_loss.mean()
            return focal_loss.mean()
        elif self.reduction == 'sum':
            if self.weight:
                return self.weight * focal_loss.sum()
            return focal_loss.sum()
        else:
            if self.weight:
                return self.weight * focal_loss
            return focal_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, weight: float=None):
        """
        :param alpha: FP
        :param beta: FN: Key[minority]
        :param smooth: be sure not zero
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.weight = weight

    def forward(self, inputs, targets):
        """
        Compute Tversky Loss
        :param inputs: Sigmode outputs[Probs]
        :param targets: Truth label
        :return: Tversky Loss
        """
        # compute TP, FP, FN
        TP = torch.sum(inputs * targets)
        FP = torch.sum((1 - targets) * inputs)
        FN = torch.sum(targets * (1 - inputs))
        
        # Tversky Index
        Tversky_Index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # Tversky Loss
        Tversky_Loss = 1 - Tversky_Index

        if self.weight:
            Tversky_Loss *= self.weight
        return Tversky_Loss.mean()

class Sensitivity_Specificity_Loss(nn.Module):
    def __init__(self, _lambda=0.5, smooth=1e-6):
        """
        :param _lambda: positive label weight
        :param smooth: be sure not zero
        """
        super(Sensitivity_Specificity_Loss, self).__init__()
        self._lambda = _lambda
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Compute SS Loss
        :param inputs: Sigmode outputs[Probs]
        :param targets: Truth label
        :return: SS Loss
        """
        loss = self._lambda * torch.sum((targets - inputs) ** 2 * targets) / (torch.sum(targets) + self.smooth) + \
        (1 - self._lambda) * torch.sum((targets - inputs) ** 2 * (1 - targets)) / (torch.sum(1 - targets) + self.smooth)

        return loss


class CombinedLoss(nn.Module):
    def __init__(self, weights=None):
        super(CombinedLoss, self).__init__()
        if weights is None:
            self.weights = [0.25, 0.25, 0.25, 0.25]
        else:
            self.weights = weights

    def get_cal_data(self, inputs, targets):
        # True Positives, False Positives, True Negatives, False Negatives
        TP = (inputs * targets).sum()
        TN = ((1 - inputs) * (1 - targets)).sum()
        FP = (inputs * (1 - targets)).sum()
        FN = ((1 - inputs) * targets).sum()

        return TP, TN, FP, FN

    def forward(self, inputs, targets):

        # compute AUC
        # auc = roc_auc_score(targets.cpu(), inputs.cpu())
        auc = 0
        
        TP, TN, FP, FN = self.get_cal_data(inputs, targets)
        # compute Accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        # compute Precision
        precision = TP / (TP + FP + 1e-7)
        
        # compute Recall
        recall = TP / (TP + FN + 1e-7)
        
        # loss = - (AUC + Accuracy + Precision + Recall)
        loss = -(auc * self.weights[0] + accuracy * self.weights[1] + precision * self.weights[2] + recall * self.weights[3])
        
        return loss

'''
def f1_loss(predict, target):
    predict = torch.sigmoid(predict)
    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return 1 - f1.mean()
'''
class F1_beta_loss(nn.Module):
    def __init__(self, beta=1):
        """
        :param beta: positive label weight
        """
        super(F1_beta_loss, self).__init__()
        self.beta = beta

    def forward(self, predict, targets):
        predict = torch.clamp(predict * (1-targets), min=0.01) + predict * targets
        tp = predict * targets
        tp = tp.sum(dim=0)
        precision = tp / (predict.sum(dim=0) + 1e-8)
        recall = tp / (targets.sum(dim=0) + 1e-8)
        f1 = 2 * (precision * recall / (precision + recall + 1e-8))
        return 1 - f1.mean()

class MFE_loss(nn.Module):
    def __init__(self, square: bool=False):
        super(MFE_loss, self).__init__()
        self.square = square
    
    def forward(self, predict, targets):

        FPE = torch.sum(predict ** 2 * (1 - targets)) / torch.sum((1 - targets))
        FNE = torch.sum((1 - predict) ** 2 * targets) / torch.sum(targets)

        if self.square:
            loss = FPE ** 2 + FNE ** 2
        else:
            loss = FPE + FNE
        
        return loss

class focal_loss_TverskyLoss(nn.Module):
    def __init__(self, p1: list, p2: list):
        super(focal_loss_TverskyLoss, self).__init__()
        self.loss1 = FocalLoss(p1[1], p1[2], reduction='mean', weight=p1[0])
        self.loss2 = TverskyLoss(alpha=p2[1], beta=p2[2], smooth=p2[3], weight=p2[0])
    
    def forward(self, predict, targets):
        loss = self.loss1(predict, targets) + self.loss2(predict, targets)
        return loss

from info_nce import InfoNCE, info_nce
class InfoNCE_BCE_loss(nn.Module):
    def __init__(self, alpha:int = 1):
        super(InfoNCE_BCE_loss, self).__init__()
        self.alpha = alpha
        self.InfoNCE_loss = InfoNCE(negative_mode='paired')
        self.BCE_loss = nn.CrossEntropyLoss()
    
    def forward(self, predict, BCE_labels, query, positive_key, negative_keys, epoch: int=None, hig_label=None):
        loss_1 = self.BCE_loss(predict, BCE_labels)
        if positive_key is None:
            return loss_1
        
        if hig_label is not None:
            hig_indices = (hig_label == 1)
            if sum(hig_indices) == 0:
                loss_2 = 0
            else:
                loss_2 = self.InfoNCE_loss(query[hig_indices], positive_key[hig_indices], negative_keys[hig_indices])
        else:
            loss_2 = self.InfoNCE_loss(query, positive_key, negative_keys)
        if self.alpha > 0:
            return loss_1 + self.alpha * loss_2
        elif self.alpha == -1:
            return loss_1 + min(epoch / 50, 1.0) * loss_2
        elif self.alpha == 0:
            return loss_1
        else:
            return None

# Copy from: https://github.com/liyi01827/noisy-contrastive/blob/main/main.py
class GCE_loss(nn.Module):
    def __init__(self, q=0.8):
        super(GCE_loss, self).__init__()
        self.q = q

    def forward(self, outputs, targets):
        pred_y = outputs * targets + (1 - outputs) * (1 - targets)
        pred_y = torch.clamp(pred_y, 1e-4)
        final_loss = torch.mean((1.0 - pred_y ** self.q) / self.q, dim=0)
        return final_loss


