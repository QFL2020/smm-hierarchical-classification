# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, f1_score, matthews_corrcoef, jaccard_score
from sklearn.metrics import auc as auc_function
sys.path.extend(['../'])
from settings.config import Config
from utils.figure_plot import plot_outputs_distribution
from models.AutoEncoder import AutoEncoder
from models.SimSiam.model_factory import SimSiam

def binary_classification_metrics(outputs, targets, threshold=0.5):
    """
    Binary Classification Metrics
    
    parameters:
    outputs (numpy.array): probability
    targets (numpy.array): true label
    threshold (float):
    
    return:
    dict: TP, FP, TN, FN, AUC, Accuracy, Precision, Recall, G_mean, AUC-PR
    """
    
    # transform binary
    predictions = (outputs >= threshold).astype(int)
    accuracy = accuracy_score(targets, predictions)
    precision = 0 if sum(predictions) == 0 else precision_score(targets, predictions)
    recall = 0 if sum(targets) == 0 else recall_score(targets, predictions)
    
    # TN, FP, FN, TP
    cm = confusion_matrix(targets, predictions)
    try:
        tn, fp, fn, tp = cm.ravel()
        G_mean = np.sqrt(recall * (tn / (tn + fp)))
    except ValueError:
        if sum(targets) == 0:
            tn, fp, fn, tp = [len(targets), 0, 0, 0]
        elif sum(targets) == len(targets):
            tn, fp, fn, tp = [0, 0, 0, len(targets)]
        else:
            assert False, "ERROR, Target: SMM={}, Non-SMM={}".format(sum(targets), len(targets) - sum(targets))
        G_mean = 1.0
    # calculate metrics
    try:
        auc = roc_auc_score(targets, outputs)
    except ValueError:
        auc = -1.0
    # save in dict
    metrics = {
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'AUC': auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'G_mean': G_mean,
        'F1-score': 0.0 if precision + recall == 0.0 or precision == 0.0 or recall == 0.0 else f1_score(targets, predictions),
        'MCC': matthews_corrcoef(targets, predictions), # Matthews Correlation Coefficient
        'Jaccard': jaccard_score(targets, predictions),
    }
    try:
        precision, recall, _ = precision_recall_curve(targets, outputs)
        auc_pr = auc_function(recall, precision)
    except ValueError:
        auc_pr = -1.0
    metrics['AUC-PR'] = auc_pr
    return metrics

def get_metrix_NN(model, test_loader):
    model.eval()
    with torch.no_grad():
        targets = []
        outputs = []
        for i, data in enumerate(test_loader):
            (inputs_cat, inputs_cot, target) = data

            if type(model) == SimSiam:
                output = model.forward_test(inputs_cat, inputs_cot)
            else:
                output = model(inputs_cat, inputs_cot)
            outputs.extend(output.detach().cpu().numpy())
            targets.extend(target.detach().cpu().numpy())
    
    targets = np.array(targets)
    outputs = np.array(outputs)

    return binary_classification_metrics(outputs, targets)

def get_metrics_AE(model, test_loader, threshold: float):
    model.eval()
    with torch.no_grad():
        loss_func = torch.nn.MSELoss(reduction='none')
        pred = []
        targets = []
        for i, data in enumerate(test_loader):
            (inputs_cat, inputs_cot, target) = data
            output = model(inputs_cat, inputs_cot)

            loss = loss_func(output, torch.cat((inputs_cat, inputs_cot), dim=-1)).mean(dim=-1).detach().cpu().numpy()
            cur_pred = np.zeros(len(target))
            cur_pred[loss > threshold] = 1
            
            pred.extend(cur_pred)
            targets.extend(target.detach().cpu().numpy())
    
    targets = np.array(targets)
    outputs = np.array(pred)

    return binary_classification_metrics(outputs, targets)

def get_metrix(model, test_loader, **kwargs):
    if type(model) == AutoEncoder:
        return get_metrics_AE(model, test_loader, **kwargs)
    return get_metrix_NN(model, test_loader)

def evaluate(model, test_loader, loss_func):
    model.eval()
    with torch.no_grad():
        cur_test_loss = 0
        for i, data in enumerate(test_loader):
            (inputs_cat, inputs_cot, target) = data
            outputs = model(inputs_cat, inputs_cot)
            if type(model) == AutoEncoder:
                loss = loss_func(outputs, torch.cat((inputs_cat, inputs_cot), dim=-1))
            else:
                loss = loss_func(outputs, target)
            cur_test_loss += loss.item()

        cur_test_loss /= len(test_loader)

    return cur_test_loss

def Only_Eval_func(model, train_dataset, test_dataset, eval_dataset: str, config: Config):
    batch_size = config.batch_size
    device = config.device
    save_dir = config.save_dir

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    load_path = config.save_dir + '/model_final.pth'
    model.load_state_dict(torch.load(load_path))
    model.to(device)
    metrics = get_metrix(model, test_dataloader)
    train_metrics = get_metrix(model, train_dataloader)

    with torch.no_grad():
        model.eval()

        if eval_dataset == 'test_dataset':
            targets = test_dataset.targets.detach().cpu().numpy().squeeze(-1)
            dataset_loader = test_dataloader
        else:
            targets = train_dataset.targets.detach().cpu().numpy().squeeze(-1)
            dataset_loader = train_dataloader

        outputs = None
        targets = []
        for (input_cat, input_cot, _) in dataset_loader:
            if isinstance(model, SimSiam):
                current_outputs = model.forward_test(input_cat, input_cot).detach().cpu().numpy()
            else:
                current_outputs = model(input_cat, input_cot).detach().cpu().numpy()
            if outputs is None:
                outputs = current_outputs
            else:
                outputs = np.concatenate([outputs, current_outputs])
            targets.extend(_.detach().cpu().numpy())

        outputs = outputs.squeeze(-1)
        targets = np.array(targets)
        # outputs = model(inputs_cat, inputs_cot).detach().cpu().numpy().squeeze(-1)
        # output_save_dir = save_dir + f'/{eval_dataset}.csv'
        # outputs_data = pd.DataFrame(outputs)
        # outputs_data.to_csv(output_save_dir, header=None, index=None)

        low_probs = np.where(outputs <= 0.5)[0]
        low_probs_SMM = np.where(targets[low_probs] == 1)[0]
        low_probs_Non_SMM = np.where(targets[low_probs] == 0)[0]
        print('low-probs: SMM={}, Non-SMM={}'.format(len(low_probs_SMM), len(low_probs_Non_SMM)))
        hig_probs = np.where(outputs > 0.5)[0]
        hig_probs_SMM = np.where(targets[hig_probs] == 1)[0]
        hig_probs_Non_SMM = np.where(targets[hig_probs] == 0)[0]
        print('hig-probs: SMM={}, Non-SMM={}'.format(len(hig_probs_SMM), len(hig_probs_Non_SMM)))
        plot_outputs_distribution(outputs, targets, config.dataset[0] + '-' + eval_dataset + '-output_distribution', config.save_dir)

    print(metrics)
    print(train_metrics)

    if eval_dataset == "test_dataset":
        return metrics
    return train_metrics
