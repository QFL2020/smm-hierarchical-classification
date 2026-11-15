# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.optim as optim
from tqdm import tqdm
from utils import tracking as wandb
sys.path.extend(["./"])

from dataset.get_dataset_from_config import get_dataset_from_config
from utils.figure_plot import plot_training_testing_curve
from utils.eval import evaluate, get_metrix, Only_Eval_func
from utils.misc import save_csv_file, setup_seed
from dataset.sampler import BalancedSampler
from utils.training_utils import get_lr_schduler, get_loss, get_threshold, get_model
from settings.config import Config

from models.AutoEncoder import AutoEncoder
from models.SimSiam.model_factory import SimSiam
from models.SimSiam.noisy_contrastive_train import train_SimSiam
from models.DynamicClustering.dynamic_cluster import train_dynamic_cluster
from models.HierarchicalDynamicClustering.train_HDC import train_HDC

from models.Traditional_ML.ML_baselines_interface import train_Traditioanl_ML


os.environ["WANDB_MODE"] = "disabled"

def train(config: Config, model, epoch: int, optimizer, scheduler, loss_func, 
          train_loader, val_loader, test_loader, save_dir, seed):
    os.environ["WANDB_DIR"] = config.save_dir
    wandb.init(
        # set the wandb project where this run will be logged
        project="SMM-{}".format(config.dataset[0]),
        group="{}".format('deep learning-based methods'),
        name='seed' + str(seed) + '-' + config.Sampling_Strategy[0],
        # track hyperparameters and run metadata
        config=config.__dict__
    )
    training_loss = []
    validate_loss = []

    best_model_loss = 1e9
    best_epoch = 0

    training_loss_step = []
    training_loop = tqdm(range(epoch), desc='epoch')
    for _epoch in training_loop:
        model.train()

        training_SMM_count = 0
        total_training_samples = 0
        cur_train_loss = 0
        for i, (inputs_cat, inputs_cot, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs_cat, inputs_cot)

            if type(model) == AutoEncoder:  # AutoEncoder type: must MSE loss
                loss = loss_func(outputs, torch.cat((inputs_cat, inputs_cot), dim=-1))
            else:
                loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()

            cur_train_loss += loss.item() * len(targets)
            training_SMM_count += torch.sum(targets).item()
            total_training_samples += len(targets)
            training_loss_step.append(loss.item())

        cur_train_loss /= total_training_samples
        training_loss.append(cur_train_loss)
        wandb.log({"Training loss": cur_train_loss}, step=_epoch)

        # save checkpoint
        if _epoch % 10 == 0:
            cur_save_path = os.path.join(save_dir, 'ckpt')
            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)
            model.save(cur_save_path + '/model-{}.pth'.format(_epoch))

            cur_save_path = os.path.join(save_dir, 'training_samples_count')
            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)
            with open(cur_save_path + '/{}.txt'.format(str(_epoch)), 'w+') as f:
                f.write("{}/{}\n".format(training_SMM_count, total_training_samples))

        # validation
        cur_val_loss = evaluate(model, val_loader, loss_func)
        validate_loss.append(cur_val_loss)
        wandb.log({"Validate loss": cur_val_loss}, step=_epoch)

        if scheduler:
            wandb.log({"lr": scheduler.get_last_lr()[0]}, step=_epoch)
        else:
            wandb.log({"lr": optimizer.state_dict()['param_groups'][0]['lr']}, step=_epoch)
        # save model_best
        if best_model_loss > cur_val_loss:
            best_model_loss = cur_val_loss
            best_epoch = _epoch
            model.save(os.path.join(save_dir, 'model_best.pth'))
        
        # save model_final
        model.save(os.path.join(save_dir, 'model_final.pth'))

        if scheduler:
            scheduler.step()

        # test
        threshold = get_threshold(model, train_loader) if type(model) == AutoEncoder else None
        metrics = get_metrix(model, test_loader, threshold=threshold)
        for key in metrics.keys():
            wandb.log({key: metrics[key]}, step=_epoch)

        if scheduler:
            cur_lr = scheduler.get_last_lr()[0]
        else:
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        # print("{}/{}: lr={}, training_loss={}, val_loss={}\n".format(_epoch, epoch,
                                                                #    cur_lr, cur_train_loss, cur_val_loss))
        training_loop.set_postfix(lr=cur_lr, train_loss=cur_train_loss, test_loss=cur_val_loss,
                                  AUC=metrics['AUC'], Precision=metrics['Precision'], Recall=metrics['Recall'], 
                                  Acc=metrics['Accuracy'], F1_score=metrics['F1-score'])

    
    with open(os.path.join(save_dir, 'model_loss.txt'), 'w+') as f:
        f.write('model_best[{}]:{}\n'.format(best_epoch, best_model_loss))
        f.write('model_final[{}]:{}\n'.format(epoch, cur_val_loss))

    save_csv_file(training_loss_step, os.path.join(save_dir, 'training_loss_step.csv'))
    plot_training_testing_curve(training_loss_step, 'step', 'loss', 'training step-loss', 'training loss[step]', config.save_dir + '/training step-loss.png')
    
    return training_loss, validate_loss


def _run(config: Config, only_evaluation: bool=False, eval_dataset: str='test_dataset'):
    if config.NN_model and config.NN_model[0] == 'AutoEncoder':
        assert config.loss_func[0] == 'MSE', "Error: AutoEncoder requires 'MSE' as the loss function."
    elif config.NN_model and config.NN_model[0] == 'FT_Transformer':
        assert config.dataset[1] == 'Label_Encoder', "Error: FT_Transformer requires LabelEncoder as the categorical encoder"
    # elif config.NN_model[0] == 'DynamicClustering':
    #     assert config.dataset[0] == 'Feature', "Error: DynamicClustering requires Feature-based dataset as the input"
    if not only_evaluation:
        config.save_to_json()
    
    if config.NN_model is None:
        return train_Traditioanl_ML(config.Traditional_ML_settings['method'], only_evaluation, config)
    elif config.NN_model[0] == 'DynamicClustering':
        return train_dynamic_cluster(config, only_evaluation)
    elif config.NN_model[0] == 'HDC':
        return train_HDC(config, only_evaluation)
    
    Sampling_Strategy = config.Sampling_Strategy
    save_dir = config.save_dir
    seed = config.seed
    device = config.device
    epoch = config.epoch
    batch_size = config.batch_size
    lr = config.lr
    lr_schduler_type = config.lr_schduler

    setup_seed(seed)
    train_dataset, val_dataset, test_dataset, cat_list = get_dataset_from_config(config, cur_path='./dataset/', dtype=torch.float32 if config.NN_model[0] == 'FT_Transformer' else torch.float64)
    
    if Sampling_Strategy[0] == 'Balance': # Mini Batch Balanced Training
        sampler = BalancedSampler(train_dataset, batch_size)
        train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else: # default imbalanced training
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # model init
    model = get_model(config, cat_list, train_dataset.cot_length)
    if only_evaluation:
        metrics = Only_Eval_func(model, train_dataset, test_dataset, eval_dataset, config)
        return metrics

    # model init
    model.reset_weights()
    model.to(device)

    # loss init
    loss = get_loss(config)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    if config.NN_model[0] == 'FT_Transformer':
        optimizer = model.make_default_optimizer()
    scheduler = get_lr_schduler(optimizer, lr_schduler_type)

    # save init weights
    model.save(os.path.join(save_dir, 'model_init.pth'))
    # training
    if isinstance(model, SimSiam):
        train_loss, val_loss = train_SimSiam(config, model, epoch, optimizer, scheduler, loss, 
                                train_dataloader, val_dataloader, test_dataloader, save_dir, seed)
    else:
        train_loss, val_loss = train(config, model, epoch, optimizer, scheduler, loss, 
                                train_dataloader, val_dataloader, test_dataloader, save_dir, seed)

    save_csv_file(train_loss, os.path.join(save_dir, 'train_loss.csv'))
    save_csv_file(val_loss, os.path.join(save_dir, 'val_loss.csv'))
    plot_training_testing_curve(train_loss, 'epoch', 'loss', 'training epoch-loss', 'training loss[epoch]', config.save_dir + '/training loss.png')
    plot_training_testing_curve(val_loss, 'epoch', 'loss', 'testing epoch-loss', 'testing loss[epoch]', config.save_dir + '/testing loss.png')

    threshold = get_threshold(model, train_dataloader) if isinstance(model, AutoEncoder) else None
    metrics = get_metrix(model, test_dataloader, threshold=threshold)
    print('final metrics')
    print(metrics)

    return metrics

# config_path = None
# config_path = r'./config.json'
from multiprocessing import Pool
import copy

import argparse
def add_args(parser):
    parser.add_argument("--config_path", type=str, default="./settings/config.json", help='config path')
    parser.add_argument("--eval", action='store_true', help="Model evaluation only from config path")
    parser.add_argument("--eval_dataset", default="test_dataset", help='Which dataset to test?[test_dataset or train_dataset]')

if __name__ == '__main__':
    config = Config()
    parser = argparse.ArgumentParser()
    add_args(parser)
    cfg = parser.parse_args()
    config_path = cfg.config_path
    only_eval = cfg.eval
    eval_dataset = cfg.eval_dataset
    config.load_from_json(config_path, only_eval=only_eval)

    cuda_list = config.device
    if type(config.seed) is list:
        pool = Pool(5)
        results = []
        seeds = config.seed
        save_dir = config.save_dir
        cuda_list_idx = 0
        for seed in seeds:
            config.save_dir = os.path.join(save_dir, str(seed))
            config.seed = seed
            if type(cuda_list) is list:
                config.device = cuda_list[cuda_list_idx % len(cuda_list)]
                cuda_list_idx += 1
            cur_config = copy.deepcopy(config)

            results.append(pool.apply_async(_run, (cur_config, only_eval, eval_dataset)))

        pool.close()
        pool.join()

        print('save results')
        for i in range(len(results)):
            results[i] = results[i].get()
        
        keys = results[0].keys()
        all_results = {}
        
        for key in keys:
            all_results[key] = []
            for i in range(len(results)):
                all_results[key].append(results[i][key])
        
        save_name = '/results[train_dataset].txt' if only_eval and eval_dataset == 'train_dataset' else '/results.txt'
        with open(save_dir + save_name, 'w+') as f:
            for key in keys:
                f.write("{}: {:.2f}±{:.2f}\n".format(key, np.mean(all_results[key]), np.std(all_results[key], ddof=1)))
        print('done')
    else:
        if type(cuda_list) is list:
            config.device = cuda_list[0]
        metrics = _run(config, only_eval, eval_dataset)
        save_name = 'results[train_dataset].txt' if only_eval and eval_dataset == 'train_dataset' else 'results.txt'
        # save metrics
        with open(os.path.join(config.save_dir, save_name), 'w+') as f:
            for key in metrics.keys():
                f.write("{}:{}".format(key, metrics[key]) + '\n')