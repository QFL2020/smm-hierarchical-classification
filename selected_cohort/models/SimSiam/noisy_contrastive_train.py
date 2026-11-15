# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import sys, os, torch
from utils import tracking as wandb
from tqdm import tqdm
import numpy as np

sys.path.extend(['../../'])
from utils.misc import save_csv_file
from settings.config import Config
from models.SimSiam.model_factory import SimSiam
from models.SimSiam.noisy_tools import add_noise, scale_features, feature_dropout
from utils.eval import get_metrix
from utils.figure_plot import plot_training_testing_curve

def evaluate_SimSiam(model: SimSiam, test_loader, loss_func):
    model.eval()
    with torch.no_grad():
        cur_test_loss = 0
        for i, data in enumerate(test_loader):
            (inputs_cat, inputs_cot, target) = data
            outputs = model.forward_test(inputs_cat, inputs_cot)
            loss = loss_func(outputs, target)
            cur_test_loss += loss.item()

        cur_test_loss /= len(test_loader)

    return cur_test_loss

# just modify the continuous feature
def transform1(inputs_cot):
    inputs_cot = add_noise(inputs_cot, noise_std=0.1)
    inputs_cot = scale_features(inputs_cot, (0.9, 1.1))
    inputs_cot = feature_dropout(inputs_cot, 0.05)
    return inputs_cot
# just modify the continuous feature
def transform2(inputs_cot):
    inputs_cot = scale_features(inputs_cot, (0.9, 1.1))
    inputs_cot = add_noise(inputs_cot, noise_std=0.1)
    inputs_cot = feature_dropout(inputs_cot, 0.05)
    return inputs_cot

def train_SimSiam(config: Config, model: SimSiam, epoch: int, optimizer, scheduler, loss_func, 
          train_loader, val_loader, test_loader, save_dir, seed):
    os.environ["WANDB_DIR"] = config.save_dir
    wandb.init(
        # set the wandb project where this run will be logged
        project="[noisy-contrastive]SMM-{}".format(config.dataset[0]),
        group="{}".format('deep learning-based methods'),
        name='seed' + str(seed) + '-' + config.Sampling_Strategy[0],
        # track hyperparameters and run metadata
        config=config.__dict__
    )
    training_loss = []
    validate_loss = []

    best_model_loss = 1e9
    best_epoch = 0
    lamb = config.NN_model[2]
    tau = config.NN_model[3]

    training_loss_step = []
    training_loop = tqdm(range(epoch), desc='epoch')
    for _epoch in training_loop:
        model.train()

        training_SMM_count = 0
        total_training_samples = 0
        cur_train_loss = 0
        for i, (inputs_cat, inputs_cot, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            bsz = targets.size(0)
            inputs_cot_aug1 = transform1(inputs_cot)
            inputs_cot_aug2 = transform2(inputs_cot)

            p1, z2, outputs = model(inputs_cat, inputs_cot_aug1, inputs_cat, inputs_cot_aug2, 
                                    inputs_cat, inputs_cot)

            # GCE loss
            loss_ce = loss_func(outputs, targets)

            # contrastive loss
            # avoid collapsing and gradient explosion
            p1 = torch.clamp(p1, 1e-4, 1.0 - 1e-4)
            z2 = torch.clamp(z2, 1e-4, 1.0 - 1e-4)

            contrast_1 = torch.matmul(p1, z2.t())  # B X B
            # <q,z> + log(1-<q,z>)
            contrast_1 = -contrast_1*torch.zeros(bsz, bsz).fill_diagonal_(1).cuda(device=config.device) + ((1-contrast_1).log()) * torch.ones(bsz, bsz).fill_diagonal_(0).cuda(device=config.device)
            contrast_logits = 2 + contrast_1

            # outputs: logits after softmax 
            contrast_mask = torch.matmul(outputs, outputs.t()).clone().detach()
            contrast_mask.fill_diagonal_(1)
            pos_mask = (contrast_mask >= tau).float()
            contrast_mask = contrast_mask * pos_mask
            contrast_mask = contrast_mask / contrast_mask.sum(1, keepdim=True)
            loss_ctr = (contrast_logits * contrast_mask).sum(dim=1).mean(0)

            loss = lamb * loss_ctr + loss_ce
            
            loss.backward()
            optimizer.step()

            cur_train_loss += loss.item()

            training_SMM_count += torch.sum(targets).item()
            total_training_samples += len(targets)
            training_loss_step.append({'contrastiveLoss': loss_ctr.item(), 'CELoss': loss_ce.item(), 'overallLoss': loss.item()})

        cur_train_loss /= len(train_loader)
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
        cur_val_loss = evaluate_SimSiam(model, val_loader, loss_func)
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
        metrics = get_metrix(model, test_loader)
        for key in metrics.keys():
            wandb.log({key: metrics[key]}, step=_epoch)

        if scheduler:
            cur_lr = scheduler.get_last_lr()[0]
        else:
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        training_loop.set_postfix(lr=cur_lr, train_loss=cur_train_loss, test_loss=cur_val_loss,
                            AUC=metrics['AUC'], Precision=metrics['Precision'], Recall=metrics['Recall'], 
                            Acc=metrics['Accuracy'], F1_score=metrics['F1-score'])
            
    with open(os.path.join(save_dir, 'model_loss.txt'), 'w+') as f:
        f.write('model_best[{}]:{}\n'.format(best_epoch, best_model_loss))
        f.write('model_final[{}]:{}\n'.format(epoch, cur_val_loss))

    save_csv_file(training_loss_step, os.path.join(save_dir, 'training_loss_step.csv'))
    overallLoss_step = [training_loss_step[_i]['overallLoss'] for _i in range(len(training_loss_step))]
    contrastiveLoss_step = [training_loss_step[_i]['contrastiveLoss'] for _i in range(len(training_loss_step))]
    CELoss_step = [training_loss_step[_i]['CELoss'] for _i in range(len(training_loss_step))]
    plot_training_testing_curve(overallLoss_step, 'step', 'loss', 'training step-loss[overallLoss]', 'training loss[step]', config.save_dir + '/training step-loss[overallLoss].png')
    plot_training_testing_curve(contrastiveLoss_step, 'step', 'loss', 'training step-loss[contrastiveLoss]', 'training loss[step]', config.save_dir + '/training step-loss[contrastiveLoss].png')
    plot_training_testing_curve(CELoss_step, 'step', 'loss', 'training step-loss[CELoss]', 'training loss[step]', config.save_dir + '/training step-loss[CELoss].png')
    
    return training_loss, validate_loss

