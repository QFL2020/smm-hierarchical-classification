# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_outputs_distribution(outputs, targets, save_name, save_path):
    if type(outputs) is list:
        outputs = np.array(outputs)
    if type(targets) is list:
        targets = np.array(targets)
    SMM_indices = np.where(targets > 0.0)[0]
    Non_SMM_indices = np.where(targets == 0.0)[0]
    # print(len(SMM_indices), sum(targets[SMM_indices]))
    # print(len(Non_SMM_indices), sum(targets[Non_SMM_indices]))
    
    # print("SMM: {}, Non-SMM: {}".format(len(SMM_indices), len(Non_SMM_indices)))

    current_palette = sns.color_palette()
    sns.kdeplot(outputs[Non_SMM_indices], shade=True, label='Non-SMM', color=current_palette[0])
    sns.kdeplot(outputs[SMM_indices], shade=True, label='SMM', color=current_palette[1])

    plt.title(save_name)
    plt.xlabel('output(probs)')
    plt.ylabel('density')
    plt.legend()

    plt.savefig(save_path + "/" + save_name)
    plt.close()


def plot_training_testing_curve(training_testing_info, x_label, y_label, title, plot_label, save_name):
    steps = range(1, len(training_testing_info) + 1)

    plt.figure(figsize=(12, 6))

    plt.plot(steps, training_testing_info, label=plot_label, color='blue', linewidth=2)
    plt.title(title, fontsize=18, pad=20)
    plt.xlabel(x_label, fontsize=16, labelpad=10)
    plt.ylabel(y_label, fontsize=16, labelpad=10)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss(train_loss_list, val_loss_list, save_dir, save_name: str='loss_plot', xtick_interval: int=5):
    epochs = np.arange(1, len(train_loss_list) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_list, label='Train Loss', color='blue', linestyle='-', marker='o')
    plt.plot(epochs, val_loss_list, label='Validation Loss', color='orange', linestyle='-', marker='s')
    
    plt.title('Train and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(1, len(train_loss_list) + 1, xtick_interval))  
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/{save_name}.png")
    
    plt.close()

def plot_score(score_list, save_dir, save_name: str='score_plot', xtick_interval: int=5):
    epochs = np.arange(1, len(score_list) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, score_list, label=save_name, color='blue', linestyle='-', marker='o')

    plt.title(f'{save_name} Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(f'{save_name}')
    plt.xticks(np.arange(1, len(score_list) + 1, xtick_interval))  
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/{save_name}.png")
    
    plt.close()
