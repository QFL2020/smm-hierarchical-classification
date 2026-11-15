# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import os
import random
from datetime import datetime

import numpy as np
import torch

def remove_file(path):
    if os.path.exists(path):
        os.remove(path)

def load_from_csv(data_path: str):
    inputs = []
    targets = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            fields = [float(_) for _ in line[:-1].split(",")]
            inputs.append(fields[:-1])
            targets.append(fields[-1])
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets

def save_csv_file(data, save_path):
    lines = []
    for i in range(len(data)):
        if data[i] is list:
            line = ",".join([str(i)] + [str(_) for _ in data[i]]) + '\n'
        else:
            line = "{},{}".format(i, data[i]) + '\n'
        
        lines.append(line)
    
    with open(save_path, 'w+') as f:
        f.writelines(lines)

def get_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")

    return formatted_time

def setup_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_gpu_available(device):
    if torch.cuda.is_available():
        try:
            torch.tensor([1]).to(device)
            return True
        except RuntimeError:
            return False
    else:
        return False
