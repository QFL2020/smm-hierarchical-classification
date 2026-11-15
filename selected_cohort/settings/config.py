# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import os, json, torch
import sys
sys.path.extend(['../'])
from utils.misc import get_time, check_gpu_available

class Config:
    def __init__(self):
        self.SMM_label = "SMM"  # define the target label
        self.dataset = ["CleanedSMM", "Binary_Encoder", "Standard_Scaler"] # dataset type, category encoder, continuous encoder
        self.Sampling_Strategy = "Balance"  # training sampler
        self.Traditional_ML_settings = None
        # {
        #     'method': self.training_strategy,
        #     'return_model': False, 
        #     'estimators': 100, # [Random Forest] parameters
        # }

        self.seed = 53  # save dir: list or int
        self.device = 'cuda:0'  # training device  => dynamic device settings
        self.epoch = 200    # training epoch
        self.batch_size = 512   # batch size
        self.loss_func = ['default']    # default = BCE
        self.lr = 2e-4
        self.dropout = 0.3
        self.NN_model = ['NN']   # 3layer
        # self.Dynamic_cluster_settings = {
        #     "Feature_model_name": "ResMLP3",
        #     "initialize_strategy": "random",
        #     "retained_best_name": "F1-score",
        #     "only_test_load_model_name": "best",
        #     "load_previous": False,
        #     "cluster_settings": {
        #         "model_name": "Residual",
        #         "model_structure": [192, 256]
        #     },
        #     "classifier_settings": {
        #         "model_name": "Residual",
        #         "model_structure": [192, 256]
        #     }
        # }
        # self.HDC_settings = {
        #     "max_layer": 8,
        #     "min_size": 3000
        # }
        self.Dynamic_cluster_settings = None
        self.HDC_settings = None
        self.weight_decay = 1e-5
        self.lr_schduler = ['exp', 0.9] # stepLR

        self.save_dir = './results/{}/{}/{}/{}'.format(self.dataset[0], self.Sampling_Strategy, self.NN_model[0], get_time())

    def check_device(self):
        if self.device == 'cpu':
            return 
        if torch.cuda.is_available():
            if check_gpu_available(self.device):
                return 
            self.device = 'cuda' # auto selective
        else:
            self.device = 'cpu'

    def load_pretrained_model(self):
        if self.NN_model and self.NN_model[0] == 'SimSiam':
            self.NN_model[5] = os.path.dirname(os.path.abspath(__file__)) + f'/../models/pretrained_model/{self.NN_model[4]}/{self.seed}/'
        # if self.dataset[0] == 'Feature':
        #     # dataset[1]: model name, dataset[2]: pretrained model path
        #     self.dataset[2] = os.path.dirname(os.path.abspath(__file__)) + f'/../models/pretrained_model/{self.dataset[1]}/{self.seed}/'

    def save_to_json(self):
        config_dict = self.__dict__.copy()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        file_path = self.save_dir + '/config.json'
        config_dict["save_dir"] = str(config_dict["save_dir"])
        self.check_device()
        self.load_pretrained_model()

        with open(file_path, 'w+') as f:
            json.dump(config_dict, f, indent=4)

    def load_from_json(self, file_path, save: bool=True, only_eval: bool=False):
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
            self.__dict__.update(config_dict)
        
        if only_eval:
            return

        method_name = None
        if self.NN_model is None:
            method_name = self.Traditional_ML_settings['method']
        else:
            method_name = self.NN_model[0]

        if self.Dynamic_cluster_settings and self.Dynamic_cluster_settings['load_previous']:
            pass
        else:
            self.save_dir = './results/{}/{}/{}/{}'.format(self.dataset[0], self.Sampling_Strategy[0], method_name, get_time())
        if not os.path.exists(self.save_dir) and save:
            os.makedirs(self.save_dir)

