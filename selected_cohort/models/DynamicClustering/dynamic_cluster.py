# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
"""
Dynamic SMM Data Classifier
"""
import sys, os, json, argparse
import pandas as pd
import numpy as np
sys.path.extend(['../../'])
from utils.early_stopping import EarlyStoppingCallback
from utils.eval import binary_classification_metrics
from utils.misc import setup_seed, remove_file
from dataset.get_dataset_from_config import get_dataset_from_config
from models.DynamicClustering.model_training_zoo import get_cluster_model
from models.DynamicClustering.sub_tree_classifier import SubTreeClassifier

from settings.config import Config

class DynamicCluster:
    def __init__(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                 save_dir: str, initialize_strategy: str='residual', 
                 only_test: bool=False, load_name: str='final', max_iter: int=30, _input_dim: int=512, device: str='cuda:2',
                 load_previous: bool=False, early_stopping: bool=False, retained_best_name: str='AUC',
                 NN_max_iters: int=1000, 
                 sub_tree_cls_name: str='NN_Training-p=0.1', sub_tree_cls_structure: list=[64,128,64], 
                 tree_cluster_name: str='NN_Training-p=0.1', tree_cluster_structure: list=[64,128,64], 
                 min_size: int=3000,
                 **kwargs):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.min_size = min_size
        self.NN_max_iters = NN_max_iters
        self.train_data = train_data
        self.val_data = val_data

        self.save_dir = save_dir
        self.device = device
        
        self.X_train = self.train_data.drop('SMM', axis=1)
        self.y_train = self.train_data['SMM']

        self.save_name = "/NN-{}-{}.pth"
        self.cluster_model = get_cluster_model(tree_cluster_name, tree_cluster_structure, self.NN_max_iters, _input_dim, device, lr=kwargs['lr'])
        self.SMM_Detector = get_cluster_model(tree_cluster_name, tree_cluster_structure, self.NN_max_iters, _input_dim, device, lr=kwargs['lr'])
        self.left_model = SubTreeClassifier(sub_tree_cls_structure, self.NN_max_iters, _input_dim=_input_dim, 
                                                 device=device, lr=kwargs['lr'], sub_tree_cls_name=sub_tree_cls_name,
                                                 save_dir=self.save_dir)
        self.right_model = SubTreeClassifier(sub_tree_cls_structure, self.NN_max_iters, _input_dim=_input_dim, 
                                                  device=device, lr=kwargs['lr'], sub_tree_cls_name=sub_tree_cls_name,
                                                 save_dir=self.save_dir)

        # early stopping settings
        # self.early_stopping = 10
        self.max_iter = max_iter

        self.initialize_strategy = initialize_strategy
        self.early_stopping_setting = early_stopping
        self.retained_best_name = retained_best_name
        self.is_leaf = False

        if 'son_class' in kwargs.keys():
            print('is son class:', kwargs['son_class'])
            return 

        if load_previous:
            if self.load_model('best'):
                current_results, _data, self.is_over = self.check()
                return
            if self.load_model('final'):
                current_results, _data, self.is_over = self.check()
                return
            
        if only_test:
            self.load_model(load_name)
            current_results, _data, self.is_over = self.check()
        else:
            self.fit()

    def save_model(self, save_name: str='final'):
        self.cluster_model.save(self.save_dir + self.save_name.format('clustering', save_name))
        self.left_model.save(self.save_dir + self.save_name.format('left_SMM_model', save_name))
        self.right_model.save(self.save_dir + self.save_name.format('right_SMM_model', save_name))

    def load_model(self, save_name: str='final') -> bool:
        if os.path.exists(self.save_dir + self.save_name.format('SMM_Detector', 'leaf_node')):
            self.SMM_Detector.load(self.save_dir + self.save_name.format('SMM_Detector', 'leaf_node'))
            self.is_leaf = True
            self.SMM_Detector.to(device=self.device)
            return True
        if not os.path.exists(self.save_dir + self.save_name.format('clustering', save_name)):
            print("loading failed: {}".format(self.save_dir + f'/{save_name}'))
            return False
        print("loading: {}".format(self.save_dir + f'/{save_name}'))
        self.cluster_model.load(self.save_dir + self.save_name.format('clustering', save_name))
        self.SMM_Detector.load(self.save_dir + self.save_name.format('SMM_Detector', 'SMM_Detector'))

        self.left_model.load(self.save_dir + self.save_name.format('left_SMM_model', save_name))
        self.right_model.load(self.save_dir + self.save_name.format('right_SMM_model', save_name))
        
        self.left_model.to(device=self.device)
        self.right_model.to(device=self.device)
        self.cluster_model.to(device=self.device)
        self.SMM_Detector.to(device=self.device)
        return True
    
    def get_current_clusters(self, le_pred=None, ri_pred=None, **kwargs):
        current_id = kwargs['train_id']
        flag = False
        if self.initialize_strategy == 'random':
            cluster_label = np.zeros(len(self.y_train))
            if le_pred is None or ri_pred is None:
                cluster_label[np.random.choice(len(cluster_label), size=len(cluster_label) // 2, replace=False)] = 1
            else:
                le_residual = np.abs(le_pred[:, 1] - self.y_train)
                ri_residual = np.abs(ri_pred[:, 1] - self.y_train)
                cluster_label[le_residual > ri_residual] = 1
            flag = True
        if not flag:
            # ending
            return True
        if sum(cluster_label) < 10 or len(cluster_label) - sum(cluster_label) < 10:
            # ending
            return True
        # training cluster model
        if os.path.exists(self.save_dir + self.save_name.format('clustering', current_id)):
            self.cluster_model.load(self.save_dir + self.save_name.format('clustering', current_id))
            self.cluster_model.to(self.device)
        else:
            self.cluster_model.fit(self.X_train, cluster_label, save_dir=self.save_dir, save_fig_name='cluster_loss')
            self.cluster_model.save(self.save_dir + self.save_name.format('clustering', current_id))
        return False

    def check(self, test_data: pd.DataFrame=None) -> tuple[dict, bool]:
        # check the testing results in these two level data
        if test_data is not None:
            X_val = test_data.drop('SMM', axis=1)
            y_val = test_data['SMM']
        else:
            X_val = self.val_data.drop('SMM', axis=1)
            y_val = self.val_data['SMM']
        val_pred = self.SMM_Detector.predict_proba(X_val)[:, 1]
        current_results = binary_classification_metrics(val_pred, y_val)
        if self.is_leaf:
            return current_results, val_pred, True

        pred_cluster_label = self.cluster_model.predict(X_val)
        le_val_indices, ri_val_indices = np.where(pred_cluster_label == 0), np.where(pred_cluster_label == 1)
        le_val_data = self.val_data.iloc[le_val_indices]
        ri_val_data = self.val_data.iloc[ri_val_indices]

        val_pred = np.zeros(len(y_val))
        le_val_pred, ri_val_pred = self.left_model.predict_proba(le_val_data.drop('SMM', axis=1)), self.right_model.predict_proba(ri_val_data.drop('SMM', axis=1))
        val_pred[le_val_indices] = le_val_pred[:, 1]
        val_pred[ri_val_indices] = ri_val_pred[:, 1]

        current_results = binary_classification_metrics(val_pred, y_val)
        return current_results, val_pred, current_results['Precision'] > 0.5 and current_results['Recall'] > 0.5

    def save_leaf(self, save_name: str):
        # only save cluster model (leaf model)
        self.SMM_Detector.save(self.save_dir + self.save_name.format('SMM_Detector', save_name))

    def leaf_check(self):
        # first training with residual (low and hig)
        # get training data residual info
        if os.path.exists(self.save_dir + self.save_name.format('SMM_Detector', 'SMM_Detector')):
            print('Detector loading')
            self.SMM_Detector.load(self.save_dir + self.save_name.format('SMM_Detector', 'SMM_Detector'))
            self.SMM_Detector.to(self.device)
        else:
            print('Detector fitting')
            self.SMM_Detector.fit(self.X_train, self.y_train, save_dir=self.save_dir, save_fig_name='SMM_Detector_loss',
                                  test_X=self.val_data.drop('SMM', axis=1), test_y=self.val_data['SMM'])
            self.save_leaf(save_name='SMM_Detector')
        val_pred = self.SMM_Detector.predict_proba(self.val_data.drop('SMM', axis=1))[:, 1]
        y_val = self.val_data['SMM']
        
        current_results = binary_classification_metrics(val_pred, y_val)
        if current_results['Precision'] > 0.5 and current_results['Recall'] > 0.5:
            self.is_leaf = True
        self.SMM_Detector_results = current_results

    def fit(self):
        self.is_leaf = False
        self.leaf_check()
        results_f = open(self.save_dir + "/results_iter.txt", 'a+', buffering=1)
        results_f.write("iter={}: {}\n".format('leaf_node', str(self.SMM_Detector_results)))
        if self.is_leaf:
            self.save_leaf('leaf_node')
            results_f.close()
            return 

        le_pred = None
        ri_pred = None
        self.early_stopping = EarlyStoppingCallback(3, 1e-4, verbose=True, save_best_name=True)
        for _ in range(self.max_iter):
            ending = self.get_current_clusters(train_id = str(_), le_pred = le_pred, ri_pred = ri_pred)
            if ending:
                self.save_leaf('leaf_node')
                self.is_leaf = True
                break
            train_pred_cluster_label = self.cluster_model.predict(self.X_train)
            val_pred_cluster_label = self.cluster_model.predict(self.val_data.drop('SMM', axis=1))
            le_data, ri_data = self.train_data.iloc[np.where(train_pred_cluster_label == 0)], self.train_data.iloc[np.where(train_pred_cluster_label == 1)]
            val_le_data, val_ri_data = self.val_data.iloc[np.where(val_pred_cluster_label == 0)], self.val_data.iloc[np.where(val_pred_cluster_label == 1)]
            
            results_f.write("iter={}: train_le_data={}/{}, train_ri_data={}/{}\n".format(_, le_data[le_data['SMM'] == 1].shape[0], len(le_data), 
                                                                             ri_data[ri_data['SMM'] == 1].shape[0], len(ri_data)))

            self.left_model.training_model(le_data, save_path=self.save_dir + self.save_name.format('left_SMM_model', str(_)),
                                           save_fig_name='left_model_loss', test_data=val_le_data, min_size=self.min_size)
            self.right_model.training_model(ri_data, save_path=self.save_dir + self.save_name.format('right_SMM_model', str(_)),
                                           save_fig_name='right_model_loss', test_data=val_ri_data, min_size=self.min_size)

            current_results, _data, is_over = self.check()
            results_f.write("iter={}: {}\n".format(_, current_results))
            self.is_over = is_over

            self.early_stopping(-current_results[self.retained_best_name], cur_name=str(_))
            if str(_) == self.early_stopping.best_name:
                self.save_model('best')
            if is_over or ending or (self.early_stopping.early_stop and self.early_stopping_setting):
                break
            
            le_pred = self.left_model.predict_proba(self.X_train)
            ri_pred = self.right_model.predict_proba(self.X_train)
        results_f.close()
        self.save_model('final')
        # load the best
        self.load_model('best')

        # erase the str(_) model:
        for _ in range(self.max_iter):
            current_path = self.save_dir + self.save_name.format('left_SMM_model', str(_))
            remove_file(current_path)
            current_path = self.save_dir + self.save_name.format('right_SMM_model', str(_))
            remove_file(current_path)
            current_path = self.save_dir + self.save_name.format('clustering', str(_))
            remove_file(current_path)

def train_dynamic_cluster(config: Config, only_test: bool=False):
    setup_seed(config.seed)
    DC_settings = config.Dynamic_cluster_settings
    initialize_strategy = DC_settings['initialize_strategy']
    retained_best_name = DC_settings['retained_best_name']
    only_test_load_model_name = DC_settings['only_test_load_model_name']
    load_previous = DC_settings['load_previous']
    cluster_settings = DC_settings['cluster_settings']
    classifier_settings = DC_settings['classifier_settings']

    save_dir = config.save_dir + "/{}/{}/".format(initialize_strategy, retained_best_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_data, val_data, test_data, _input_dim = get_dataset_from_config(config, os.path.dirname(os.path.abspath(__file__)) + r'/../../dataset/')
    Dynamic_model = DynamicCluster(train_data, val_data, save_dir, initialize_strategy, 
                                    only_test, only_test_load_model_name, max_iter=10, 
                                    _input_dim=_input_dim, device=config.device, load_previous=load_previous, early_stopping=True,
                                    retained_best_name=retained_best_name, NN_max_iters=config.epoch, 
                                    sub_tree_cls_name=classifier_settings['model_name'], sub_tree_cls_structure=classifier_settings['model_structure'], 
                                    tree_cluster_name=cluster_settings['model_name'], tree_cluster_structure=cluster_settings['model_structure'], 
                                    lr=config.lr, train_id=0)

    metrics, pred, _ = Dynamic_model.check(test_data)
    # save analysis
    if True:
        _, test_pred, _ = Dynamic_model.check()
        '''test dataset'''
        clustering_label_fake = Dynamic_model.cluster_model.predict(test_data.drop('SMM', axis=1))

        le_pred = Dynamic_model.left_model.predict_proba(test_data.drop('SMM', axis=1))[:, 1]
        ri_pred = Dynamic_model.right_model.predict_proba(test_data.drop('SMM', axis=1))[:, 1]

        test_data['le_pred'] = le_pred
        test_data['ri_pred'] = ri_pred
        test_data['Dynamic_cls_probs'] = test_pred
        test_data['clustering_label'] = clustering_label_fake

        save_feature = ['SMM', 'le_pred', 'ri_pred', 'Dynamic_cls_probs', 'clustering_label']
        test_data.to_csv(save_dir + "/test_analysis.csv", columns=save_feature)
        '''train dataset'''
        clustering_label_fake = Dynamic_model.cluster_model.predict(train_data.drop('SMM', axis=1))

        le_pred = Dynamic_model.left_model.predict_proba(train_data.drop('SMM', axis=1))[:, 1]
        ri_pred = Dynamic_model.right_model.predict_proba(train_data.drop('SMM', axis=1))[:, 1]

        train_data['le_pred'] = le_pred
        train_data['ri_pred'] = ri_pred
        # train_data['Dynamic_cls_probs'] = test_pred
        train_data['clustering_label'] = clustering_label_fake
        save_feature.pop(3)
        train_data.to_csv(save_dir + "/train_analysis.csv", columns=save_feature)

    return metrics