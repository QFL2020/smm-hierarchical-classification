# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
from torch.utils.data import Sampler
import numpy as np
import sys
sys.path.extend(['../'])
from settings.config import Config

class BalancedSampler(Sampler):
    def __init__(self, dataset, batch_size, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        self.indices = np.arange(len(dataset))
        self.labels = dataset.targets.cpu().detach().numpy()[self.indices].copy().reshape(-1)

    def __iter__(self):
        positive_indices = np.where(self.labels == 1)[0]
        negative_indices = np.where(self.labels == 0)[0]

        all_batches = []
        negative_replace = False
        if len(negative_indices) < self.batch_size - self.batch_size // 2 - 1:
            negative_replace = True
        for _ in range((len(self.dataset) + self.batch_size - 1) // self.batch_size):
            pos_sample_count = self.batch_size // 2 + 1 # half: positive samples
            neg_sample_count = self.batch_size - pos_sample_count # half: negative samples

            pos_samples = np.random.choice(positive_indices, pos_sample_count, replace=True)
            neg_samples = np.random.choice(negative_indices, neg_sample_count, replace=negative_replace)

            batch = np.concatenate([pos_samples, neg_samples])
            np.random.shuffle(batch)
            all_batches.extend(batch)
        
        return iter(all_batches)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size * self.batch_size


from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

def undersampling_or_oversampling(config: Config, _inputs, _targets, categorical_features: list[int]=None):
    train_inputs = _inputs
    train_targets = _targets
    if config is None:
        return train_inputs, train_targets
    print("---------{} Before-----------".format(config.dataset[0]))
    print(len(train_inputs), sum(train_targets))

    # hybrid
    if config.Sampling_Strategy[0] == 'SMOTE-Tomek':
        kos = SMOTETomek(random_state=config.seed)  # oversampling and undersampling
        train_inputs, train_targets = kos.fit_resample(train_inputs, train_targets)
    elif config.Sampling_Strategy[0] == 'SMOTE+ENN':
        enn = SMOTEENN(random_state=config.seed)
        train_inputs, train_targets = enn.fit_resample(train_inputs, train_targets)

    # oversampling
    elif config.Sampling_Strategy[0] == 'SMOTE':
        smote  = SMOTE(random_state=config.seed)
        train_inputs, train_targets = smote.fit_resample(train_inputs, train_targets)
    elif config.Sampling_Strategy[0] == 'ADASYN':
        adasyn = ADASYN(random_state=config.seed)
        train_inputs, train_targets = adasyn.fit_resample(train_inputs, train_targets)

    # undersampling
    elif config.Sampling_Strategy[0] == 'RandomUndersampling':
        rus = RandomUnderSampler(random_state=config.seed)
        train_inputs, train_targets = rus.fit_resample(train_inputs, train_targets)
    elif config.Sampling_Strategy[0] == 'Cluster-Centroids':
        cc = ClusterCentroids(random_state=config.seed)
        train_inputs, train_targets = cc.fit_resample(train_inputs, train_targets)

    print("---------{} After-----------".format(config.Sampling_Strategy[0]))
    print(len(train_inputs), sum(train_targets))
    return train_inputs, train_targets
