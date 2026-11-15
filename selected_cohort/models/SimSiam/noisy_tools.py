# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
import numpy as np
import torch

# 1. Category replacement (One-hot type)
def category_replace(one_hot_data, replace_prob=0.1):
    """Randomly replace some categories in one-hot encoding"""
    new_data = one_hot_data.copy()
    for i in range(len(new_data)):
        if np.random.rand() < replace_prob:
            # Randomly select a category for replacement
            new_class = np.random.randint(0, one_hot_data.shape[1])
            new_data[i] = np.eye(one_hot_data.shape[1])[new_class]
    return new_data

# 2. Category noise
def category_noise(one_hot_data, noise_prob=0.05):
    """Randomly flip parts of the one-hot encoding to add noise"""
    new_data = one_hot_data.copy()
    for i in range(len(new_data)):
        if np.random.rand() < noise_prob:
            flip_index = np.random.randint(0, one_hot_data.shape[1])
            new_data[i] = 0
            new_data[i, flip_index] = 1
    return new_data

# 3. Add noise to continuous data
def add_noise(continuous_data, noise_std=0.1):
    """Add Gaussian noise"""
    noise = torch.normal(0, noise_std, size=continuous_data.shape, device=continuous_data.device)
    return continuous_data + noise

# 4. Scale continuous data
def scale_features(continuous_data, scale_range=(0.9, 1.1)):
    """Randomly scale the feature data"""
    scale_factor = torch.empty(1, device=continuous_data.device).uniform_(*scale_range)
    return continuous_data * scale_factor

# 5. Feature dropout
def feature_dropout(continuous_data, dropout_prob=0.2):
    """Randomly set some feature values to zero"""
    mask = torch.rand(continuous_data.shape, device=continuous_data.device) > dropout_prob
    return continuous_data * mask

# 6. Mixup (for continuous data and labels)
def mixup(data, labels, alpha=0.2):
    """Mixup data augmentation method"""
    lam = np.random.beta(alpha, alpha)
    batch_size = data.shape[0]
    index = np.random.permutation(batch_size)
    mixed_data = lam * data + (1 - lam) * data[index, :]
    mixed_labels = lam * labels + (1 - lam) * labels[index, :]
    return mixed_data, mixed_labels

"""
# Test code
new_one_hot_data = category_replace(one_hot_data, replace_prob=0.9)
print("One-hot data after replacement:\n", new_one_hot_data)

noisy_one_hot_data = category_noise(one_hot_data, noise_prob=0.9)
print("One-hot data after adding noise:\n", noisy_one_hot_data)

noisy_continuous_data = add_noise(continuous_data, noise_std=0.1)
print("Continuous data after adding noise:\n", noisy_continuous_data)

scaled_continuous_data = scale_features(continuous_data, scale_range=(0.9, 1.1))
print("Continuous data after scaling:\n", scaled_continuous_data)

dropped_continuous_data = feature_dropout(continuous_data, dropout_prob=0.2)
print("Continuous data after feature dropout:\n", dropped_continuous_data)
"""