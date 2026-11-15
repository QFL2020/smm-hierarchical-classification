# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
# get clearned data from raw data
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import sys, torch, os
sys.path.extend(['../'])
from dataset.selected_features import get_SMM_features
from settings.config import Config
from dataset.sampler import undersampling_or_oversampling

from models.NN import Multi_MLP_Cat_Num
from models.Residual_factory.residual_block import ResidualNN

class MixedDataset(Dataset):
    def __init__(self, inputs, targets, cat_length, cot_length, device: str, dtype=torch.float):
        self._inputs = inputs
        self._targets = targets
        
        self.device = device
        self.dtype = dtype

        self.cat_length = cat_length
        self.cot_length = cot_length
        
        # print(inputs)
        assert self.cat_length + self.cot_length == len(inputs[0]), \
        'Error: diff feature length! ([{}]+[{}]!=[{}])'.format(self.cat_length, self.cot_length, len(inputs[0]))
        
        self.cat_inputs = torch.tensor(inputs[:, :self.cat_length], dtype=torch.long, device=device)
        self.cot_inputs = torch.tensor(inputs[:, -self.cot_length:], dtype=dtype, device=device)
        self.targets = torch.tensor(targets, dtype=dtype, device=device).unsqueeze(1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.cat_inputs[idx], self.cot_inputs[idx], self.targets[idx]

class FeatureDataset(Dataset):
    def __init__(self, inputs, targets, device: str, dtype=torch.float):
        self._inputs = inputs
        self._targets = targets
        
        self.device = device
        self.dtype = dtype
        
        self.inputs = torch.tensor(inputs, dtype=dtype, device=device)
        self.targets = torch.tensor(targets, dtype=dtype, device=device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def get_cleaned_data(dir_path: str='./', target_column: str='SMM', feature_type: str='default'):
    """
    Filtering original data:
    Healthy <==> Non-SMM
    Unhealthy <==> SMM
    1. SMM and Non-SMM with the same features => SMM data
    2. Deduplication 
    """
    # original data path
    data_path = os.environ['SMM_DATA_CSV']  # Authorized prepared CSV outside the repository
    SMM_features = get_SMM_features(feature_type)
    categorical_vars = SMM_features.categorical_vars
    continuous_vars = SMM_features.continuous_vars

    data = pd.read_csv(data_path, low_memory=False)[categorical_vars + continuous_vars + [target_column]]
    # Healthy / Unhealthy data and remove duplicates separately
    healthy_data = data[data[target_column] == 0].drop(target_column, axis=1)
    unhealthy_data = data[data[target_column] == 1].drop(target_column, axis=1)
    # Identify common rows between healthy and unhealthy data, removing common data from healthy
    # common_data = pd.merge(healthy_data.drop_duplicates(), unhealthy_data.drop_duplicates())
    common_data = pd.merge(healthy_data.drop_duplicates(), unhealthy_data.drop_duplicates())
    healthy_data = healthy_data.merge(common_data, how='left', indicator=True)
    healthy_data = healthy_data[healthy_data['_merge'] == 'left_only'].drop(columns=['_merge'])
    common_data = pd.merge(healthy_data.drop_duplicates(), unhealthy_data.drop_duplicates())

    # Combine cleaned datasets
    healthy_data[target_column] = 0
    unhealthy_data[target_column] = 1
    cleaned_data = pd.concat([healthy_data, unhealthy_data])

    # X = cleaned_data.drop(target_column, axis=1)
    # y = cleaned_data[target_column]
    return cleaned_data

def get_train_val_test(cleaned_data, config: Config=None, features_name: str="default", dtype=torch.float64):
    if config.SMM_label not in ["SMM", "SMM_GROUP1", "SMM_GROUP2", "SMM_GROUP3", "SMM_GROUP4", 
                                "SMM_GROUP5", "SMM_GROUP6", "SMM_GROUP7"]:
        assert False, 'Error label_name'

    cat_encoder, cot_encoder = config.dataset[1:3]
    SMM_features = get_SMM_features(features_name)
    categorical_vars = SMM_features.categorical_vars
    continuous_vars = SMM_features.continuous_vars
    label_name = config.SMM_label
    selected_vars = categorical_vars + continuous_vars + [label_name]

    # read data
    data = cleaned_data[selected_vars]
    cat_data = data[categorical_vars]
    cat_list = [len(cat_data[col].unique()) for col in cat_data.columns]

    # categorical encoder
    if cat_encoder == 'Binary_Encoder':
        scaler = ce.BinaryEncoder(cols=categorical_vars)
        X_categorical_encoded = scaler.fit_transform(pd.DataFrame(data[categorical_vars])).values
    elif cat_encoder == 'Label_Encoder':
        from sklearn.preprocessing import LabelEncoder
        from collections import defaultdict
        d = defaultdict(LabelEncoder)
        X_categorical_encoded = pd.DataFrame(data[categorical_vars])
        X_categorical_encoded = X_categorical_encoded.apply(lambda x: d[x.name].fit_transform(x)).values
        scaler = d
    else:
        assert False, 'Error, don\'t found {} categorical encoder setting.'.format(cat_encoder)
    cat_scaler = scaler

    # encoder cot
    if cot_encoder == 'Standard_Scaler':
        scaler = StandardScaler()
        X_numerical_normalized = scaler.fit_transform(data[continuous_vars])
    else:
        assert False, 'Error, don\'t found {} continuous encoder setting.'.format(cat_encoder)
    cot_scaler = scaler
    print("Encoder type: Category={}, Continuous={}".format(cat_encoder, cot_encoder))

    cat_length, cot_length = len(X_categorical_encoded[0]), len(X_numerical_normalized[0])
    # combine
    _inputs = np.concatenate([X_categorical_encoded, X_numerical_normalized], axis=-1)
    _targets = np.array(data[label_name])

    print('current inputs.shape={}'.format(_inputs.shape))
    print('current targets.shape={}'.format(_targets.shape))

    train_inputs, test_inputs, train_targets, test_targets, training_indices, testing_indices = train_test_split(
        _inputs, _targets, list(range(len(_inputs))), test_size=0.20, random_state=config.seed, stratify=_targets)
    # under sampling or over sampling for training data
    train_inputs, train_targets = undersampling_or_oversampling(config, 
            train_inputs, train_targets, categorical_features=list(range(cat_length)))
    
    print('train dataset: SMM samples={}, total samples={}'.format(sum(train_targets), len(train_targets)))
    print('test dataset: SMM samples={}, total samples={}'.format(sum(test_targets), len(test_targets)))

    train_dataset = MixedDataset(train_inputs, train_targets, cat_length, cot_length, config.device, dtype=dtype)
    test_dataset = MixedDataset(test_inputs, test_targets, cat_length, cot_length, config.device, dtype=dtype)
    val_dataset = test_dataset

    return train_dataset, val_dataset, test_dataset, SMM_features.cat_list

def get_feature_dataset(dataset: MixedDataset, model, **kwargs) -> pd.DataFrame:
    """
    Extract feature embeddings from the model and return them as a DataFrame.

    Args:
        dataset (MixedDataset): The dataset containing categorical and continuous inputs.
        model (torch.nn.Module): The trained model used to extract embeddings.
        **kwargs: Additional arguments (if needed).

    Returns:
        pd.DataFrame: A DataFrame with feature embeddings and the target column 'SMM'.
    """
    # Ensure the model is in evaluation mode and disable gradient computation
    with torch.no_grad():
        model.eval()
        # Extract embeddings from the model
        embedding = model.embedding_layers(dataset.cat_inputs, dataset.cot_inputs)
        
        # Pass embeddings through the appropriate layers based on the model type
        if isinstance(model, ResidualNN):
            feature_input = model.deep_layers(embedding)
        elif isinstance(model, Multi_MLP_Cat_Num):
            feature_input = model.stacking_layers(embedding)
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        
        # Extract targets
        targets = dataset.targets

        # Convert tensors to numpy arrays
        feature_input = feature_input.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

    # Create a DataFrame for the feature embeddings
    feature_columns = [str(i) for i in range(feature_input.shape[1])]
    features_df = pd.DataFrame(feature_input, columns=feature_columns)
    features_df['SMM'] = targets
    return features_df

def MixedDataset_transform(dataset: MixedDataset) -> pd.DataFrame:
    with torch.no_grad():
        # Convert tensors to numpy arrays
        inputs = np.array(dataset._inputs)
        targets = np.array(dataset._targets)

    # Create a DataFrame for the feature embeddings
    feature_columns = [str(i) for i in range(inputs.shape[1])]
    features_df = pd.DataFrame(inputs, columns=feature_columns)
    features_df['SMM'] = targets
    return features_df

def get_train_val_test_NNfeature(train_dataset: MixedDataset, val_dataset: MixedDataset, test_dataset: MixedDataset, cat_list, pretrained_model_name: str, config: Config):
    pretrained_path = os.path.dirname(os.path.abspath(__file__)) + f'/../models/pretrained_model/{pretrained_model_name}/{config.seed}/{pretrained_model_name}.pth'
    if pretrained_model_name == 'MLP3':
        model = Multi_MLP_Cat_Num(cat_list, train_dataset.cot_length, 
                                  Stacking_NN=[128, 256, 512], MLP_NN=[512, 256, 128], p=0.0)
        input_dim = 512
    elif pretrained_model_name == 'ResMLP3':
        model = ResidualNN(cat_list, train_dataset.cot_length, stack_layers=[192, 256], 
                           act="ReLu", p=0.0)
        input_dim = 256
    else:
        assert False, "Error: pretrained model name must set as ['MLP3', 'ResMLP3']"

    model.load_state_dict(torch.load(pretrained_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    return get_feature_dataset(train_dataset, model), get_feature_dataset(val_dataset, model), get_feature_dataset(test_dataset, model), input_dim


