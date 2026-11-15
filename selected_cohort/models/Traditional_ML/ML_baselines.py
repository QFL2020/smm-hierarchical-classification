# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.
"""
Baseline definition:
1. Isolation Forest
2. SVM
3. Logistic regression
4. Decision Tree
5. LASSO
6. AdaBoost
7. SMOTE AdaBoost
"""
import sys
from sklearn.ensemble import IsolationForest
from sklearn import svm
sys.path.extend(['../../'])
from dataset.data_processing import MixedDataset

def ISO_Forest(train_dataset: MixedDataset, val_dataset: MixedDataset, test_dataset: MixedDataset, seed, 
               return_model: bool=False, estimators: int=100, contamination='auto', **args):
    print(contamination)
    clf = IsolationForest(contamination=contamination, n_estimators=estimators, random_state=seed)
    inputs = train_dataset._inputs
    clf.fit(inputs)

    y_pred = clf.predict(test_dataset._inputs)
    y_pred[y_pred == 1] = 2
    y_pred[y_pred == -1] = 1
    y_pred[y_pred == 2] = 0
    if return_model:
        return y_pred, clf

    return y_pred

def SVM(train_dataset: MixedDataset, val_dataset: MixedDataset, test_dataset: MixedDataset):
    clf = svm.SVC(kernel='rbf', C=10.0)
    clf.fit(train_dataset._inputs, train_dataset._targets)
    return clf.predict(test_dataset._inputs)

from sklearn.linear_model import LogisticRegression as LR
def Logistic_regression(train_dataset: MixedDataset, val_dataset: MixedDataset, test_dataset: MixedDataset, return_model: bool=False, **args):
    model = LR(penalty = 'l2', solver = 'saga', C = 0.7, max_iter = 100000).fit(train_dataset._inputs, train_dataset._targets)
    pred = model.predict(test_dataset._inputs)
    if return_model:
        return pred, model
    return pred

from sklearn import tree
def Decision_Tree(train_dataset: MixedDataset, val_dataset: MixedDataset, test_dataset: MixedDataset, 
                  return_model: bool=False, max_depth: int=None, **args):
    if max_depth == -1:
        max_depth = None
    model = tree.DecisionTreeClassifier(max_depth=max_depth).fit(train_dataset._inputs, train_dataset._targets)
    pred = model.predict(test_dataset._inputs)

    if return_model:
        return pred, model
    print(pred)
    return pred

from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.utils import resample
def LASSO(train_dataset: MixedDataset, val_dataset: MixedDataset, test_dataset: MixedDataset):
    model = Lasso(alpha=1).fit(train_dataset._inputs, train_dataset._targets)
    pred = model.predict(test_dataset._inputs)
    return pred

from sklearn.ensemble import AdaBoostClassifier
def AdaBoost(train_dataset: MixedDataset, val_dataset: MixedDataset, test_dataset: MixedDataset, 
             return_model: bool=False, estimators: int=100, max_depth: int=None, **args):
    if max_depth == -1:
        max_depth = None
    model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(max_depth=max_depth), 
                               n_estimators=estimators).fit(train_dataset._inputs, train_dataset._targets)
    pred = model.predict(test_dataset._inputs)

    if return_model:
        return pred, model
    return pred

from imblearn.over_sampling import SMOTE
def SMOTE_AdaBoost(train_dataset: MixedDataset, val_dataset: MixedDataset, test_dataset: MixedDataset):
    sm = SMOTE()
    X_train_sm, y_train_sm = sm.fit_resample(train_dataset._inputs, train_dataset._targets)
    model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(max_depth=5), 
                               n_estimators=50, random_state=42).fit(X_train_sm, y_train_sm)
    y_pred = model.predict(test_dataset._inputs)
    return y_pred

from sklearn.ensemble import RandomForestClassifier
def Random_Forest(train_dataset: MixedDataset, val_dataset: MixedDataset, test_dataset: MixedDataset, return_model: bool=False, estimators: int=100, **args):
    model = RandomForestClassifier(n_estimators=estimators).fit(train_dataset._inputs, train_dataset._targets)
    pred = model.predict(test_dataset._inputs)

    if return_model:
        return pred, model
    return pred

import xgboost as xgb
def XGBoost(train_dataset: MixedDataset, val_dataset: MixedDataset, test_dataset: MixedDataset, return_model: bool=False):
    model = xgb.XGBClassifier().fit(train_dataset._inputs, train_dataset._targets)
    pred = model.predict(test_dataset._inputs)

    if return_model:
        return pred, model
    return pred

