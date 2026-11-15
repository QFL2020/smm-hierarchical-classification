"""Fitting-only categorical encoding and continuous standardization."""
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from paths import feature_names
class TrainPreprocessor:
    def __init__(self,cat,num):self.cat=cat;self.num=num
    def fit(self,data):
        self.encoder=ce.BinaryEncoder(cols=self.cat,return_df=True)
        self.encoder.fit(data[self.cat]);self.scaler=StandardScaler().fit(data[self.num])
        self.n_binary=len(self.encoder.get_feature_names_out())
        # Embedding_layers uses only the sum of per-entry binary widths. Derive
        # that width from the trained encoder rather than full-cohort counts.
        self.input_dim=[[1]*self.n_binary,len(self.num)]
        self.columns=[str(i) for i in range(self.n_binary+len(self.num))]
        return self
    def transform(self,data):
        x=np.c_[self.encoder.transform(data[self.cat]).to_numpy(dtype=float),self.scaler.transform(data[self.num])]
        assert np.isfinite(x).all()
        return pd.DataFrame(x,columns=self.columns)
