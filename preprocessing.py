import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

class Preprocessor:
    def __init__(self):
        self.ohe = None
        self.ordinal_encoder = None
        self.scaler = None

    def fit_ohe(self, df):
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.ohe.fit(df)
        return self

    def transform_ohe(self, df):
        encoded = self.ohe.transform(df)
        cols = self.ohe.get_feature_names_out(df.columns)
        return pd.DataFrame(encoded, columns=cols, index=df.index)

    def fit_ordinal(self, df, orders_list):
        self.ordinal_encoder = OrdinalEncoder(categories=orders_list, handle_unknown='use_encoded_value', unknown_value=-1)
        self.ordinal_encoder.fit(df)
        return self

    def transform_ordinal(self, df):
        encoded = self.ordinal_encoder.transform(df)
        return pd.DataFrame(encoded, columns=df.columns, index=df.index)

    def fit_scaler(self, df):
        self.scaler = StandardScaler()
        self.scaler.fit(df)
        return self

    def transform_scaler(self, df):
        scaled = self.scaler.transform(df)
        return pd.DataFrame(scaled, columns=df.columns, index=df.index)