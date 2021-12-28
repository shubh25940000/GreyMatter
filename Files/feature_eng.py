from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
"""This class will use the cleaned up dataset for feature engineering"""


class feature_eng:
    def __init__(self, df, cols_encode, cols_hotencode, id_cols, num_cont, scaler='MinMaxScaler'):
        self.df = df
        self.cols_encode = cols_encode
        self.cols_hotencode = cols_hotencode
        self.id_cols = id_cols
        self.num_cont = num_cont
        self.scaler = scaler

    def label_encoding(self):
        encoder = preprocessing.LabelEncoder()
        df = self.df
        df[self.cols_encode] = df[self.cols_encode].apply(encoder.fit_transform)
        return df

    def hot_encoding(self):
        data = self.label_encoding()
        encoded = pd.get_dummies(data=data, columns=self.cols_hotencode)
        return encoded

    def scale(self):
        if self.scaler == 'MinMaxScaler':
            scaler_o = MinMaxScaler()
            df_scaled = self.hot_encoding()
            df_scaled[self.num_cont] = scaler_o.fit_transform(df_scaled[self.num_cont])
            return df_scaled
        else:
            scaler_o = StandardScaler()
            df_scaled = self.hot_encoding()
            df_scaled[self.num_cont] = scaler_o.fit_transform(df_scaled[self.num_cont])
            return df_scaled
