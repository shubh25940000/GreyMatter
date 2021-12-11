import pandas as pd
from sklearn import preprocessing
""" This module will use the cleanued up dataset for training as per the ML model required"""
if __name__ == "__main__":
    df = pd.read_csv('Sale_Clean.csv')

    class train:
        def __init__(self, df, model_type, cols_encode):
            self.df = df
            self.model_type = model_type
            self.cols_encode = cols_encode

        def label_encoding(self):
            encoder = preprocessing.LabelEncoder()
            self.df.cols_encode = encoder.fit_transform(df.cols_encode)



