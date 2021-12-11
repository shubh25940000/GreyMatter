import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
# from dotenv import load_dotenv

""" This module will use the cleaned up dataset for training as per the ML model required"""


class train:
    def __init__(self, df, model_type, cols_encode, cols_hotencode, id_cols, target):
        self.df = df
        self.model_type = model_type
        self.cols_encode = cols_encode
        self.cols_hotencode = cols_hotencode
        self.id_cols = id_cols
        self.target = target

    def label_encoding(self):
        encoder = preprocessing.LabelEncoder()
        self.df.cols_encode = encoder.fit_transform(df.cols_encode)
        return self.df

    def hot_encoding(self):
        encoded = pd.get_dummies(data=self.df, columns=self.cols_hotencode)
        self.df = pd.concat([self.df.drop(self.cols_hotencode, axis=1), encoded], axis=1)
        return self.df

    def train_model(self):
        encoded_df = self.hot_encoding()
        df_train, df_test = train_test_split(encoded_df, train_size= 0.7, test_size= 0.3, random_state= 100)
        if len(self.id_cols) > 0:
            df_train.drop(self.id_cols, axis = 1, inplace = True)
            X_train = df_train.drop(self.target, axis=1)
            y_train = df_train[self.target]
        if self.model_type == 'Random_Forest':
            pass





if __name__ == "__main__":
    # load_dotenv('Files/Variables.env.env')
    df = pd.read_csv('Files/Sales_Clean.csv')
    cols_encode = ['Utilities','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond']
    cols_hotencode = ['MSZoning','Street','LandContour','LotConfig','Neighborhood','Condition1','Condition2',
                      'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                      'Foundation','Heating','Electrical','Functional','GarageType','GarageFinish','PavedDrive',
                      'SaleType','SaleCondition']
    T1 = train(df, 'LinearRegression', cols_encode, cols_hotencode)
    df = T1.hot_encoding()







