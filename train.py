import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score, KFold

import os
# from dotenv import load_dotenv

""" This module will use the cleaned up dataset for training as per the ML model required"""


class train:
    def __init__(self, df, model_type, cols_encode, cols_hotencode, id_cols, target, params, iter, scoring, gridsearch = True):
        self.df = df
        self.model_type = model_type
        self.cols_encode = cols_encode
        self.cols_hotencode = cols_hotencode
        self.id_cols = id_cols
        self.target = target
        self.params = params
        self.gridsearch = gridsearch
        self.iter = iter
        self.scoring = scoring

    def label_encoding(self):
        encoder = preprocessing.LabelEncoder()
        self.df[cols_encode] = self.df[cols_encode].apply(encoder.fit_transform)
        return self.df

    def hot_encoding(self):
        encoded = pd.get_dummies(data=self.label_encoding(), columns=self.cols_hotencode)
        self.df = pd.concat([self.df.drop(self.cols_hotencode, axis=1), encoded], axis=1)
        return self.df

    def hyperparameter_tuning(self, model):
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        cv_inner = KFold(n_splits=3, shuffle= True, random_state=42)
        if self.gridsearch == False:
            RandomSearchCV = RandomizedSearchCV(estimator=model, param_distributions=self.params,
                                                cv=cv_inner, verbose=1, n_jobs=8, n_iter=self.iter, refit=True)
            return RandomSearchCV
        else:
            GridSearchCV = GridSearchCV(estimator=model, param_grid=self.params,
                                        cv=4, verbose=1, n_jobs=8, scoring=self.scoring, refit= True)
            return GridSearchCV

    def cross_val_scorer(self, X_train, y_train, search):
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)
        score = cross_val_score(estimator = search, X=X_train, y=y_train, scoring= self.scoring, n_jobs=8,cv = cv_outer )
        return score



    def train_model(self):
        encoded_df = self.hot_encoding()
        df_train, df_test = train_test_split(encoded_df, train_size= 0.7, test_size= 0.3, random_state= 100)
        if len(self.id_cols) > 0:
            df_train.drop(self.id_cols, axis = 1, inplace = True)
        X_train = df_train.drop(self.target, axis=1)
        y_train = df_train[self.target]
        if self.model_type == 'Random_Forest':
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(random_state=42, n_jobs= 8)
            search = self.hyperparameter_tuning(rf)
            score = self.cross_val_scorer(X_train, y_train, search)
            print(np.sqrt(score))
            return search.best_estimator_

if __name__ == "__main__":
    # load_dotenv('Files/Variables.env.env')
    df = pd.read_csv('Files/Sales_Clean.csv')
    cols_encode = ['LotShape', 'Utilities','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond', 'BsmtExposure',
                   'BsmtFinType1','BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 'GarageQual', 'GarageCond']
    cols_hotencode = ['MSZoning','Street','LandContour','LotConfig','Neighborhood','Condition1','Condition2',
                      'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                      'Foundation','Heating','Electrical','Functional','GarageType','GarageFinish','PavedDrive',
                      'SaleType','SaleCondition']
    idcols = ['Id']
    target = 'SalePrice'
    max_depth = [int(x) for x in np.linspace(10, 60, 2)]
    n_estimator = [int(x) for x in np.linspace(4000, 5000, 2)]
    min_samples_split = [int(x) for x in np.linspace(10, 100, 4)]
    min_samples_leaf = [int(x) for x in range(10,15)]
    model_type = ['Random_Forest', 'Lasso_Regression', 'Elasti_Net', 'Gradient_boosting']
    params = {'max_depth': max_depth,
          'n_estimators': n_estimator,
          'min_samples_split': min_samples_split,
           'min_samples_leaf': min_samples_leaf}
    T1 = train(df, 'Random_Forest', cols_encode, cols_hotencode, idcols,target, params, 100, 'neg_mean_squared_error', gridsearch=False)
    df = T1.train_model()







