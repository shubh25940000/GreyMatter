import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import multiprocessing

n_cpus = multiprocessing.cpu_count()
# from dotenv import load_dotenv

"""This module will use the cleaned up dataset for training as per the ML model required"""


class train:
    def __init__(self, model_type, X_train, y_train, params, itere, scoring, gridsearch=True):
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.params = params
        self.gridsearch = gridsearch
        self.itere = itere
        self.scoring = scoring

    def hyperparameter_tuning(self, model):
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        cv_inner = KFold(n_splits=4, shuffle=True, random_state=42)
        if self.gridsearch == False:
            RandomSearchCV = RandomizedSearchCV(estimator=model, param_distributions=self.params,
                                                cv=cv_inner, verbose=1, n_jobs=-1, n_iter=self.itere, refit=True)
            return RandomSearchCV
        else:
            GridSearchCV = GridSearchCV(estimator=model, param_grid=self.params,
                                        cv=4, verbose=1, n_jobs=-1, scoring=self.scoring, refit=True)
            return GridSearchCV

    def cross_val_scorer(self, X_train, y_train, search):
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=42)
        score = cross_val_score(estimator=search, X=X_train, y=y_train, scoring=self.scoring, n_jobs=-1, cv=cv_outer)
        return score

    def train_model(self):
        X_train = self.X_train
        y_train = self.y_train
        if self.model_type == 'Random_Forest':
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            search = self.hyperparameter_tuning(rf)
            search.fit(X_train, y_train)
            score = self.cross_val_scorer(X_train, y_train, search)
            print(np.average(score))
            return search.best_estimator_
        if self.model_type == 'Lasso':
            from sklearn.linear_model import LassoCV
            las = LassoCV(cv=4, random_state=42, alphas=[int(i) for i in np.linspace(0.001, 500, 5000)]).fit(X_train,
                                                                                                             y_train)
            return las
        if self.model_type == 'Ridge':
            from sklearn.linear_model import RidgeCV
            ridge = RidgeCV(cv=4, alphas=[int(i) for i in np.linspace(0.001, 500, 5000)]).fit(X_train, y_train)
            return ridge
        if self.model_type == 'SimpleLinearRegression':
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            return lr


if __name__ == "__main__":
    # load_dotenv('Files/Variables.env.env')
    df = pd.read_csv('../../Files/Sales_Clean.csv')
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







