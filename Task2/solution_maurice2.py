import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, WhiteKernel
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer

train_df = pd.read_csv("train.csv").dropna(subset = ['price_CHF'])
test_df = pd.read_csv("test.csv")

X_train = train_df.drop(['price_CHF'], axis = 1).to_numpy()
y_train = train_df['price_CHF'].to_numpy()
X_test = test_df.to_numpy()

#preprocessing and feature imputation
    
preprocessor = ColumnTransformer(transformers= [("encoder", preprocessing.OrdinalEncoder(), [0]), 
                                                ("imputer", IterativeImputer(max_iter = 10, random_state=0), [1,2,3,4,5,6,7,8,9])])
    

#build pipline and fit the model
pipe = Pipeline(steps=[('preprocessor', preprocessor), 
                       ('model', GaussianProcessRegressor(kernel=  RationalQuadratic() + Matern(), random_state= 0))])

param_grid = [
    {
        'preprocessor__imputer': [IterativeImputer(max_iter = 10, random_state=0), SimpleImputer(strategy='mean'), SimpleImputer(strategy='median')],
    },
    {
        'model': [GaussianProcessRegressor()],
        'model__kernel': [RationalQuadratic(), Matern(), DotProduct(), RBF(), WhiteKernel(), RationalQuadratic() + Matern(), RBF() + WhiteKernel()], 
        'model__random_state': [0],
        'model__alpha': [1e-10, 1e-5, 1e-2]
    }
]

grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)   


#predicting the results and saving them in a csv file

y_pred = pipe.predict(X_test)
assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"


dt = pd.DataFrame(y_pred) 
dt.columns = ['price_CHF']
dt.to_csv('results.csv', index=False)
print("\nResults file successfully generated!") 