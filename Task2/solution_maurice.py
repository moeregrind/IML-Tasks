import numpy as np
import pandas as pd




def load_data(max_iter_imp = 10):
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    # if the price_CHF value is missing, we discard the data point
    train_df = train_df.dropna(subset = ['price_CHF'])

    # Load test data
    test_df = pd.read_csv("test.csv")

    #------------------ preprocessing and feature imputation ----------------------------------#
    
    X_train = train_df.drop(['price_CHF'], axis = 1).to_numpy()
    y_train = train_df['price_CHF'].to_numpy()
    X_test = test_df.to_numpy()
    
    
    
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn import preprocessing
    from sklearn.compose import ColumnTransformer
    
    preprocessor = ColumnTransformer(transformers= [("encoder", preprocessing.OrdinalEncoder(), [0]), 
                                                    ("imputer", IterativeImputer(max_iter = max_iter_imp, random_state=0), [1,2,3,4,5,6,7,8,9])])
    
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    #------------------------------------------------------------------------#
 
    assert (X_train_prep.shape[1] == X_test_prep.shape[1]) and (X_train_prep.shape[0] == y_train.shape[0]) and (X_test_prep.shape[0] == 100), "Invalid data shape"
    return X_train_prep, y_train, X_test_prep


class Model(object):
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None
        self.model = None
        

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        
        self._x_train = X_train
        self._y_train = y_train
        
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, WhiteKernel
        
        #weights are adjusted such that we automatically select the best kernel
        gpr = GaussianProcessRegressor(kernel=  RBF(), random_state= 0)
       
        gpr.fit(X_train, y_train)
        
        print("GPR Score")
        print(gpr.score(X_train, y_train))
        self.model = gpr
        
        

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        
        y_pred= self.model.predict(X_test)
        
        
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = load_data()
    model = Model()
    # Use this function to fit the model
    model.fit(X_train=X_train, y_train=y_train)
    # Use this function for inference
    y_pred = model.predict(X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

