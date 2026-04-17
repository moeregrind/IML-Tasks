import numpy as np
import pandas as pd


def load_data():
    train_df = pd.read_csv("train.csv")
    train_df = train_df.dropna(subset=['price_CHF'])

    test_df = pd.read_csv("test.csv")

    X_train = train_df.drop(['price_CHF'], axis=1).to_numpy()
    y_train = train_df['price_CHF'].to_numpy()
    X_test = test_df.to_numpy()

    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer(transformers=[
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [0]),
        ("imputer", IterativeImputer(max_iter=10, random_state=0), [1, 2, 3, 4, 5, 6, 7, 8, 9])
    ])

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    assert (X_train_prep.shape[1] == X_test_prep.shape[1]) and \
           (X_train_prep.shape[0] == y_train.shape[0]) and \
           (X_test_prep.shape[0] == 100), "Invalid data shape"

    return X_train_prep, y_train, X_test_prep


class Model(object):
    def __init__(self):
        super().__init__()
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
        from sklearn.model_selection import cross_val_score

        kernels = [
            RBF() + WhiteKernel(),
            Matern() + WhiteKernel(),
            RationalQuadratic() + WhiteKernel(),
            DotProduct() + WhiteKernel(),
            RBF() + Matern() + WhiteKernel(),
        ]

        best_score = -np.inf
        best_kernel = None
        for kernel in kernels:
            gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           n_restarts_optimizer=5, random_state=0)
            scores = cross_val_score(gpr, X_train, y_train, cv=5, scoring='r2')
            mean_score = scores.mean()
            print(f"Kernel: {kernel}  ->  CV R2: {mean_score:.4f}")
            if mean_score > best_score:
                best_score = mean_score
                best_kernel = kernel

        print(f"\nBest kernel: {best_kernel}  (CV R2: {best_score:.4f})")
        best_model = GaussianProcessRegressor(kernel=best_kernel, normalize_y=True,
                                              n_restarts_optimizer=5, random_state=0)
        best_model.fit(X_train, y_train)
        self.model = best_model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict(X_test)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred


if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    model = Model()
    model.fit(X_train=X_train, y_train=y_train)
    y_pred = model.predict(X_test)
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")
