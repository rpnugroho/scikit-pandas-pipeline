import pandas as pd
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin


class DFPCA(BaseEstimator, TransformerMixin):

    def __init__(self,
                 n_components=5,
                 drop_raw=False,
                 prefix='pca_'):
        """Scikit-learn PCA() wrapper to return pandas DataFrame.

        Args:
            n_components (int, optional):
                Number of components to create. 
                Defaults to 5.
            drop_raw (bool, optional):
                If True raw columns will be droped. 
                Defaults to False.
            prefix (str, optional):
                Prefix for column name.
                Defaults to 'pca_'.
        """
        self.n_components = n_components
        self.prefix = prefix
        self.drop_raw = drop_raw
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        pca = pd.DataFrame(self.pca.transform(X),
                           index=X.index).add_prefix(self.prefix)
        if self.drop_raw:
            return pca
        X_ = pd.concat([X, pca], axis=1)
        return X_
