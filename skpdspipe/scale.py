import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class DFStandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        """Scikit-learn StandardScaler() wrapper to return pandas DataFrame.

        """
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        return pd.DataFrame(self.scaler.transform(X),
                            columns=X.columns,
                            index=X.index)
