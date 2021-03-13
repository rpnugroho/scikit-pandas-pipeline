import pandas as pd
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   MaxAbsScaler, RobustScaler, Normalizer)
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


class DFScaler(BaseEstimator, TransformerMixin):

    def __init__(self, scaler_type='standard'):
        """Scikit-learn scaler wrapper to return pandas DataFrame.

        Args:
            scaler_type (str, optional):
                Scaler type 'standard', 'minmax', 'maxabs', 'robust', and 'normalizer'.
                Defaults to 'standard'.
        """
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'maxabs': MaxAbsScaler(),
            'robust': RobustScaler(),
            'normalizer': Normalizer()
        }
        self.scaler = scalers[scaler_type]

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        return pd.DataFrame(self.scaler.transform(X),
                            columns=X.columns,
                            index=X.index)
