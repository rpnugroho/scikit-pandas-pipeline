import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DFApplyFn(TransformerMixin, BaseEstimator):

    def __init__(self, name, fn):
        """Apply function to create new data.

        Args:
            name (str):
                Column name.
            fn (function):
                Function to apply.
        """
        self.name = name
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        return self.fn(X).to_frame(self.name)


class DFUseApplyFn(TransformerMixin, BaseEstimator):

    def __init__(self, use=True):
        """Add ability for hyperparameter tuning to use a pipeline result.

        Args:
            use (bool, optional):
                "True" pipeline output a result, "False" pipeline output nothing.
                Defaults to True.
        """
        self.use = use

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        if not self.use:
            return None
        return X


class DFSetDType(TransformerMixin, BaseEstimator):

    def __init__(self, dtype='int'):
        """Set DataFrame dtype.

        Args:
            dtype (str, optional):
                Data type.
                Defaults to 'int'.
        """
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        return X.astype(self.dtype)


class DFStringify(TransformerMixin, BaseEstimator):

    def __init__(self, fill_value=np.nan):
        """Convert float or int to string by adding "_" in front of data.

        Args:
            fill_value (str, optional):
                value for missing data.
                Defaults to np.nan.
        """
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X_ = "_" + X.astype(str)
        return X_.replace({'_nan': self.fill_value})
