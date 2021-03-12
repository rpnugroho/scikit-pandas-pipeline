# TODO: Add RareImputer => Impute rare value with another value.
# TODO: Add MissingIndicator
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class DFSimpleImputer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 missing_values=np.nan,
                 strategy='constant',
                 fill_value=None):
        """Scikit-learn SimpleImputer() wrapper to return pandas DataFrame.

        Args:
            missing_values (str or int, optional):
                String or numerical value.
                Defaults to np.nan.
            strategy (str, optional):
                The imputation strategy ('mean', 'median', 'most_frequent', or 'constant').
                Defaults to 'constant'.
            fill_value (str or int, optional):
                String or numerical value.
                Defaults to None.
        """
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = SimpleImputer(missing_values=self.missing_values,
                                     strategy=self.strategy,
                                     fill_value=self.fill_value)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        return pd.DataFrame(self.imputer.transform(X),
                            columns=X.columns,
                            index=X.index)
