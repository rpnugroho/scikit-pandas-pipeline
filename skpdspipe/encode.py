import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer


class DFOrdinalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self,
                 categories='auto',
                 dtype=np.float64,
                 handle_unknown='use_encoded_value',
                 unknown_value=np.nan,
                 fill_value=np.nan):
        """EXPERIMENTAL

        Args:
            categories (str, optional): [description]. Defaults to 'auto'.
            dtype ([type], optional): [description]. Defaults to np.float64.
            handle_unknown (str, optional): [description]. Defaults to 'use_encoded_value'.
            unknown_value ([type], optional): [description]. Defaults to np.nan.
            fill_value ([type], optional): [description]. Defaults to np.nan.
        """
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.fill_value = fill_value
        self.imputer = SimpleImputer(strategy='most_frequent')
        self.encoder = OrdinalEncoder(categories=self.categories,
                                      dtype=self.dtype,
                                      handle_unknown=self.handle_unknown,
                                      unknown_value=self.unknown_value)

    def fit(self, X, y=None):
        X = X.copy()
        # Create mask to preserve missing value
        # self.na_mask = X.isna()
        # Impute missing value with most_frequent variable
        # becase OrdinalEncoder cannot takes missing value
        # self.imputer.fit(X)
        self.encoder.fit(self.imputer.fit_transform(X))
        return self

    def transform(self, X):
        X = X.copy()
        na_mask = X.isna()
        # Impute missing value, and encode
        data = self.encoder.transform(self.imputer.transform(X))
        X_ = pd.DataFrame(data=data,
                          columns=X.columns,
                          index=X.index)
        # Get back missing using mask
        return X_.mask(na_mask, self.fill_value)


class DFDummyEncoder(BaseEstimator, TransformerMixin):
    """One hot encoding DataFrame using pandas dummies.
    """

    def fit(self, X, y=None):
        # persist the dummy variables found in train set
        self.dummies = pd.get_dummies(X).columns
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X_ = pd.concat([X, pd.get_dummies(X)],
                       axis=1)

        X_.drop(labels=X.columns, axis=1, inplace=True)
        # add missing dummies if any
        missing_vars = [var for var in self.dummies if var not in X_.columns]
        if len(missing_vars) != 0:
            for var in missing_vars:
                X_[var] = 0
        return X_


class DFKBinsDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_bins=5,
                 encode='onehot-dense',
                 strategy='quantile',
                 prefix='bin_',
                 fill_value=np.nan,
                 dtype=np.int32):

        self.n_bins = n_bins
        self.prefix = prefix
        self.encode = encode
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy='most_frequent')
        self.discretizer = KBinsDiscretizer(n_bins=self.n_bins,
                                            encode=self.encode,
                                            strategy=self.strategy)
        self.fill_value = fill_value
        self.dtype = dtype

    def fit(self, X, y=None):
        X = X.copy()
        # Create mask to preserve missing value
        self.na_mask = X.isna()
        # Impute missing value with most_frequent variable
        # becase OrdinalEncoder cannot takes missing value
        self.imputer.fit(X)
        self.discretizer.fit(self.imputer.transform(X))
        return self

    def transform(self, X):
        X = X.copy()
        if self.encode == 'ordinal':
            data = self.discretizer.transform(self.imputer.transform(X))
            X_ = pd.DataFrame(data=data.astype(self.dtype),
                              index=X.index,
                              columns=X.columns)
            # Get back missing using mask
            return X_.mask(self.na_mask, self.fill_value)
        # self.encode == 'onehot'
        # At the time i write this code,
        # Its to hard for me to apply back missing value to each row
        # But this is my idea
        # 1. Get row index of missing value for each column
        # 2. Each column has encoded to n_bins, apply back missing value
        #   to this bins.
        return pd.DataFrame(self.discretizer.transform(X).astype(self.dtype),
                            index=X.index).add_prefix(self.prefix)
