import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one


class DFFeatureUnion(FeatureUnion):
    """[EXPERIMENTAL] Wrapper class for FeatureUnion to return Pandas DataFrame 
    instead of scipy sparse matrix.

    Parameters:
        transformer_list : list of (string, transformer) tuples
            List of transformer objects to be applied to the data.

        n_jobs : int, default=None
            Number of jobs to run in parallel.

        transformer_weights : dict, default=None
            Multiplicative weights for features per transformer.

        verbose : bool, default=False
            If True, the time elapsed while fitting each transformer will be
            printed as it is completed.

    Example:
        union = DFFeatureUnion([
            ('num', num_pipe),
            ('cat', cat_pipe),
            ('buc', buc_pipe)
        ])
    """

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())
        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        Xs = self.merge_dataframes_by_column(Xs)
        # get all fitted columns
        self.fitted_columns = Xs.columns

        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        X = X.copy()
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        X_ = self.merge_dataframes_by_column(Xs)
        # Create data for missing columns
        missing_vars = [
            var for var in self.fitted_columns if var not in X_.columns]
        if len(missing_vars) != 0:
            for var in missing_vars:
                X_[var] = 0
        # Matching transformed column and fitted columns
        X_ = X_[self.fitted_columns].copy()
        return X_


class DFColumnsSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns, dtype=None):
        """Select columns from DataFrame.

        Args:
            columns (str or list): A list or string of column names.
            dtype (type, optional): Select by data type. Defaults to None.
        """
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.dtype:
            return X[self.columns].astype(self.dtype)
        else:
            return X[self.columns]


class DataFrameUnion(TransformerMixin, BaseEstimator):
    """
    In: list of (string, transformer) tuples :
    Out: pd.DataFrame
    """

    def __init__(self, transformer_list):
        self.feature_names = None
        # (string, Transformer)-tuple list
        self.transformer_list = transformer_list

    def __getitem__(self, attrib):
        return self.__dict__[attrib]

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data to be transformed.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = (self._transform_one(trans, X)
              for name, trans in self.transformer_list)
        df_merged_result = self._merge_results(Xs)
        return df_merged_result

    def fit(self, X, y=None):
        """Fit all transformers using X.
        Parameters
        ----------
        :param X: pd.DataFrame
            Input data, used to fit transformers.
        :param y:
        """
        transformers = (self._fit_one_transformer(trans, X, y)
                        for name, trans in self.transformer_list)
        self._update_transformer_list(transformers)
        return self

    def _merge_results(self, transformed_result_generator):
        df_merged_result = ''
        for transformed in transformed_result_generator:
            if isinstance(transformed, pd.Series):
                transformed = pd.DataFrame(data=transformed)
            if not isinstance(df_merged_result, pd.DataFrame):
                df_merged_result = transformed
            else:
                df_merged_result = pd.concat(
                    [df_merged_result, transformed], axis=1)

        if self.feature_names is None:
            self.feature_names = df_merged_result.columns
        elif (len(self.feature_names) != len(df_merged_result.columns)) or \
                ((self.feature_names != df_merged_result.columns).any()):
            custom_dataframe = pd.DataFrame(
                data=0, columns=self.feature_names, index=df_merged_result.index)
            custom_dataframe.update(df_merged_result)
            df_merged_result = custom_dataframe
        return df_merged_result

    def _update_transformer_list(self, transformers):
        self.transformer_list[:] = [
            (name, new)
            for ((name, old), new) in zip(self.transformer_list, transformers)
        ]

    def _fit_one_transformer(self, transformer, X, y):
        return transformer.fit(X, y)

    def _transform_one(self, transformer, X):
        return transformer.transform(X)
