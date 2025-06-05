import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scripts.preprocess_init import columns_selection


class IdAccordCadreEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer that converts idAccordCadre to a binary indicator.

    Transforms the idAccordCadre column by checking if it contains a value (1)
    or is null (0), indicating whether the contract is part of a framework agreement.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['idAccordCadre'] = X_transformed['idAccordCadre'].notnull().astype(int)
        return X_transformed


class TauxAvanceCategorizer(BaseEstimator, TransformerMixin):
    """
    Transformer that categorizes the tauxAvance (advance payment rate) into bins.

    Converts continuous advance payment rates into discrete categories based on
    the provided bins and labels.

    Parameters
    ----------
    bins : list, default=[-0.001, 0.001, 0.05, 0.15, 1.0]
        Bin edges for categorizing advance payment rates.
    labels : list, default=['no_advance', 'small_advance', 'medium_advance', 'large_advance']
        Labels for the resulting categories.
    """
    def __init__(self, bins=[-0.001, 0.001, 0.05, 0.15, 1.0],
                 labels=['no_advance', 'small_advance', 'medium_advance', 'large_advance']):
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['tauxAvance_cat'] = pd.cut(X_transformed['tauxAvance'],
                                                bins=self.bins,
                                                labels=self.labels)
        return X_transformed.drop(columns=['tauxAvance'])


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies log1p transformation to numerical features.

    Applies np.log1p (natural logarithm of 1 + x) to specified numerical columns
    or to all numerical columns if none are specified. Handles both DataFrame
    and numpy array inputs.

    Parameters
    ----------
    columns : list or None, default=None
        List of column names to transform. If None, transforms all numerical columns.
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like or None, default=None
            Input feature names

        Returns
        -------
        feature_names_out : ndarray of str objects
            Same as input_features if provided, empty array otherwise.
        """
        return input_features if input_features is not None else np.array([])

    def transform(self, X):
        """
        Apply log1p transformation to numerical columns.

        Parameters
        ----------
        X : pandas DataFrame or numpy array
            Data to transform

        Returns
        -------
        X_transformed : pandas DataFrame or numpy array
            Log-transformed data in the same format as the input
        """
        # Convert to DataFrame if input is an array
        is_array = not hasattr(X, 'columns')
        if is_array:
            X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])

        X_transformed = X.copy()
        # Apply log transform to all numeric columns
        if self.columns:
            for col in self.columns:
                if col in X_transformed.columns and np.issubdtype(X_transformed[col].dtype, np.number):
                    X_transformed[col] = np.log1p(X_transformed[col])
        else:
            # If no columns specified, transform all numeric columns
            for col in X_transformed.select_dtypes(include=[np.number]).columns:
                X_transformed[col] = np.log1p(X_transformed[col])

        # Return array if input was array
        if is_array:
            return X_transformed.values
        return X_transformed


class InitTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for initial data preprocessing and feature selection.

    Applies the columns_selection function from preprocess_init module with
    the specified parameters to filter and transform the data.

    Parameters
    ----------
    cat : list or None, default=None
        Categories to select. If None, defaults to ['anomalie', 'pred_montant', 'marche_sim']
    min : int, default=20000
        Minimum value threshold for filtering
    max : int, default=50000000
        Maximum value threshold for filtering
    top_n : int, default=40
        Number of top categories to select
    level : int, default=2
        Level parameter for the selection process
    """
    def __init__(self, cat=None, min=20000, max=50000000, top_n=40, level=2):
        self.cat = cat if cat is not None else ['anomalie', 'pred_montant', 'marche_sim']
        self.min = min
        self.max = max
        self.top_n = top_n
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Apply initial data transformations and selections.

        Parameters
        ----------
        X : pandas DataFrame
            Input data

        Returns
        -------
        pandas DataFrame
            Transformed and filtered data
        """
        return columns_selection(X, self.cat, self.min, self.max, self.top_n, self.level)


class DureeMoisDropper(BaseEstimator, TransformerMixin):
    """
    Transformer that removes rows with missing values in 'dureeMois' column.

    This ensures that all remaining contracts have valid duration information.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Remove rows with missing 'dureeMois' values.

        Parameters
        ----------
        X : pandas DataFrame
            Input data

        Returns
        -------
        pandas DataFrame
            Data with rows containing NaN in 'dureeMois' removed
        """
        return X.dropna(subset=['dureeMois'])


class StringConverter(BaseEstimator, TransformerMixin):
    """
    Transformer that converts all values to strings.

    Ensures categorical data is uniformly represented as strings before encoding.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Convert all values in X to strings.

        Parameters
        ----------
        X : array-like
            Input data

        Returns
        -------
        array-like
            Data with all values converted to strings
        """
        return X.astype(str)

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like or None, default=None
            Input feature names

        Returns
        -------
        feature_names_out : ndarray of str objects
            Same as input_features if provided, empty array otherwise.
        """
        return input_features if input_features is not None else np.array([])


def create_preprocessing_pipeline_init(cat, min=20000, max=50000000, top_n=40, level=2):
    """
    Create a scikit-learn pipeline for initial preprocessing of public contract data.

    This pipeline handles:
    1. Converting idAccordCadre to a binary indicator
    2. Categorizing tauxAvance into bins
    3. Selecting and filtering features based on the specified parameters

    Parameters
    ----------
    cat : list
        Categories to select for filtering
    min : int, default=20000
        Minimum value threshold for filtering
    max : int, default=50000000
        Maximum value threshold for filtering
    top_n : int, default=40
        Number of top categories to select
    level : int, default=2
        Level parameter for the selection process

    Returns
    -------
    sklearn.pipeline.Pipeline
        Complete initial preprocessing pipeline
    """
    preprocessing_pipeline = Pipeline([
        ('id_accord_encoder', IdAccordCadreEncoder()),
        ('taux_avance_categorizer', TauxAvanceCategorizer()),
        ('outliers_feature_rows_selector', InitTransformer(cat=cat, min=min, max=max, top_n=top_n, level=level)),
     ])

    return preprocessing_pipeline



def create_preprocessing_pipeline_follow():
    """
    Create a scikit-learn pipeline for follow-up preprocessing of public contract data.

    This pipeline handles:
    1. Removing rows with missing dureeMois values
    2. Transforming numerical columns with log and scaling
    3. Processing categorical columns with one-hot encoding
    4. Converting the result to a pandas DataFrame with proper column names

    The pipeline processes these column types:
    - Numerical: montant, dureeMois, offresRecues
    - Binary: marcheInnovant, sousTraitanceDeclaree (kept as is)
    - Categorical: nature, procedure, formePrix, etc. (one-hot encoded)

    Returns
    -------
    sklearn.pipeline.Pipeline
        Complete follow-up preprocessing pipeline that outputs a pandas DataFrame
    """
    numerical_columns = ['montant', 'dureeMois', 'offresRecues', 'origineFrance']
    binary_columns = ['marcheInnovant', 'sousTraitanceDeclaree', 'idAccordCadre']
    categorical_columns = ['nature', 'procedure', 'formePrix', 'ccag',
                          'typeGroupementOperateurs',
                          'codeCPV_2', 'tauxAvance_cat']

    preprocessing_pipeline = Pipeline([
        ('duree_mois_dropper', DureeMoisDropper()),
        ('column_transformer', ColumnTransformer([
            # Process offresRecues
            ('offres_recues_pipeline', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('log_transform', LogTransformer()),
                ('scaler', StandardScaler())
            ]), ['offresRecues'] if 'offresRecues' in numerical_columns else []),

            # Process other numerical columns
            ('other_num_pipeline', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
                ('log_transform', LogTransformer()),
                ('scaler', StandardScaler())
            ]), [col for col in numerical_columns if col != 'offresRecues']),

            # Process binary columns - keep as is, just impute missing values
            ('binary_pipeline', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            ]), binary_columns),

            # Process categorical columns
            ('cat_pipeline', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('string_converter', StringConverter()),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_columns)
        ], remainder='drop', verbose_feature_names_out=True))
    ])


    def transform_to_df(X, feature_names):
        """
        Convert array to DataFrame with specified column names.

        Parameters
        ----------
        X : numpy array
            Data to convert
        feature_names : list
            Column names for the DataFrame

        Returns
        -------
        pandas.DataFrame
            DataFrame with the provided column names
        """
        return pd.DataFrame(X, columns=feature_names)

    final_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('to_dataframe', FunctionTransformer(
            lambda X: transform_to_df(
                X,
                preprocessing_pipeline.named_steps['column_transformer'].get_feature_names_out()
            ),
            validate=False
        ))
    ])

    return final_pipeline



def create_complete_pipeline(cat, min=20000, max=50000000, top_n=40, level=2):
    """Create a complete preprocessing pipeline combining both init and follow steps."""
    init_pipeline = create_preprocessing_pipeline_init(cat, min, max, top_n, level)
    follow_pipeline = create_preprocessing_pipeline_follow()

    complete_pipeline = Pipeline([
        ('init', init_pipeline),
        ('follow', follow_pipeline)
    ])

    return complete_pipeline
