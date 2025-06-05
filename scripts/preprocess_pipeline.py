import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                  FunctionTransformer)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class IdAccordCadreEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer that converts idAccordCadre to a binary indicator.

    Transforms the idAccordCadre column by checking if it contains a value (1)
    or is null (0), indicating whether the contract is part of a framework
    agreement.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['idAccordCadre'] = (
            X_transformed['idAccordCadre'].notnull().astype(int))
        return X_transformed


class TauxAvanceCategorizer(BaseEstimator, TransformerMixin):
    """
    Transformer that categorizes the tauxAvance (advance payment rate) into
    bins.

    Converts continuous advance payment rates into discrete categories based on
    the provided bins and labels.

    Parameters
    ----------
    bins : list, default=[-0.001, 0.001, 0.05, 0.15, 1.0]
        Bin edges for categorizing advance payment rates.
    labels : list, default=['no_advance', 'small_advance', 'medium_advance',
        'large_advance']
        Labels for the resulting categories.
    """
    def __init__(self, bins=[-0.001, 0.001, 0.05, 0.15, 1.0],
                 labels=['no_advance', 'small_advance', 'medium_advance',
                         'large_advance']):
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['tauxAvance_cat'] = pd.cut(
            X_transformed['tauxAvance'],
            bins=self.bins,
            labels=self.labels)
        return X_transformed.drop(columns=['tauxAvance'])


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies log1p transformation to numerical features.

    Applies np.log1p (natural logarithm of 1 + x) to specified numerical
    columns or to all numerical columns if none are specified. Handles both
    DataFrame and numpy array inputs.

    Parameters
    ----------
    columns : list or None, default=None
        List of column names to transform. If None, transforms all numerical
        columns.
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
        return (input_features if input_features is not None
                else np.array([]))

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
            X = pd.DataFrame(X, columns=[f'col_{i}'
                                       for i in range(X.shape[1])])

        X_transformed = X.copy()
        # Apply log transform to all numeric columns
        if self.columns:
            for col in self.columns:
                if (col in X_transformed.columns and
                        np.issubdtype(X_transformed[col].dtype, np.number)):
                    X_transformed[col] = np.log1p(X_transformed[col])
        else:
            # If no columns specified, transform all numeric columns
            for col in X_transformed.select_dtypes(
                    include=[np.number]).columns:
                X_transformed[col] = np.log1p(X_transformed[col])

        # Return array if input was array
        if is_array:
            return X_transformed.values
        return X_transformed


class InitTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for initial data preprocessing and feature selection.

    Applies the columns_selection function to filter and transform the data
    based on the specified category.

    Parameters
    ----------
    cat : str or None, default=None
        Category to select. If None, defaults to 'pred_montant'.
        Valid options: 'anomalie', 'pred_montant', 'marche_sim'
    """
    def __init__(self, cat=None):
        self.cat = (cat if cat is not None
                    else 'pred_montant')

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
        return columns_selection(X, self.cat)


class StringConverter(BaseEstimator, TransformerMixin):
    """
    Transformer that converts all values to strings.

    Ensures categorical data is uniformly represented as strings before
    encoding.
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
        return (input_features if input_features is not None
                else np.array([]))


def columns_selection(df, cat):
    """
    Select and filter columns based on category.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    cat : str
        Category for selection ('pred_montant', 'marche_sim', 'anomalie')

    Returns
    -------
    pandas.DataFrame
        Filtered and transformed DataFrame
    """

    if cat == 'pred_montant':
        # selection des colonnes
        df = df[['procedure', 'dureeMois', 'nature', 'formePrix',
                 'offresRecues', 'ccag', 'sousTraitanceDeclaree',
                 'typeGroupementOperateurs', 'tauxAvance_cat',
                 'origineFrance', 'idAccordCadre', 'dateNotification',
                 'marcheInnovant', 'codeCPV_2']]


        return df

    elif cat == 'marche_sim':
        # selection des colonnes
        df = df[['procedure', 'dureeMois', 'nature', 'formePrix',
                 'offresRecues', 'ccag', 'sousTraitanceDeclaree',
                 'typeGroupementOperateurs', 'tauxAvance_cat',
                 'origineFrance', 'idAccordCadre', 'montant',
                 'marcheInnovant', 'codeCPV_2']]

        return df

    elif cat == 'anomalie':
        # selection des colonnes for anomaly detection
        # TODO: Customize column selection for anomaly detection
        # For now, using similar structure to marche_sim
        df = df[['procedure', 'dureeMois', 'nature', 'formePrix',
                 'offresRecues', 'ccag', 'sousTraitanceDeclaree',
                 'typeGroupementOperateurs', 'tauxAvance_cat',
                 'origineFrance', 'idAccordCadre', 'montant',
                 'marcheInnovant', 'codeCPV_2']]


        return df

    else:
        return "Error, cat not in 'montant', 'marche_sim', 'anomalie'."


def create_preprocessing_pipeline(cat):
    """
    Create a complete scikit-learn pipeline for preprocessing public 
    contract data.

    This pipeline combines initial preprocessing (feature selection, 
    categorical encoding) with follow-up preprocessing (imputation, 
    scaling, one-hot encoding) in a single pipeline. The column 
    selection and processing adapts based on the specified category.

    Parameters
    ----------
    cat : str
        Category to select. Valid options: 'pred_montant', 'marche_sim', 
        'anomalie'

    Returns
    -------
    sklearn.pipeline.Pipeline
        Complete preprocessing pipeline that outputs a pandas DataFrame
    """
    
    # Determine column lists based on category
    def get_column_lists(category):
        """
        Get column lists based on the category.
        
        Returns
        -------
        tuple
            (numerical_columns, binary_columns, categorical_columns)
        """
        if category == 'pred_montant':
            # Columns after pred_montant processing (no montant, has dateNotification)
            numerical_cols = ['dureeMois', 'offresRecues', 'origineFrance']
            binary_cols = ['marcheInnovant', 'sousTraitanceDeclaree',
                          'idAccordCadre']
            categorical_cols = ['nature', 'procedure', 'formePrix', 'ccag',
                               'typeGroupementOperateurs', 'codeCPV_2',
                               'tauxAvance_cat']
        elif category == 'marche_sim':
            # Columns after marche_sim processing (has montant, no dateNotification)
            numerical_cols = ['montant', 'dureeMois', 'offresRecues',
                             'origineFrance']
            binary_cols = ['marcheInnovant', 'sousTraitanceDeclaree',
                          'idAccordCadre']
            categorical_cols = ['nature', 'procedure', 'formePrix', 'ccag',
                               'typeGroupementOperateurs', 'codeCPV_2',
                               'tauxAvance_cat']
        elif category == 'anomalie':
            # TODO: Define columns for anomalie category
            # For now, using similar structure to marche_sim
            numerical_cols = ['montant', 'dureeMois', 'offresRecues',
                             'origineFrance']
            binary_cols = ['marcheInnovant', 'sousTraitanceDeclaree',
                          'idAccordCadre']
            categorical_cols = ['nature', 'procedure', 'formePrix', 'ccag',
                               'typeGroupementOperateurs', 'codeCPV_2',
                               'tauxAvance_cat']
        else:
            raise ValueError(f"Invalid category: {category}. "
                           "Must be 'pred_montant', 'marche_sim', or 'anomalie'")
        
        return numerical_cols, binary_cols, categorical_cols

    # Get column lists for this category
    numerical_columns, binary_columns, categorical_columns = get_column_lists(cat)

    # Create the initial preprocessing pipeline
    init_pipeline = Pipeline([
        ('id_accord_encoder', IdAccordCadreEncoder()),
        ('taux_avance_categorizer', TauxAvanceCategorizer()),
        ('outliers_feature_rows_selector',
         InitTransformer(cat=cat))
    ])

    # Create the column transformer for follow-up processing
    column_transformer = ColumnTransformer([
        # Process offresRecues
        ('offres_recues_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('log_transform', LogTransformer()),
            ('scaler', StandardScaler())
        ]), (['offresRecues'] if 'offresRecues' in numerical_columns
             else [])),

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
            ('imputer',
             SimpleImputer(strategy='constant', fill_value='missing')),
            ('string_converter', StringConverter()),
            ('onehot', OneHotEncoder(handle_unknown='ignore',
                                    sparse_output=False))
        ]), categorical_columns)
    ], remainder='drop', verbose_feature_names_out=True)

    # Follow-up processing pipeline
    follow_pipeline = Pipeline([
        ('column_transformer', column_transformer)
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

    # Complete pipeline combining init, follow, and DataFrame conversion
    complete_pipeline = Pipeline([
        ('init', init_pipeline),
        ('follow', follow_pipeline),
        ('to_dataframe', FunctionTransformer(
            lambda X: transform_to_df(
                X,
                follow_pipeline.named_steps['column_transformer'].get_feature_names_out()
            ),
            validate=False
        ))
    ])

    return complete_pipeline


# Keep the old functions for backward compatibility
def create_preprocessing_pipeline_init(cat):
    """
    Create a scikit-learn pipeline for initial preprocessing of public
    contract data.

    This function is deprecated. Use create_preprocessing_pipeline() instead.

    Parameters
    ----------
    cat : str
        Category to select for filtering

    Returns
    -------
    sklearn.pipeline.Pipeline
        Initial preprocessing pipeline
    """
    preprocessing_pipeline = Pipeline([
        ('id_accord_encoder', IdAccordCadreEncoder()),
        ('taux_avance_categorizer', TauxAvanceCategorizer()),
        ('outliers_feature_rows_selector',
         InitTransformer(cat=cat))
    ])

    return preprocessing_pipeline


def create_preprocessing_pipeline_follow():
    """
    Create a scikit-learn pipeline for follow-up preprocessing of public
    contract data.

    This function is deprecated. Use create_preprocessing_pipeline() instead.

    The pipeline processes these column types:
    - Numerical: montant, dureeMois, offresRecues
    - Binary: marcheInnovant, sousTraitanceDeclaree (kept as is)
    - Categorical: nature, procedure, formePrix, etc. (one-hot encoded)

    Returns
    -------
    sklearn.pipeline.Pipeline
        Follow-up preprocessing pipeline that outputs a pandas DataFrame
    """

    # adapt these lists to your dataset !!!!
    numerical_columns = ['montant', 'dureeMois', 'offresRecues',
                         'origineFrance']
    binary_columns = ['marcheInnovant', 'sousTraitanceDeclaree',
                      'idAccordCadre']
    categorical_columns = ['nature', 'procedure', 'formePrix', 'ccag',
                           'typeGroupementOperateurs',
                           'codeCPV_2', 'tauxAvance_cat']

    preprocessing_pipeline = Pipeline([
        ('column_transformer', ColumnTransformer([
            # Process offresRecues
            ('offres_recues_pipeline', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('log_transform', LogTransformer()),
                ('scaler', StandardScaler())
            ]), (['offresRecues'] if 'offresRecues' in numerical_columns
                 else [])),

            # Process other numerical columns
            ('other_num_pipeline', Pipeline([
                ('imputer',
                 SimpleImputer(strategy='constant', fill_value=0.0)),
                ('log_transform', LogTransformer()),
                ('scaler', StandardScaler())
            ]), [col for col in numerical_columns if col != 'offresRecues']),

            # Process binary columns - keep as is, just impute missing values
            ('binary_pipeline', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            ]), binary_columns),

            # Process categorical columns
            ('cat_pipeline', Pipeline([
                ('imputer',
                 SimpleImputer(strategy='constant', fill_value='missing')),
                ('string_converter', StringConverter()),
                ('onehot', OneHotEncoder(handle_unknown='ignore',
                                        sparse_output=False))
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
                preprocessing_pipeline.named_steps[
                    'column_transformer'].get_feature_names_out()
            ),
            validate=False
        ))
    ])

    return final_pipeline


def create_complete_pipeline(cat):
    """
    Create a complete preprocessing pipeline combining both init and follow
    steps.
    
    This function is deprecated. Use create_preprocessing_pipeline() instead.
    """
    return create_preprocessing_pipeline(cat)


def create_pred_montant_pipeline():
    """
    Create a preprocessing pipeline specifically for 'pred_montant' category.
    
    This pipeline is optimized for predicting contract amounts and excludes
    the montant column from processing while including dateNotification-related
    features.
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        Complete preprocessing pipeline for pred_montant
    """
    # Column configuration for pred_montant
    numerical_columns = ['dureeMois', 'offresRecues', 'origineFrance']
    binary_columns = ['marcheInnovant', 'sousTraitanceDeclaree', 'idAccordCadre']
    categorical_columns = ['nature', 'procedure', 'formePrix', 'ccag',
                          'typeGroupementOperateurs', 'codeCPV_2',
                          'tauxAvance_cat']
    
    # Initial preprocessing
    init_pipeline = Pipeline([
        ('id_accord_encoder', IdAccordCadreEncoder()),
        ('taux_avance_categorizer', TauxAvanceCategorizer()),
        ('outliers_feature_rows_selector',
         InitTransformer(cat='pred_montant'))
    ])
    
    # Column transformer
    column_transformer = ColumnTransformer([
        ('offres_recues_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('log_transform', LogTransformer()),
            ('scaler', StandardScaler())
        ]), ['offresRecues']),
        
        ('other_num_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
            ('log_transform', LogTransformer()),
            ('scaler', StandardScaler())
        ]), [col for col in numerical_columns if col != 'offresRecues']),
        
        ('binary_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0))
        ]), binary_columns),
        
        ('cat_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('string_converter', StringConverter()),
            ('onehot', OneHotEncoder(handle_unknown='ignore',
                                    sparse_output=False))
        ]), categorical_columns)
    ], remainder='drop', verbose_feature_names_out=True)
    
    # Follow-up processing
    follow_pipeline = Pipeline([
        ('column_transformer', column_transformer)
    ])
    
    def transform_to_df(X, feature_names):
        return pd.DataFrame(X, columns=feature_names)
    
    # Complete pipeline
    complete_pipeline = Pipeline([
        ('init', init_pipeline),
        ('follow', follow_pipeline),
        ('to_dataframe', FunctionTransformer(
            lambda X: transform_to_df(
                X,
                follow_pipeline.named_steps[
                    'column_transformer'].get_feature_names_out()
            ),
            validate=False
        ))
    ])
    
    return complete_pipeline


def create_marche_sim_pipeline():
    """
    Create a preprocessing pipeline specifically for 'marche_sim' category.
    
    This pipeline is optimized for similar contract analysis and includes
    the montant column for processing.
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        Complete preprocessing pipeline for marche_sim
    """
    # Column configuration for marche_sim
    numerical_columns = ['montant', 'dureeMois', 'offresRecues', 'origineFrance']
    binary_columns = ['marcheInnovant', 'sousTraitanceDeclaree', 'idAccordCadre']
    categorical_columns = ['nature', 'procedure', 'formePrix', 'ccag',
                          'typeGroupementOperateurs', 'codeCPV_2',
                          'tauxAvance_cat']
    
    # Initial preprocessing
    init_pipeline = Pipeline([
        ('id_accord_encoder', IdAccordCadreEncoder()),
        ('taux_avance_categorizer', TauxAvanceCategorizer()),
        ('outliers_feature_rows_selector',
         InitTransformer(cat='marche_sim'))
    ])
    
    # Column transformer
    column_transformer = ColumnTransformer([
        ('offres_recues_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('log_transform', LogTransformer()),
            ('scaler', StandardScaler())
        ]), ['offresRecues']),
        
        ('other_num_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
            ('log_transform', LogTransformer()),
            ('scaler', StandardScaler())
        ]), [col for col in numerical_columns if col != 'offresRecues']),
        
        ('binary_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0))
        ]), binary_columns),
        
        ('cat_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('string_converter', StringConverter()),
            ('onehot', OneHotEncoder(handle_unknown='ignore',
                                    sparse_output=False))
        ]), categorical_columns)
    ], remainder='drop', verbose_feature_names_out=True)
    
    # Follow-up processing
    follow_pipeline = Pipeline([
        ('column_transformer', column_transformer)
    ])
    
    def transform_to_df(X, feature_names):
        return pd.DataFrame(X, columns=feature_names)
    
    # Complete pipeline
    complete_pipeline = Pipeline([
        ('init', init_pipeline),
        ('follow', follow_pipeline),
        ('to_dataframe', FunctionTransformer(
            lambda X: transform_to_df(
                X,
                follow_pipeline.named_steps[
                    'column_transformer'].get_feature_names_out()
            ),
            validate=False
        ))
    ])
    
    return complete_pipeline


def create_anomalie_pipeline():
    """
    Create a preprocessing pipeline specifically for 'anomalie' category.
    
    This pipeline is optimized for anomaly detection and includes features
    relevant for identifying irregular contract patterns.
    
    Note: Currently uses similar configuration to marche_sim. 
    TODO: Customize columns for anomaly detection specific features.
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        Complete preprocessing pipeline for anomalie
    """
    # Column configuration for anomalie
    # TODO: Customize these columns for anomaly detection
    numerical_columns = ['montant', 'dureeMois', 'offresRecues', 'origineFrance']
    binary_columns = ['marcheInnovant', 'sousTraitanceDeclaree', 'idAccordCadre']
    categorical_columns = ['nature', 'procedure', 'formePrix', 'ccag',
                          'typeGroupementOperateurs', 'codeCPV_2',
                          'tauxAvance_cat']
    
    # Initial preprocessing
    init_pipeline = Pipeline([
        ('id_accord_encoder', IdAccordCadreEncoder()),
        ('taux_avance_categorizer', TauxAvanceCategorizer()),
        ('outliers_feature_rows_selector',
         InitTransformer(cat='anomalie'))
    ])
    
    # Column transformer
    column_transformer = ColumnTransformer([
        ('offres_recues_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('log_transform', LogTransformer()),
            ('scaler', StandardScaler())
        ]), ['offresRecues']),
        
        ('other_num_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
            ('log_transform', LogTransformer()),
            ('scaler', StandardScaler())
        ]), [col for col in numerical_columns if col != 'offresRecues']),
        
        ('binary_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0))
        ]), binary_columns),
        
        ('cat_pipeline', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('string_converter', StringConverter()),
            ('onehot', OneHotEncoder(handle_unknown='ignore',
                                    sparse_output=False))
        ]), categorical_columns)
    ], remainder='drop', verbose_feature_names_out=True)
    
    # Follow-up processing
    follow_pipeline = Pipeline([
        ('column_transformer', column_transformer)
    ])
    
    def transform_to_df(X, feature_names):
        return pd.DataFrame(X, columns=feature_names)
    
    # Complete pipeline
    complete_pipeline = Pipeline([
        ('init', init_pipeline),
        ('follow', follow_pipeline),
        ('to_dataframe', FunctionTransformer(
            lambda X: transform_to_df(
                X,
                follow_pipeline.named_steps[
                    'column_transformer'].get_feature_names_out()
            ),
            validate=False
        ))
    ])
    
    return complete_pipeline 