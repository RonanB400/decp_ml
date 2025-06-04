import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class IdAccordCadreEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['idAccordCadre'] = X_transformed['idAccordCadre'].notnull().astype(int)
        return X_transformed


class TauxAvanceCategorizer(BaseEstimator, TransformerMixin):
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

class MissingValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scripts.preprocess_missing_values import clean_missing_values
        return clean_missing_values(X)


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            X_transformed[col] = np.log1p(X_transformed[col])
        return X_transformed

class InitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat=['anomalie', 'pred_motant', 'marche_sim'],
                 min=20000,
                 max=50000000,
                 top_n=40,
                 level=2):
        self.cat = cat
        self.min = min
        self.max = max
        self.top_n = top_n
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scripts.preprocess_missing_values import columns_selection
        return columns_selection(X, self.cat, self.min, self.max, self.top_n, self.level)()



def create_preprocessing_pipeline_init(cat, min=20000, max=50000000, top_n=40, level=2):
    """
    Crée un pipeline sklearn pour le prétraitement des données de marchés publics.

    Retourne:
    ---------
    sklearn.pipeline.Pipeline
        Pipeline de prétraitement complet
    """

    preprocessing_pipeline = Pipeline([
        ('id_accord_encoder', IdAccordCadreEncoder()),
        ('taux_avance_categorizer', TauxAvanceCategorizer()),
        ('missing_values_handler', MissingValues()),
        ('outliers_feature_rows_selector', InitTransformer(
            cat=,
            min=min,
            max=max,
            top_n=40,
            level=2))
     ])



def create_preprocessing_pipeline_follow():
    """
    Crée un pipeline sklearn pour le prétraitement des données de marchés publics.
    """
    numerical_columns = ['montant', 'dureeMois', 'offresRecues']
    categorical_columns = ['nature', 'procedure', 'formePrix', 'marcheInnovant', 'ccag',
                          'sousTraitanceDeclaree', 'typeGroupementOperateurs', 'origineFrance',
                          'idAccordCadre', 'codeCPV_2', 'tauxAvance_cat']

    preprocessing_pipeline = Pipeline([
        ('log_transformer', LogTransformer(numerical_columns)),
        ('column_transformer', ColumnTransformer([
            ('num_pipeline', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_columns),
            ('cat_pipeline', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_columns)
        ]))
    ])

    return preprocessing_pipeline

# Fonction originale maintenue pour compatibilité
def preprocessing_pipeline(df, top_n=40):
    """
    Pipeline complet de prétraitement des données de marchés publics.

    Paramètres:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données brutes
    top_n : int, optional (default=40)
        Nombre de groupes CPV à conserver

    Retourne:
    ---------
    pandas.DataFrame
        DataFrame prétraité prêt pour le clustering
    """
    pipeline = create_preprocessing_pipeline()
    return pipeline.fit_transform(df)
