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


class DureeMoisDropper(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X.dropna(subset=['dureeMois'])


def create_preprocessing_pipeline_init(cat, min=20000, max=50000000, top_n=40, level=2):
    """
    Crée un pipeline sklearn pour le prétraitement des données de marchés publics.

    Retourne:
    ---------
    sklearn.pipeline.Pipeline
        Pipeline de prétraitement complet
    """
    from scripts.preprocess_init import InitTransformer

    preprocessing_pipeline = Pipeline([
        ('id_accord_encoder', IdAccordCadreEncoder()),
        ('taux_avance_categorizer', TauxAvanceCategorizer()),
        ('outliers_feature_rows_selector', InitTransformer(
            cat=cat,
            min=min,
            max=max,
            top_n=40,
            level=2))
     ])
    return preprocessing_pipeline



def create_preprocessing_pipeline_follow():
    """
    Crée un pipeline sklearn pour le prétraitement des données de marchés publics.
    """
    numerical_columns = ['montant', 'dureeMois', 'offresRecues']
    categorical_columns = ['nature', 'procedure', 'formePrix', 'marcheInnovant', 'ccag',
                          'sousTraitanceDeclaree', 'typeGroupementOperateurs', 'origineFrance',
                          'idAccordCadre', 'codeCPV_2', 'codeCPV_3', 'tauxAvance_cat']

    preprocessing_pipeline = Pipeline([
       ('duree_mois_dropper', DureeMoisDropper()),
       ('log_transformer', LogTransformer(numerical_columns)),
       ('column_transformer', ColumnTransformer([
           ('offres_recues_pipeline', Pipeline([
               ('imputer', SimpleImputer(strategy='median')),
               ('scaler', StandardScaler())
           ]), ['offresRecues']),

           ('other_num_pipeline', Pipeline([
               ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
               ('scaler', StandardScaler())
           ]), ['montant', 'dureeMois']),

           ('cat_pipeline', Pipeline([
               ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
               ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
           ]), categorical_columns)
       ]))
    ])

    return preprocessing_pipeline
