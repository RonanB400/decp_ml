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
    def __init__(self, y=None):


    def fit(self, X, y=None):
        # Logique pour adapter le transformateur aux données
        # Généralement, stocke des statistiques ou paramètres
        return self

    def transform(self, X):
        # Logique pour transformer les données
        X_transformed = X.copy()
        # Effectuer des modifications...
        return X_transformed

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



def create_preprocessing_pipeline():
    """
    Crée un pipeline sklearn pour le prétraitement des données de marchés publics.

    Retourne:
    ---------
    sklearn.pipeline.Pipeline
        Pipeline de prétraitement complet
    """
    numerical_columns = ['montant', 'dureeMois', 'offresRecues']
    categorical_columns = ['nature', 'procedure', 'formePrix', 'marcheInnovant', 'ccag',
                          'sousTraitanceDeclaree', 'typeGroupementOperateurs', 'origineFrance',
                          'idAccordCadre', 'codeCPV_2', 'tauxAvance_cat']

    # Définir le pipeline
    preprocessing_pipeline = Pipeline([
        ('id_accord_encoder', IdAccordCadreEncoder()),
        ('taux_avance_categorizer', TauxAvanceCategorizer()),
        ('missing_values_handler', MissingValues()),
        ('outliers_feature_rows_selector', MissingValues()),
        ('log_transformer', LogTransformer(numerical_columns)),
        ('cpv_hierarchy', CPVHierarchyTransformer()),
        ('cpv_filter', CPVFilterer(top_n=40)),
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
