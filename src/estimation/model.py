"""
Modèle d'estimation des montants de marché par classification

Transforme le problème de prédiction de montant en classification
par fourchettes pour une meilleure robustesse.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import List, Tuple, Optional
import yaml
import os


class EstimationModel:
    """
    Modèle de classification pour l'estimation des montants de marché
    
    Transforme le problème en classification par fourchettes :
    - Marché de faible montant (0-25k€)
    - Marché adapté (25k-90k€) 
    - Procédure formalisée européenne (90k-221k€)
    - Marché important (221k-1M€)
    - Très gros marché (>1M€)
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise le modèle d'estimation
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.model = None
        self.feature_columns = []
        self.amount_ranges = []
        self.range_labels = []
        
        # Charger la configuration
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                model_config = config.get('models', {}).get('estimation', {})
                self.feature_columns = model_config.get('features', [])
                business_config = config.get('business', {})
                self.amount_ranges = business_config.get('amount_ranges', [])
    
    def _create_amount_categories(self, amounts: pd.Series) -> pd.Series:
        """
        Convertit les montants en catégories de fourchettes
        
        Args:
            amounts: Série des montants à catégoriser
            
        Returns:
            Série des catégories correspondantes
        """
        categories = []
        labels = [
            "Marché de faible montant",
            "Marché adapté", 
            "Procédure formalisée européenne",
            "Marché important",
            "Très gros marché"
        ]
        
        for amount in amounts:
            for i, (min_val, max_val) in enumerate(self.amount_ranges):
                if min_val <= amount < max_val:
                    categories.append(i)
                    break
            else:
                categories.append(len(self.amount_ranges) - 1)  # Dernière catégorie
        
        self.range_labels = labels[:len(self.amount_ranges)]
        return pd.Series(categories)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Entraîne le modèle de classification
        
        Args:
            X: Features d'entraînement
            y: Montants cibles à transformer en catégories
            
        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        # Transformer les montants en catégories
        y_categorical = self._create_amount_categories(y)
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42
        )
        
        # Entraîner le modèle
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, 
                                     target_names=self.range_labels,
                                     output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédit la fourchette de montant pour de nouveaux marchés
        
        Args:
            X: Features des nouveaux marchés
            
        Returns:
            Tuple (prédictions, probabilités)
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def get_amount_range(self, category: int) -> Tuple[float, float]:
        """
        Retourne la fourchette de montant pour une catégorie donnée
        
        Args:
            category: Index de la catégorie
            
        Returns:
            Tuple (montant_min, montant_max)
        """
        if 0 <= category < len(self.amount_ranges):
            return self.amount_ranges[category]
        else:
            return self.amount_ranges[-1]
    
    def save_model(self, filepath: str):
        """Sauvegarde le modèle entraîné"""
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'amount_ranges': self.amount_ranges,
            'range_labels': self.range_labels
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Charge un modèle pré-entraîné"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.amount_ranges = model_data['amount_ranges']
        self.range_labels = model_data['range_labels'] 