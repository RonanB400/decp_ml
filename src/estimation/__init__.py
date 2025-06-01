"""
Module d'estimation des montants de marché

Ce module implémente la classification par fourchettes de montants
pour prédire le coût d'un nouveau marché public.
"""

from .model import EstimationModel
from .preprocessing import preprocess_market_data
from .features import extract_features

__all__ = ['EstimationModel', 'preprocess_market_data', 'extract_features'] 