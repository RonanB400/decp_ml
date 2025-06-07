"""
Synthetic Anomaly Generator for Public Procurement Data

This module creates synthetic anomalies in procurement datasets to test
the effectiveness of anomaly detection models, particularly GNN-based approaches.

The anomalies are based on red flags identified in procurement literature
and real-world corruption patterns.

Author: RonanB400
Date: January 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticAnomalyGenerator:
    """Generate synthetic anomalies in procurement data for testing detection models."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the anomaly generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Track which rows have been modified for each anomaly type
        self.anomaly_labels = {}
        
    def generate_anomalies(self, 
                          df: pd.DataFrame,
                          anomaly_types: List[str] = None,
                          anomaly_percentage: float = 0.05,
                          preserve_originals: bool = True) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Generate synthetic anomalies by adding new rows to the dataset.
        
        Args:
            df: Original procurement dataframe
            anomaly_types: List of anomaly types to generate. If None, generates all types
            anomaly_percentage: Percentage of synthetic anomalies relative to original data
            preserve_originals: Whether to keep original anomalous patterns intact
            
        Returns:
            Tuple of (dataframe with added anomalous rows, dictionary of anomaly labels)
        """
        
        if anomaly_types is None:
            anomaly_types = [
                'single_bid_competitive',
                'high_market_concentration', 
                'price_manipulation',
                'procedure_manipulation',
                'suspicious_modifications',
                'temporal_clustering',
                'excessive_subcontracting',
                'unusual_contract_duration',
                'suspicious_buyer_supplier_pairs'
            ]
        
        # Start with original dataframe
        df_original = df.copy()
        
        # Calculate number of anomalies to create per type
        total_anomalies = int(len(df_original) * anomaly_percentage)
        anomalies_per_type = max(1, total_anomalies // len(anomaly_types))
        
        logger.info(f"Generating {total_anomalies} total synthetic anomaly rows across {len(anomaly_types)} types")
        logger.info(f"Approximately {anomalies_per_type} anomalies per type")
        
        # Store all new anomalous rows
        all_anomalous_rows = []
        
        # Generate each type of anomaly
        for anomaly_type in anomaly_types:
            logger.info(f"Generating {anomaly_type} anomalies...")
            
            if anomaly_type == 'single_bid_competitive':
                new_rows = self._generate_single_bid_anomalies(
                    df_original, anomalies_per_type, anomaly_type)
                    
            elif anomaly_type == 'high_market_concentration':
                new_rows = self._generate_market_concentration_anomalies(
                    df_original, anomalies_per_type, anomaly_type)
                    
            elif anomaly_type == 'price_manipulation':
                new_rows = self._generate_price_manipulation_anomalies(
                    df_original, anomalies_per_type, anomaly_type)
                    
            elif anomaly_type == 'procedure_manipulation':
                new_rows = self._generate_procedure_manipulation_anomalies(
                    df_original, anomalies_per_type, anomaly_type)
                    
            elif anomaly_type == 'suspicious_modifications':
                new_rows = self._generate_suspicious_modification_anomalies(
                    df_original, anomalies_per_type, anomaly_type)
                    
            elif anomaly_type == 'temporal_clustering':
                new_rows = self._generate_temporal_clustering_anomalies(
                    df_original, anomalies_per_type, anomaly_type)
                    
            elif anomaly_type == 'excessive_subcontracting':
                new_rows = self._generate_excessive_subcontracting_anomalies(
                    df_original, anomalies_per_type, anomaly_type)
                    
            elif anomaly_type == 'unusual_contract_duration':
                new_rows = self._generate_unusual_duration_anomalies(
                    df_original, anomalies_per_type, anomaly_type)
                    
            elif anomaly_type == 'suspicious_buyer_supplier_pairs':
                new_rows = self._generate_suspicious_pairs_anomalies(
                    df_original, anomalies_per_type, anomaly_type)
            
            if len(new_rows) > 0:
                all_anomalous_rows.extend(new_rows)
        
        # Combine original data with synthetic anomalies
        if all_anomalous_rows:
            anomalous_df = pd.DataFrame(all_anomalous_rows)
            df_combined = pd.concat([df_original, anomalous_df], ignore_index=True)
        else:
            df_combined = df_original.copy()
        
        # Create anomaly labels for the combined dataset
        n_original = len(df_original)
        n_total = len(df_combined)
        
        # Initialize labels - original rows are False, synthetic rows will be True
        self.anomaly_labels = {}
        for anomaly_type in anomaly_types:
            self.anomaly_labels[anomaly_type] = np.zeros(n_total, dtype=bool)
        
        # Mark synthetic anomalies
        current_idx = n_original
        for row_data in all_anomalous_rows:
            if 'anomaly_type' in row_data:
                anomaly_type = row_data['anomaly_type']
                self.anomaly_labels[anomaly_type][current_idx] = True
            current_idx += 1
        
        # Add general anomaly indicator column
        df_combined['is_synthetic_anomaly'] = np.zeros(len(df_combined), dtype=bool)
        df_combined.loc[n_original:, 'is_synthetic_anomaly'] = True
        
        # Log summary
        total_synthetic_anomalies = np.sum(df_combined['is_synthetic_anomaly'])
        logger.info(f"Generated {total_synthetic_anomalies} total synthetic anomaly rows")
        logger.info(f"Original dataset: {n_original} rows")
        logger.info(f"Combined dataset: {len(df_combined)} rows ({total_synthetic_anomalies/len(df_combined)*100:.2f}% synthetic)")
        
        for anomaly_type, labels in self.anomaly_labels.items():
            count = np.sum(labels)
            logger.info(f"  - {anomaly_type}: {count} anomalies")
        
        return df_combined, self.anomaly_labels
    
    def _generate_single_bid_anomalies(self, df: pd.DataFrame, n_anomalies: int, anomaly_type: str) -> List[Dict]:
        """Generate new rows with single bid competitive anomalies."""
        
        # Find competitive procedures with more than 1 bid as templates
        competitive_procedures = ["Appel d'offres ouvert", "Appel d'offres restreint", 
                                "Procédure concurrentielle avec négociation"]
        
        mask = (df['procedure'].isin(competitive_procedures) & 
                (df['offresRecues'] > 1) & 
                df['offresRecues'].notna())
        
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for single_bid_competitive anomalies")
            return []
        
        # Randomly select template rows
        selected_rows = eligible_rows.sample(n=min(n_anomalies, len(eligible_rows)), random_state=self.random_seed)
        
        new_rows = []
        for _, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            
            # Make it anomalous: set to exactly 1 bid
            new_row['offresRecues'] = 1
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
        
        logger.info(f"Generated {len(new_rows)} single bid competitive anomaly rows")
        return new_rows
    
    def _generate_market_concentration_anomalies(self, df: pd.DataFrame, n_anomalies: int, anomaly_type: str) -> List[Dict]:
        """Generate new rows where one supplier dominates a buyer's contracts."""
        
        # Find buyer-CPV combinations with multiple suppliers
        buyer_cpv_groups = df.groupby(['acheteur_id', 'codeCPV_3'])
        
        eligible_groups = []
        for (buyer_id, cpv), group in buyer_cpv_groups:
            if len(group) >= 3 and group['titulaire_id'].nunique() > 1:
                eligible_groups.append((buyer_id, cpv, group))
        
        if len(eligible_groups) == 0:
            logger.warning("No eligible buyer-CPV groups found for market concentration anomalies")
            return []
        
        # Select random groups to create anomalies from
        selected_groups = random.sample(eligible_groups, min(n_anomalies // 3, len(eligible_groups)))
        
        new_rows = []
        for buyer_id, cpv, group in selected_groups:
            # Pick one supplier to dominate
            suppliers = group['titulaire_id'].unique()
            dominant_supplier = random.choice(suppliers)
            
            # Create new contracts for this dominant supplier
            template_rows = group.sample(n=min(3, len(group)), random_state=self.random_seed)
            
            for _, row in template_rows.iterrows():
                new_row = row.copy()
                new_row['titulaire_id'] = dominant_supplier
                
                # Add anomaly metadata
                new_row['anomaly_type'] = anomaly_type
                new_row['source_type'] = 'synthetic'
                
                new_rows.append(new_row.to_dict())
        
        logger.info(f"Generated {len(new_rows)} market concentration anomaly rows")
        return new_rows
    
    def _generate_price_manipulation_anomalies(self, df: pd.DataFrame, n_anomalies: int, anomaly_type: str) -> List[Dict]:
        """Generate new rows with artificially inflated or deflated prices."""
        
        # Find contracts with valid amounts as templates
        mask = df['montant'].notna() & (df['montant'] > 0)
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for price manipulation anomalies")
            return []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(n=min(n_anomalies, len(eligible_rows)), random_state=self.random_seed)
        
        new_rows = []
        for _, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            original_amount = new_row['montant']
            
            # Randomly choose inflation or deflation
            if random.random() < 0.5:
                # Inflate by 200-500%
                multiplier = random.uniform(3.0, 6.0)
            else:
                # Deflate to 10-30% of original
                multiplier = random.uniform(0.1, 0.3)
            
            new_row['montant'] = original_amount * multiplier
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
        
        logger.info(f"Generated {len(new_rows)} price manipulation anomaly rows")
        return new_rows
    
    def _generate_procedure_manipulation_anomalies(self, df: pd.DataFrame, n_anomalies: int) -> pd.DataFrame:
        """Generate anomalies where buyers switch to non-competitive procedures suspiciously."""
        
        # Find buyers who normally use competitive procedures
        buyer_procedure_stats = df.groupby('acheteur_id')['procedure'].agg(['count', 'nunique']).reset_index()
        buyer_procedure_stats = buyer_procedure_stats[buyer_procedure_stats['count'] >= 5]  # At least 5 contracts
        
        competitive_procedures = ['Appel d\'offres ouvert', 'Appel d\'offres restreint']
        non_competitive = ['Procédure adaptée', 'Marché négocié sans publicité']
        
        eligible_buyers = []
        for buyer_id in buyer_procedure_stats['acheteur_id']:
            buyer_contracts = df[df['acheteur_id'] == buyer_id]
            competitive_ratio = buyer_contracts['procedure'].isin(competitive_procedures).mean()
            
            if competitive_ratio > 0.7:  # Normally uses competitive procedures
                eligible_buyers.append(buyer_id)
        
        if len(eligible_buyers) == 0:
            logger.warning("No eligible buyers found for procedure manipulation anomalies")
            return df
        
        # Select contracts from these buyers and change to non-competitive
        anomaly_count = 0
        for buyer_id in random.sample(eligible_buyers, min(len(eligible_buyers), n_anomalies // 2)):
            buyer_contracts = df[(df['acheteur_id'] == buyer_id) & 
                               df['procedure'].isin(competitive_procedures)]
            
            if len(buyer_contracts) > 0:
                # Change 1-3 contracts to non-competitive
                n_to_change = min(random.randint(1, 3), len(buyer_contracts))
                selected_indices = random.sample(buyer_contracts.index.tolist(), n_to_change)
                
                for idx in selected_indices:
                    df.loc[idx, 'procedure'] = random.choice(non_competitive)
                
                self.anomaly_labels['procedure_manipulation'][selected_indices] = True
                anomaly_count += len(selected_indices)
        
        logger.info(f"Generated {anomaly_count} procedure manipulation anomalies")
        return df
    
    def _generate_suspicious_modification_anomalies(self, df: pd.DataFrame, n_anomalies: int) -> pd.DataFrame:
        """Generate anomalies suggesting suspicious contract modifications."""
        
        # Find contracts that could be "modified" (increase duration dramatically)
        mask = df['dureeMois'].notna() & (df['dureeMois'] > 0) & (df['dureeMois'] < 36)
        eligible_indices = df[mask].index.tolist()
        
        if len(eligible_indices) == 0:
            logger.warning("No eligible contracts found for suspicious modification anomalies")
            return df
        
        selected_indices = np.random.choice(
            eligible_indices,
            size=min(n_anomalies, len(eligible_indices)),
            replace=False)
        
        for idx in selected_indices:
            original_duration = df.loc[idx, 'dureeMois']
            # Dramatically increase duration (simulate contract modification)
            new_duration = original_duration * random.uniform(2.5, 5.0)
            df.loc[idx, 'dureeMois'] = new_duration
        
        self.anomaly_labels['suspicious_modifications'][selected_indices] = True
        
        logger.info(f"Generated {len(selected_indices)} suspicious modification anomalies")
        return df
    

    
    def _generate_temporal_clustering_anomalies(self, df: pd.DataFrame, n_anomalies: int) -> pd.DataFrame:
        """Generate anomalies with suspicious temporal patterns."""
        
        # Convert date strings to datetime if needed
        if 'dateNotification' in df.columns:
            df['dateNotification'] = pd.to_datetime(df['dateNotification'], errors='coerce')
        
        # Find buyer-supplier pairs
        buyer_supplier_pairs = df.groupby(['acheteur_id', 'titulaire_id']).size()
        eligible_pairs = buyer_supplier_pairs[buyer_supplier_pairs >= 3].index.tolist()
        
        if len(eligible_pairs) == 0:
            logger.warning("No eligible buyer-supplier pairs found for temporal clustering anomalies")
            return df
        
        # Select pairs and cluster their contracts in time
        selected_pairs = random.sample(
            eligible_pairs,
            min(len(eligible_pairs), n_anomalies // 3))
        
        anomaly_count = 0
        for buyer_id, supplier_id in selected_pairs:
            pair_contracts = df[(df['acheteur_id'] == buyer_id) & 
                              (df['titulaire_id'] == supplier_id)]
            
            # Pick a random date and cluster contracts around it
            base_date = datetime(2023, random.randint(1, 12), random.randint(1, 28))
            
            for i, idx in enumerate(pair_contracts.index[:3]):  # Cluster up to 3 contracts
                # Contracts within 2 weeks of each other
                clustered_date = base_date + timedelta(days=random.randint(0, 14))
                df.loc[idx, 'dateNotification'] = clustered_date
                
                self.anomaly_labels['temporal_clustering'][idx] = True
                anomaly_count += 1
        
        logger.info(f"Generated {anomaly_count} temporal clustering anomalies")
        return df
    
    def _generate_excessive_subcontracting_anomalies(self, df: pd.DataFrame, n_anomalies: int) -> pd.DataFrame:
        """Generate anomalies with excessive subcontracting patterns."""
        
        # Find contracts that currently don't declare subcontracting
        mask = (df['sousTraitanceDeclaree'].notna() & 
                (df['sousTraitanceDeclaree'] == 0))
        eligible_indices = df[mask].index.tolist()
        
        if len(eligible_indices) == 0:
            logger.warning("No eligible contracts found for excessive subcontracting anomalies")
            return df
        
        selected_indices = np.random.choice(
            eligible_indices,
            size=min(n_anomalies, len(eligible_indices)),
            replace=False)
        
        # Mark these as having declared subcontracting
        df.loc[selected_indices, 'sousTraitanceDeclaree'] = 1
        
        self.anomaly_labels['excessive_subcontracting'][selected_indices] = True
        
        logger.info(f"Generated {len(selected_indices)} excessive subcontracting anomalies")
        return df
    
    def _generate_unusual_duration_anomalies(self, df: pd.DataFrame, n_anomalies: int) -> pd.DataFrame:
        """Generate anomalies with unusually short or long contract durations."""
        
        mask = df['dureeMois'].notna() & (df['dureeMois'] > 0)
        eligible_indices = df[mask].index.tolist()
        
        if len(eligible_indices) == 0:
            logger.warning("No eligible contracts found for unusual duration anomalies")
            return df
        
        selected_indices = np.random.choice(
            eligible_indices,
            size=min(n_anomalies, len(eligible_indices)),
            replace=False)
        
        for idx in selected_indices:
            # Randomly choose very short (< 1 month) or very long (> 10 years)
            if random.random() < 0.5:
                df.loc[idx, 'dureeMois'] = random.uniform(0.1, 0.9)  # Very short
            else:
                df.loc[idx, 'dureeMois'] = random.uniform(120, 240)  # Very long (10-20 years)
        
        self.anomaly_labels['unusual_contract_duration'][selected_indices] = True
        
        logger.info(f"Generated {len(selected_indices)} unusual duration anomalies")
        return df
    
    def _generate_suspicious_pairs_anomalies(self, df: pd.DataFrame, n_anomalies: int) -> pd.DataFrame:
        """Generate anomalies with suspicious buyer-supplier relationship patterns."""
        
        # Create artificial "suspicious" relationships by making certain pairs 
        # win disproportionately high amounts
        buyer_supplier_amounts = df.groupby(['acheteur_id', 'titulaire_id'])['montant'].agg(['sum', 'count']).reset_index()
        eligible_pairs = buyer_supplier_amounts[buyer_supplier_amounts['count'] >= 2]
        
        if len(eligible_pairs) == 0:
            logger.warning("No eligible buyer-supplier pairs found for suspicious pairs anomalies")
            return df
        
        # Select pairs and inflate their contract amounts
        selected_pairs = eligible_pairs.sample(min(len(eligible_pairs), n_anomalies // 2))
        
        anomaly_count = 0
        for _, row in selected_pairs.iterrows():
            buyer_id = row['acheteur_id']
            supplier_id = row['titulaire_id']
            
            pair_contracts = df[(df['acheteur_id'] == buyer_id) & 
                              (df['titulaire_id'] == supplier_id)]
            
            # Inflate amounts for this pair
            for idx in pair_contracts.index:
                original_amount = df.loc[idx, 'montant']
                if pd.notna(original_amount) and original_amount > 0:
                    df.loc[idx, 'montant'] = original_amount * random.uniform(1.5, 3.0)
                    
                    self.anomaly_labels['suspicious_buyer_supplier_pairs'][idx] = True
                    anomaly_count += 1
        
        logger.info(f"Generated {anomaly_count} suspicious buyer-supplier pair anomalies")
        return df
    
    def get_anomaly_summary(self) -> pd.DataFrame:
        """Get a summary of generated anomalies."""
        
        summary_data = []
        for anomaly_type, labels in self.anomaly_labels.items():
            summary_data.append({
                'anomaly_type': anomaly_type,
                'count': np.sum(labels),
                'percentage': np.sum(labels) / len(labels) * 100 if len(labels) > 0 else 0
            })
        
        return pd.DataFrame(summary_data).sort_values('count', ascending=False)
    
    def save_anomaly_labels(self, filepath: str):
        """Save anomaly labels to a file."""
        
        # Convert boolean arrays to a DataFrame
        labels_df = pd.DataFrame(self.anomaly_labels)
        labels_df.to_csv(filepath, index=True)
        logger.info(f"Anomaly labels saved to {filepath}")
    
    def load_anomaly_labels(self, filepath: str):
        """Load anomaly labels from a file."""
        
        labels_df = pd.read_csv(filepath, index_col=0)
        self.anomaly_labels = {}
        for col in labels_df.columns:
            self.anomaly_labels[col] = labels_df[col].values.astype(bool)
        logger.info(f"Anomaly labels loaded from {filepath}")


def demonstrate_anomaly_generation(df: pd.DataFrame, sample_size: int = 1000) -> None:
    """Demonstrate the anomaly generation functionality."""
    
    print("=== Synthetic Anomaly Generation Demonstration ===\n")
    
    # Take a sample for demonstration
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Using a sample of {sample_size} contracts for demonstration")
    else:
        df_sample = df.copy()
        print(f"Using full dataset of {len(df)} contracts")
    
    # Initialize generator
    generator = SyntheticAnomalyGenerator(random_seed=42)
    
    # Generate anomalies
    df_with_anomalies, anomaly_labels = generator.generate_anomalies(
        df_sample,
        anomaly_percentage=0.10,  # 10% anomalies
        anomaly_types=['single_bid_competitive', 'price_manipulation', 
                      'high_market_concentration', 'procedure_manipulation']
    )
    
    # Show summary
    summary = generator.get_anomaly_summary()
    print("\nAnomaly Generation Summary:")
    print(summary.to_string(index=False))
    
    # Show some examples
    print(f"\nTotal synthetic anomalies: {np.sum(df_with_anomalies['is_synthetic_anomaly'])}")
    print(f"Percentage of dataset: {np.sum(df_with_anomalies['is_synthetic_anomaly'])/len(df_with_anomalies)*100:.2f}%")
    
    # Show examples of each anomaly type
    print("\nExamples of generated anomalies:")
    for anomaly_type in ['single_bid_competitive', 'price_manipulation']:
        if anomaly_type in anomaly_labels:
            anomalous_rows = df_with_anomalies[anomaly_labels[anomaly_type]]
            if len(anomalous_rows) > 0:
                print(f"\n{anomaly_type.upper()} example:")
                relevant_cols = ['acheteur_id', 'titulaire_id', 'procedure', 'montant', 'offresRecues']
                available_cols = [col for col in relevant_cols if col in anomalous_rows.columns]
                print(anomalous_rows[available_cols].head(1).to_string(index=False))


if __name__ == "__main__":
    # Example usage
    print("Synthetic Anomaly Generator - Example Usage")
    print("This module should be imported and used with your procurement dataset.")
    print("See the demonstrate_anomaly_generation() function for usage examples.")