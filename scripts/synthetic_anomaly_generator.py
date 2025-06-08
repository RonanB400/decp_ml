"""
Synthetic Anomaly Generator for Public Procurement Data (Version 2)

This module creates synthetic anomalies by adding new rows to procurement 
datasets to test the effectiveness of anomaly detection models.

The anomalies are based on red flags identified in procurement literature
and real-world corruption patterns.

Author: RonanB400
Date: January 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticAnomalyGenerator:
    """Generate synthetic anomalies by adding new rows to procurement data."""
    
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
                          anomaly_percentage: float = 0.05) -> pd.DataFrame:
        """Generate synthetic anomalies by adding new rows to the dataset.
        
        Args:
            df: Original procurement dataframe
            anomaly_types: List of anomaly types to generate
            anomaly_percentage: Percentage of synthetic anomalies relative to 
                original data
            
        Returns:
            DataFrame with added anomalous rows and is_synthetic_anomaly column
        """
        
        if anomaly_types is None:
            anomaly_types = [
                'single_bid_competitive',
                'price_inflation',
                'price_deflation',
                'procedure_manipulation',
                'suspicious_modifications',
                'high_market_concentration',
                'temporal_clustering',
                'excessive_subcontracting',
                'short_contract_duration',
                'suspicious_buyer_supplier_pairs'
            ]
        
        # Start with original dataframe
        df_original = df.copy()
        
        # Calculate number of anomalies to create per type
        total_anomalies = int(len(df_original) * anomaly_percentage)
        anomalies_per_type = max(1, total_anomalies // len(anomaly_types))
        
        logger.info(f"Generating {total_anomalies} total synthetic anomaly "
                    f"rows")
        logger.info(f"Approximately {anomalies_per_type} anomalies per type")
        
        # Store all new anomalous rows
        all_anomalous_rows = []
        
        # Method mapping for cleaner code
        method_map = {
            'single_bid_competitive': self._generate_single_bid_anomalies,
            'price_inflation': self._generate_price_inflation_anomalies,
            'price_deflation': self._generate_price_deflation_anomalies,
            'procedure_manipulation': (
                self._generate_procedure_manipulation_anomalies),
            'suspicious_modifications': (
                self._generate_suspicious_modification_anomalies),
            'high_market_concentration': (
                self._generate_high_market_concentration_anomalies),
            'temporal_clustering': (
                self._generate_temporal_clustering_anomalies),
            'excessive_subcontracting': (
                self._generate_excessive_subcontracting_anomalies),
            'short_contract_duration': self._generate_short_duration_anomalies,
            'suspicious_buyer_supplier_pairs': (
                self._generate_suspicious_pairs_anomalies)
        }
        
        # Generate each type of anomaly
        for anomaly_type in anomaly_types:
            if anomaly_type in method_map:
                logger.info(f"Generating {anomaly_type} anomalies...")
                new_rows = method_map[anomaly_type](df_original,
                                                    anomalies_per_type,
                                                    anomaly_type)
                if len(new_rows) > 0:
                    all_anomalous_rows.extend(new_rows)
        
        # Combine original data with synthetic anomalies
        if all_anomalous_rows:
            anomalous_df = pd.DataFrame(all_anomalous_rows)
            df_combined = pd.concat([df_original, anomalous_df],
                                    ignore_index=True)
        else:
            df_combined = df_original.copy()
        
        # Create anomaly labels for the combined dataset
        n_original = len(df_original)
        n_total = len(df_combined)
        
        # Initialize labels - original rows are False, synthetic rows will be
        # True
        self.anomaly_labels = {}
        for anomaly_type in anomaly_types:
            self.anomaly_labels[anomaly_type] = np.zeros(n_total, dtype=bool)
        
        # Create anomaly type mapping (0 = no anomaly, 1+ = specific anomaly
        # types)
        anomaly_type_mapping = {anomaly_type: i + 1
                                for i, anomaly_type in enumerate(anomaly_types)}
        
        # Store the mapping as instance variable for easy access
        self.anomaly_type_mapping = anomaly_type_mapping
        
        # Add general anomaly indicator column (0 = no anomaly, 1+ = anomaly
        # type number)
        df_combined['is_synthetic_anomaly'] = np.zeros(len(df_combined),
                                                       dtype=int)
        
        # Mark synthetic anomalies
        current_idx = n_original
        for row_data in all_anomalous_rows:
            if 'anomaly_type' in row_data:
                anomaly_type = row_data['anomaly_type']
                if anomaly_type in self.anomaly_labels:
                    self.anomaly_labels[anomaly_type][current_idx] = True
                    # Set the anomaly type number in the indicator column
                    if anomaly_type in anomaly_type_mapping:
                        df_combined.loc[current_idx, 'is_synthetic_anomaly'] = (
                            anomaly_type_mapping[anomaly_type])
            current_idx += 1
        
        # Log summary
        total_synthetic_anomalies = np.sum(
            df_combined['is_synthetic_anomaly'] > 0)
        logger.info(f"Generated {total_synthetic_anomalies} total synthetic "
                    f"anomaly rows")
        logger.info(f"Original dataset: {n_original} rows")
        logger.info(f"Combined dataset: {len(df_combined)} rows "
                    f"({total_synthetic_anomalies/len(df_combined)*100:.2f}% "
                    f"synthetic)")
        
        # Log anomaly type mapping
        logger.info("Anomaly type mapping:")
        for anomaly_type, type_number in anomaly_type_mapping.items():
            count = np.sum(self.anomaly_labels[anomaly_type])
            logger.info(f"  - {type_number}: {anomaly_type} "
                        f"({count} anomalies)")
        
        return df_combined
    
    def _generate_single_bid_anomalies(self, df: pd.DataFrame, n_anomalies: int, anomaly_type: str) -> List[Dict]:
        """Generate new rows with single bid competitive anomalies."""
        
        # Find competitive procedures with more than 1 bid as templates
        competitive_procedures = ["Appel d'offres ouvert", "Appel d'offres restreint"]
        
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
    
    def _generate_price_inflation_anomalies(self, df: pd.DataFrame, n_anomalies: int, anomaly_type: str) -> List[Dict]:
        """Generate new rows with artificially inflated prices."""
        
        # Find contracts with valid amounts as templates
        mask = df['montant'].notna() & (df['montant'] > 0)
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for price inflation anomalies")
            return []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(n=min(n_anomalies, len(eligible_rows)), random_state=self.random_seed)
        
        new_rows = []
        for _, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            original_amount = new_row['montant']
            
            # Inflate by 200-500%
            multiplier = random.uniform(3.0, 6.0)

            new_row['montant'] = original_amount * multiplier
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
        
        logger.info(f"Generated {len(new_rows)} price inflation anomaly rows")
        return new_rows
    
    def _generate_price_deflation_anomalies(self, df: pd.DataFrame, n_anomalies: int, anomaly_type: str) -> List[Dict]:
        """Generate new rows with artificially deflated prices."""
        
        # Find contracts with valid amounts as templates
        mask = df['montant'].notna() & (df['montant'] > 0)
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for price deflation anomalies")
            return []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(n=min(n_anomalies, len(eligible_rows)), random_state=self.random_seed)
        
        new_rows = []
        for _, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            original_amount = new_row['montant']
            
            # Deflate to 10-30% of original
            multiplier = random.uniform(0.1, 0.3)
            new_row['montant'] = original_amount * multiplier
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
        
        logger.info(f"Generated {len(new_rows)} price deflation anomaly rows")
        return new_rows
    
    def _generate_procedure_manipulation_anomalies(self, df: pd.DataFrame, n_anomalies: int, anomaly_type: str) -> List[Dict]:
        """Generate new rows with suspicious procedure manipulation."""
        
        # Find competitive procedures as templates
        competitive_procedures = ["Appel d'offres ouvert", "Appel d'offres restreint"]
        mask = df['procedure'].isin(competitive_procedures)
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for procedure manipulation anomalies")
            return []
        
        # Non-competitive procedures to switch to
        non_competitive = ['Procédure adaptée', 'Marché négocié sans publicité']
        
        # Select random template rows
        selected_rows = eligible_rows.sample(n=min(n_anomalies, len(eligible_rows)), random_state=self.random_seed)
        
        new_rows = []
        for _, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            
            # Switch to non-competitive procedure
            new_row['procedure'] = random.choice(non_competitive)
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
        
        logger.info(f"Generated {len(new_rows)} procedure manipulation anomaly rows")
        return new_rows
    
    def _generate_suspicious_modification_anomalies(self, df: pd.DataFrame, n_anomalies: int, anomaly_type: str) -> List[Dict]:
        """Generate new rows suggesting suspicious contract modifications."""
        
        # Find contracts with reasonable duration as templates
        mask = df['dureeMois'].notna() & (df['dureeMois'] > 0) & (df['dureeMois'] < 36)
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for suspicious modification anomalies")
            return []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(n=min(n_anomalies, len(eligible_rows)), random_state=self.random_seed)
        
        new_rows = []
        for _, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            original_duration = new_row['dureeMois']
            
            # Dramatically increase duration (simulate contract modification)
            new_duration = original_duration * random.uniform(2.5, 5.0)
            new_row['dureeMois'] = new_duration
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
        
        logger.info(f"Generated {len(new_rows)} suspicious modification anomaly rows")
        return new_rows
    
    def _generate_high_market_concentration_anomalies(self, df: pd.DataFrame, n_anomalies: int, anomaly_type: str) -> List[Dict]:
        """Generate new rows with high market concentration anomalies.
        
        Creates anomalies where a single supplier dominates a buyer's contracts,
        indicating potential market concentration issues.
        """
        
        # Find buyer-CPV combinations with multiple suppliers as templates
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
        
        logger.info(f"Generated {len(new_rows)} high market concentration anomaly rows")
        return new_rows
    
    def _generate_temporal_clustering_anomalies(self, df: pd.DataFrame, n_anomalies: int, anomaly_type: str) -> List[Dict]:
        """Generate new rows with suspicious temporal clustering patterns."""
        
        # Find buyer-supplier pairs with multiple contracts as templates
        buyer_supplier_pairs = df.groupby(['acheteur_id', 'titulaire_id']).size()
        eligible_pairs = buyer_supplier_pairs[buyer_supplier_pairs >= 3].index.tolist()
        
        if len(eligible_pairs) == 0:
            logger.warning("No eligible buyer-supplier pairs found for temporal clustering anomalies")
            return []
        
        # Select pairs to create clustered contracts from
        selected_pairs = random.sample(
            eligible_pairs,
            min(len(eligible_pairs), n_anomalies // 3))
        
        new_rows = []
        for buyer_id, supplier_id in selected_pairs:
            pair_contracts = df[(df['acheteur_id'] == buyer_id) & 
                              (df['titulaire_id'] == supplier_id)]
            
            # Pick a template contract and create clustered ones
            template_contracts = pair_contracts.sample(n=min(3, len(pair_contracts)), random_state=self.random_seed)
            
            # Pick a random date and cluster contracts around it
            base_date = datetime(2023, random.randint(1, 12), random.randint(1, 28))
            
            for i, (_, row) in enumerate(template_contracts.iterrows()):
                new_row = row.copy()
                
                # Create clustered date (within 2 weeks of each other)
                clustered_date = base_date + timedelta(days=random.randint(0, 14))
                new_row['dateNotification'] = clustered_date.strftime('%Y-%m-%d')
                
                # Add anomaly metadata
                new_row['anomaly_type'] = anomaly_type
                new_row['source_type'] = 'synthetic'
                
                new_rows.append(new_row.to_dict())
        
        logger.info(f"Generated {len(new_rows)} temporal clustering anomaly rows")
        return new_rows
    
    def _generate_excessive_subcontracting_anomalies(self, df: pd.DataFrame, n_anomalies: int, anomaly_type: str) -> List[Dict]:
        """Generate new rows with excessive subcontracting patterns."""
        
        # Find contracts that currently don't declare subcontracting as templates
        mask = (df['sousTraitanceDeclaree'].notna() & 
                (df['sousTraitanceDeclaree'] == 0))
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for excessive subcontracting anomalies")
            return []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(n=min(n_anomalies, len(eligible_rows)), random_state=self.random_seed)
        
        new_rows = []
        for _, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            
            # Make it anomalous: mark as having declared subcontracting
            new_row['sousTraitanceDeclaree'] = 1
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
        
        logger.info(f"Generated {len(new_rows)} excessive subcontracting anomaly rows")
        return new_rows
    
    def _generate_short_duration_anomalies(self, df: pd.DataFrame, n_anomalies: int, anomaly_type: str) -> List[Dict]:
        """Generate new rows with unusually short contract durations."""
        
        # Find contracts with valid duration (> 1 month) as templates
        mask = df['dureeMois'].notna() & (df['dureeMois'] > 1)
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for unusual duration anomalies")
            return []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(n=min(n_anomalies, len(eligible_rows)), random_state=self.random_seed)
        
        new_rows = []
        for _, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            
            # Make it anomalous: set to very short duration (< 1 month)
            new_row['dureeMois'] = random.uniform(0.1, 0.9)
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
        
        logger.info(f"Generated {len(new_rows)} unusual short duration anomaly rows")
        return new_rows
    
    def _generate_suspicious_pairs_anomalies(self, df: pd.DataFrame, n_anomalies: int, anomaly_type: str) -> List[Dict]:
        """Generate new rows with suspicious buyer-supplier relationship patterns."""
        
        # Find buyer-supplier pairs with multiple contracts and valid amounts
        buyer_supplier_amounts = df.groupby(['acheteur_id', 'titulaire_id']).agg({
            'montant': ['sum', 'count']
        }).reset_index()
        buyer_supplier_amounts.columns = ['acheteur_id', 'titulaire_id', 'sum_montant', 'count_contracts']
        
        # Filter for pairs with at least 2 contracts and valid amounts
        eligible_pairs = buyer_supplier_amounts[
            (buyer_supplier_amounts['count_contracts'] >= 2) & 
            (buyer_supplier_amounts['sum_montant'].notna()) &
            (buyer_supplier_amounts['sum_montant'] > 0)
        ]
        
        if len(eligible_pairs) == 0:
            logger.warning("No eligible buyer-supplier pairs found for suspicious pairs anomalies")
            return []
        
        # Select pairs to create suspicious relationships from
        selected_pairs = eligible_pairs.sample(min(len(eligible_pairs), n_anomalies // 2), random_state=self.random_seed)
        
        new_rows = []
        for _, row in selected_pairs.iterrows():
            buyer_id = row['acheteur_id']
            supplier_id = row['titulaire_id']
            
            # Find contracts for this pair as templates
            pair_contracts = df[(df['acheteur_id'] == buyer_id) & 
                              (df['titulaire_id'] == supplier_id)]
            
            # Create new contracts with inflated amounts for this pair
            template_contracts = pair_contracts.sample(n=min(2, len(pair_contracts)), random_state=self.random_seed)
            
            for _, template_row in template_contracts.iterrows():
                new_row = template_row.copy()
                
                # Make it anomalous: inflate the amount significantly
                original_amount = new_row['montant']
                if pd.notna(original_amount) and original_amount > 0:
                    new_row['montant'] = original_amount * random.uniform(1.5, 3.0)
                
                # Add anomaly metadata
                new_row['anomaly_type'] = anomaly_type
                new_row['source_type'] = 'synthetic'
                
                new_rows.append(new_row.to_dict())
        
        logger.info(f"Generated {len(new_rows)} suspicious buyer-supplier pair anomaly rows")
        return new_rows
    
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
    
    def get_anomaly_type_mapping(self) -> Dict[str, int]:
        """Get the mapping between anomaly types and their numeric codes.
        
        Returns:
            Dictionary mapping anomaly type names to numeric codes (1-N)
        """
        return getattr(self, 'anomaly_type_mapping', {})
    
    def get_reverse_anomaly_mapping(self) -> Dict[int, str]:
        """Get the reverse mapping from numeric codes to anomaly type names.
        
        Returns:
            Dictionary mapping numeric codes to anomaly type names
        """
        mapping = self.get_anomaly_type_mapping()
        return {v: k for k, v in mapping.items()}


def demonstrate_anomaly_generation(df: pd.DataFrame, sample_size: int = 1000) -> None:
    """Demonstrate the anomaly generation functionality."""
    
    print("=== Synthetic Anomaly Generation Demonstration (V2) ===\n")
    
    # Take a sample for demonstration
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Using a sample of {sample_size} contracts for demonstration")
    else:
        df_sample = df.copy()
        print(f"Using full dataset of {len(df)} contracts")
    
    # Initialize generator
    generator = SyntheticAnomalyGenerator(random_seed=42)
    
    # Generate anomalies (adding new rows)
    df_with_anomalies = generator.generate_anomalies(
        df_sample,
        anomaly_percentage=0.10,  # 10% anomalies
        anomaly_types=['single_bid_competitive', 'price_manipulation']
    )
    
    # Show summary
    summary = generator.get_anomaly_summary()
    print("\nAnomaly Generation Summary:")
    print(summary.to_string(index=False))
    
    # Show some examples
    total_synthetic = np.sum(df_with_anomalies['is_synthetic_anomaly'])
    print(f"\nTotal synthetic anomaly rows: {total_synthetic}")
    print(f"Original rows: {len(df_sample)}")
    print(f"Combined dataset: {len(df_with_anomalies)} rows")
    print(f"Percentage synthetic: {total_synthetic/len(df_with_anomalies)*100:.2f}%")
    
    # Show examples of synthetic anomalies
    synthetic_rows = df_with_anomalies[df_with_anomalies['is_synthetic_anomaly'] > 0]
    if len(synthetic_rows) > 0:
        print("\nExample synthetic anomaly rows:")
        relevant_cols = ['acheteur_id', 'titulaire_id', 'procedure', 'montant', 'offresRecues', 'anomaly_type', 'is_synthetic_anomaly']
        available_cols = [col for col in relevant_cols if col in synthetic_rows.columns]
        print(synthetic_rows[available_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    # Example usage
    print("Synthetic Anomaly Generator V2 - Example Usage")
    print("This module creates synthetic anomalies by adding new rows to the dataset.")
    print("See the demonstrate_anomaly_generation() function for usage examples.") 