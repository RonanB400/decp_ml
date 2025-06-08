"""
Synthetic Anomaly Generator for Public Procurement Data

This module creates synthetic anomalies by either adding new rows or replacing
existing rows in procurement datasets to test the effectiveness of anomaly 
detection models.

The anomalies are based on red flags identified in procurement literature
and real-world corruption patterns.

Author: RonanB400
Date: January 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
import random
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class OriginalAnomalyAnalyzer:
    """Analyze original dataset for existing patterns similar to synthetic anomalies."""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_all_anomaly_types(self, df: pd.DataFrame) -> Dict:
        """Analyze all anomaly types in the original dataset.
        
        Args:
            df: Original dataframe (before synthetic anomalies)
            
        Returns:
            Dictionary with analysis results for each anomaly type
        """
        logger.info("Analyzing original dataset for existing anomaly patterns...")
        
        results = {
            'total_contracts': len(df),
            'anomaly_analysis': {}
        }
        
        # Analyze each anomaly type
        anomaly_functions = {
            'single_bid_competitive': self._analyze_single_bid_competitive,
            'price_inflation': self._analyze_price_inflation,
            'price_deflation': self._analyze_price_deflation,
            'procedure_manipulation': self._analyze_procedure_manipulation,
            'suspicious_modifications': self._analyze_suspicious_modifications,
            'high_market_concentration': self._analyze_high_market_concentration,
            'temporal_clustering': self._analyze_temporal_clustering,
            'excessive_subcontracting': self._analyze_excessive_subcontracting,
            'short_contract_duration': self._analyze_short_contract_duration,
            'suspicious_buyer_supplier_pairs': self._analyze_suspicious_pairs
        }
        
        for anomaly_type, analyze_func in anomaly_functions.items():
            try:
                analysis = analyze_func(df)
                results['anomaly_analysis'][anomaly_type] = analysis
                logger.info(f"{anomaly_type}: {analysis['count']} contracts "
                           f"({analysis['percentage']:.2f}%)")
            except Exception as e:
                logger.error(f"Error analyzing {anomaly_type}: {str(e)}")
                results['anomaly_analysis'][anomaly_type] = {
                    'count': 0, 'percentage': 0.0, 'error': str(e)
                }
        
        self.analysis_results = results
        return results
    
    def _analyze_single_bid_competitive(self, df: pd.DataFrame) -> Dict:
        """Analyze single bid competitive anomalies in original data."""
        competitive_procedures = ["Appel d'offres ouvert", 
                                "Appel d'offres restreint"]
        
        mask = (df['procedure'].isin(competitive_procedures) & 
                (df['offresRecues'] == 1) & 
                df['offresRecues'].notna())
        
        count = np.sum(mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': 'Competitive procedures with exactly 1 bid',
            'threshold_used': 'offresRecues == 1 AND competitive procedure'
        }
    
    def _analyze_price_inflation(self, df: pd.DataFrame) -> Dict:
        """Analyze price inflation anomalies in original data."""
        # Define inflation as top 2% of amounts
        if 'montant' not in df.columns or df['montant'].isna().all():
            return {'count': 0, 'percentage': 0.0, 'error': 'No amount data'}
        
        valid_amounts = df['montant'].dropna()
        if len(valid_amounts) == 0:
            return {'count': 0, 'percentage': 0.0, 'error': 'No valid amounts'}
        
        # Use 98th percentile as threshold for "inflated" prices
        inflation_threshold = np.percentile(valid_amounts, 98)
        mask = (df['montant'] > inflation_threshold) & df['montant'].notna()
        
        count = np.sum(mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': f'Contracts above 98th percentile (>{inflation_threshold:,.0f})',
            'threshold_used': f'montant > {inflation_threshold:,.0f}'
        }
    
    def _analyze_price_deflation(self, df: pd.DataFrame) -> Dict:
        """Analyze price deflation anomalies in original data."""
        if 'montant' not in df.columns or df['montant'].isna().all():
            return {'count': 0, 'percentage': 0.0, 'error': 'No amount data'}
        
        valid_amounts = df['montant'].dropna()
        if len(valid_amounts) == 0:
            return {'count': 0, 'percentage': 0.0, 'error': 'No valid amounts'}
        
        # Use 2nd percentile as threshold for "deflated" prices
        deflation_threshold = np.percentile(valid_amounts, 2)
        mask = (df['montant'] < deflation_threshold) & df['montant'].notna()
        
        count = np.sum(mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': f'Contracts below 2nd percentile (<{deflation_threshold:,.0f})',
            'threshold_used': f'montant < {deflation_threshold:,.0f}'
        }
    
    def _analyze_procedure_manipulation(self, df: pd.DataFrame) -> Dict:
        """Analyze procedure manipulation in original data."""
        # High-value contracts using non-competitive procedures
        if 'montant' not in df.columns or df['montant'].isna().all():
            return {'count': 0, 'percentage': 0.0, 'error': 'No amount data'}
        
        # Define high-value as top 25% of contracts
        high_value_threshold = np.percentile(df['montant'].dropna(), 75)
        
        non_competitive = ['Procédure adaptée', 'Marché négocié sans publicité']
        
        mask = (df['montant'] > high_value_threshold) & \
               df['procedure'].isin(non_competitive) & \
               df['montant'].notna()
        
        count = np.sum(mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': f'High-value contracts (>{high_value_threshold:,.0f}) using non-competitive procedures',
            'threshold_used': f'montant > {high_value_threshold:,.0f} AND non-competitive procedure'
        }
    
    def _analyze_suspicious_modifications(self, df: pd.DataFrame) -> Dict:
        """Analyze suspicious contract modifications in original data."""
        if 'dureeMois' not in df.columns or df['dureeMois'].isna().all():
            return {'count': 0, 'percentage': 0.0, 'error': 'No duration data'}
        
        valid_durations = df['dureeMois'].dropna()
        if len(valid_durations) == 0:
            return {'count': 0, 'percentage': 0.0, 'error': 'No valid durations'}
        
        # Define suspicious as top 5% of durations (very long contracts)
        long_duration_threshold = np.percentile(valid_durations, 95)
        mask = (df['dureeMois'] > long_duration_threshold) & df['dureeMois'].notna()
        
        count = np.sum(mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': f'Contracts with very long duration (>{long_duration_threshold:.1f} months)',
            'threshold_used': f'dureeMois > {long_duration_threshold:.1f}'
        }
    
    def _analyze_high_market_concentration(self, df: pd.DataFrame) -> Dict:
        """Analyze high market concentration in original data."""
        if 'acheteur_id' not in df.columns or 'codeCPV_3' not in df.columns or 'titulaire_id' not in df.columns:
            return {'count': 0, 'percentage': 0.0, 'error': 'Missing required columns'}
        
        # Find buyer-CPV combinations where one supplier has >70% of contracts
        buyer_cpv_supplier_counts = df.groupby(['acheteur_id', 'codeCPV_3', 'titulaire_id']).size()
        buyer_cpv_total_counts = df.groupby(['acheteur_id', 'codeCPV_3']).size()
        
        # Calculate supplier market share within each buyer-CPV combination
        supplier_shares = buyer_cpv_supplier_counts / buyer_cpv_total_counts
        
        # Find cases where supplier has >70% market share and >2 total contracts
        high_concentration_mask = (supplier_shares > 0.7) & (buyer_cpv_total_counts > 2)
        
        # Count contracts in these high-concentration scenarios
        high_concentration_combinations = high_concentration_mask[high_concentration_mask].index
        
        count = 0
        for (buyer, cpv, supplier) in high_concentration_combinations:
            count += len(df[(df['acheteur_id'] == buyer) & 
                           (df['codeCPV_3'] == cpv) & 
                           (df['titulaire_id'] == supplier)])
        
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': 'Contracts in buyer-CPV combinations with >70% supplier concentration',
            'threshold_used': 'supplier_share > 0.7 AND total_contracts > 2'
        }
    
    def _analyze_temporal_clustering(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal clustering in original data."""
        if 'dateNotification' not in df.columns:
            return {'count': 0, 'percentage': 0.0, 'error': 'No date data'}
        
        # Convert dates
        try:
            df_temp = df.copy()
            df_temp['date_parsed'] = pd.to_datetime(df_temp['dateNotification'], errors='coerce')
            df_temp = df_temp.dropna(subset=['date_parsed', 'acheteur_id', 'titulaire_id'])
            
            if len(df_temp) == 0:
                return {'count': 0, 'percentage': 0.0, 'error': 'No valid dates'}
            
            # Group by buyer-supplier pairs
            buyer_supplier_groups = df_temp.groupby(['acheteur_id', 'titulaire_id'])
            
            clustered_contracts = 0
            for (buyer, supplier), group in buyer_supplier_groups:
                if len(group) >= 3:  # Need at least 3 contracts to detect clustering
                    dates = sorted(group['date_parsed'])
                    
                    # Check for clustering: 3+ contracts within 30 days
                    for i in range(len(dates) - 2):
                        if (dates[i+2] - dates[i]).days <= 30:
                            clustered_contracts += 3
                            break
            
            percentage = (clustered_contracts / len(df)) * 100
            
            return {
                'count': clustered_contracts,
                'percentage': percentage,
                'description': 'Contracts in buyer-supplier pairs with 3+ contracts within 30 days',
                'threshold_used': '3+ contracts within 30 days for same buyer-supplier pair'
            }
        except Exception as e:
            return {'count': 0, 'percentage': 0.0, 'error': f'Date parsing error: {str(e)}'}
    
    def _analyze_excessive_subcontracting(self, df: pd.DataFrame) -> Dict:
        """Analyze excessive subcontracting in original data."""
        if 'sousTraitanceDeclaree' not in df.columns:
            return {'count': 0, 'percentage': 0.0, 'error': 'No subcontracting data'}
        
        # Contracts declaring subcontracting
        mask = (df['sousTraitanceDeclaree'] == 1) & df['sousTraitanceDeclaree'].notna()
        count = np.sum(mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': 'Contracts declaring subcontracting',
            'threshold_used': 'sousTraitanceDeclaree == 1'
        }
    
    def _analyze_short_contract_duration(self, df: pd.DataFrame) -> Dict:
        """Analyze short contract duration in original data."""
        if 'dureeMois' not in df.columns or df['dureeMois'].isna().all():
            return {'count': 0, 'percentage': 0.0, 'error': 'No duration data'}
        
        # Very short contracts (< 1 month)
        mask = (df['dureeMois'] < 1) & (df['dureeMois'] > 0) & df['dureeMois'].notna()
        count = np.sum(mask)
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': 'Contracts with duration < 1 month',
            'threshold_used': 'dureeMois < 1'
        }
    
    def _analyze_suspicious_pairs(self, df: pd.DataFrame) -> Dict:
        """Analyze suspicious buyer-supplier pairs in original data."""
        if 'montant' not in df.columns or df['montant'].isna().all():
            return {'count': 0, 'percentage': 0.0, 'error': 'No amount data'}
        
        # Calculate total amounts per buyer-supplier pair
        pair_totals = df.groupby(['acheteur_id', 'titulaire_id'])['montant'].agg(['sum', 'count'])
        
        # Define suspicious as top 1% of total amounts with 2+ contracts
        suspicious_threshold = np.percentile(pair_totals['sum'].dropna(), 99)
        suspicious_pairs = pair_totals[(pair_totals['sum'] > suspicious_threshold) & 
                                     (pair_totals['count'] >= 2)]
        
        # Count contracts in suspicious pairs
        count = 0
        for (buyer, supplier), _ in suspicious_pairs.iterrows():
            count += len(df[(df['acheteur_id'] == buyer) & 
                           (df['titulaire_id'] == supplier)])
        
        percentage = (count / len(df)) * 100
        
        return {
            'count': count,
            'percentage': percentage,
            'description': f'Contracts in buyer-supplier pairs with total amount >{suspicious_threshold:,.0f}',
            'threshold_used': f'total_pair_amount > {suspicious_threshold:,.0f} AND contract_count >= 2'
        }
    
    def print_analysis_summary(self, results: Dict = None):
        """Print a comprehensive summary of the original dataset analysis."""
        if results is None:
            results = self.analysis_results
        
        if not results:
            logger.error("No analysis results available. Run analyze_all_anomaly_types first.")
            return
        
        print("\n" + "="*80)
        print("ORIGINAL DATASET ANOMALY PATTERN ANALYSIS")
        print("="*80)
        print(f"Total contracts analyzed: {results['total_contracts']:,}")
        print(f"\nAnalyzing prevalence of synthetic anomaly patterns in original data:")
        print("-" * 80)
        
        # Sort by percentage descending
        anomaly_data = []
        for anomaly_type, analysis in results['anomaly_analysis'].items():
            if 'error' not in analysis:
                anomaly_data.append((anomaly_type, analysis))
        
        anomaly_data.sort(key=lambda x: x[1]['percentage'], reverse=True)
        
        print(f"{'Anomaly Type':<35} {'Count':<10} {'Percentage':<12} {'Status':<15}")
        print("-" * 80)
        
        total_flagged = 0
        for anomaly_type, analysis in anomaly_data:
            count = analysis['count']
            percentage = analysis['percentage']
            total_flagged += count
            
            # Determine status based on prevalence
            if percentage > 10:
                status = "🔴 Very High"
            elif percentage > 5:
                status = "🟡 High" 
            elif percentage > 1:
                status = "🟠 Medium"
            elif percentage > 0.1:
                status = "🟢 Low"
            else:
                status = "✅ Very Low"
            
            type_name = anomaly_type.replace('_', ' ').title()
            print(f"{type_name:<35} {count:<10,} {percentage:<12.2f}% {status:<15}")
        
        print("-" * 80)
        print(f"{'TOTAL FLAGGED (with overlap)':<35} {total_flagged:<10,} {(total_flagged/results['total_contracts']*100):<12.1f}%")
        
        # Print detailed descriptions
        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS")
        print(f"{'='*80}")
        
        for anomaly_type, analysis in anomaly_data:
            print(f"\n{anomaly_type.replace('_', ' ').title()}:")
            print(f"  Count: {analysis['count']:,} contracts ({analysis['percentage']:.2f}%)")
            print(f"  Description: {analysis['description']}")
            print(f"  Threshold: {analysis['threshold_used']}")
        
        # Print errors if any
        errors = [(k, v) for k, v in results['anomaly_analysis'].items() if 'error' in v]
        if errors:
            print(f"\n{'='*80}")
            print("ANALYSIS ERRORS")
            print(f"{'='*80}")
            for anomaly_type, analysis in errors:
                print(f"{anomaly_type}: {analysis['error']}")
        
        # Recommendations
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        
        high_prevalence = [name for name, analysis in anomaly_data 
                          if analysis['percentage'] > 5]
        medium_prevalence = [name for name, analysis in anomaly_data 
                            if 1 < analysis['percentage'] <= 5]
        
        if high_prevalence:
            print("🔴 HIGH PREVALENCE PATTERNS (>5%):")
            for pattern in high_prevalence:
                print(f"  • {pattern.replace('_', ' ').title()}")
            print("   → These patterns are very common in your dataset.")
            print("   → Synthetic anomalies of these types may not be distinguishable.")
            print("   → Consider removing these from synthetic anomaly generation.")
        
        if medium_prevalence:
            print("\n🟠 MEDIUM PREVALENCE PATTERNS (1-5%):")
            for pattern in medium_prevalence:
                print(f"  • {pattern.replace('_', ' ').title()}")
            print("   → These patterns exist but are less common.")
            print("   → Synthetic anomalies might still be learnable but with reduced signal.")
        
        low_prevalence = [name for name, analysis in anomaly_data 
                         if analysis['percentage'] <= 1]
        if low_prevalence:
            print("\n✅ LOW PREVALENCE PATTERNS (<1%):")
            for pattern in low_prevalence:
                print(f"  • {pattern.replace('_', ' ').title()}")
            print("   → These patterns are rare in your dataset.")
            print("   → Synthetic anomalies of these types should be most learnable.")
        
        print(f"\n💡 SUMMARY:")
        total_percentage = sum(analysis['percentage'] for _, analysis in anomaly_data)
        print(f"   • Total coverage of anomaly patterns: {total_percentage:.1f}% (with overlap)")
        print(f"   • High prevalence patterns: {len(high_prevalence)}/{len(anomaly_data)}")
        print(f"   → Focus synthetic training on low-prevalence patterns for best results.")


def analyze_original_dataset_anomalies(df: pd.DataFrame) -> Dict:
    """Convenience function to analyze original dataset for anomaly patterns.
    
    Args:
        df: Original dataframe (before synthetic anomalies)
        
    Returns:
        Dictionary with analysis results
        
    Usage:
        # Load your original dataset
        df_original = pd.read_csv('your_data.csv')
        
        # Analyze existing anomaly patterns
        results = analyze_original_dataset_anomalies(df_original)
    """
    analyzer = OriginalAnomalyAnalyzer()
    results = analyzer.analyze_all_anomaly_types(df)
    analyzer.print_analysis_summary(results)
    return results



class SyntheticAnomalyGenerator:
    """Generate synthetic anomalies by adding new rows or replacing existing 
    rows in procurement data."""
    
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
        self.replaced_indices = set()
        
    def generate_anomalies(self,
                           df: pd.DataFrame,
                           anomaly_types: List[str] = None,
                           anomaly_percentage: float = 0.05,
                           replace_rows: bool = True) -> pd.DataFrame:
        """Generate synthetic anomalies by adding new rows or replacing 
        existing ones.
        
        Args:
            df: Original procurement dataframe
            anomaly_types: List of anomaly types to generate
            anomaly_percentage: Percentage of synthetic anomalies relative to 
                original data
            replace_rows: If True, replace existing rows instead of adding 
                new ones
            
        Returns:
            DataFrame with anomalous rows and is_synthetic_anomaly column
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
        if replace_rows:
            # When replacing, we can't exceed the original dataset size
            total_anomalies = min(int(len(df_original) * anomaly_percentage),
                                len(df_original))
        else:
            total_anomalies = int(len(df_original) * anomaly_percentage)
            
        anomalies_per_type = max(1, total_anomalies // len(anomaly_types))
        
        mode_str = "replacing" if replace_rows else "adding"
        logger.info(f"Generating {total_anomalies} total synthetic anomaly "
                    f"rows by {mode_str}")
        logger.info(f"Approximately {anomalies_per_type} anomalies per type")
        
        # Store all new anomalous rows and their corresponding indices
        all_anomalous_data = []
        self.replaced_indices = set()
        
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
                new_rows, template_indices = method_map[anomaly_type](
                    df_original, anomalies_per_type, anomaly_type)
                
                if len(new_rows) > 0:
                    for row, template_idx in zip(new_rows, template_indices):
                        all_anomalous_data.append({
                            'row_data': row,
                            'template_index': template_idx,
                            'anomaly_type': anomaly_type
                        })
                        if replace_rows:
                            self.replaced_indices.add(template_idx)
        
        # Create the final dataset
        if replace_rows:
            df_result = self._create_replacement_dataset(
                df_original, all_anomalous_data, anomaly_types)
        else:
            df_result = self._create_additive_dataset(
                df_original, all_anomalous_data, anomaly_types)
        
        # Log summary
        total_synthetic_anomalies = np.sum(
            df_result['is_synthetic_anomaly'] > 0)
        logger.info(f"Generated {total_synthetic_anomalies} total synthetic "
                    f"anomaly rows")
        logger.info(f"Original dataset: {len(df_original)} rows")
        logger.info(f"Final dataset: {len(df_result)} rows")
        
        if replace_rows:
            logger.info(f"Replaced {len(self.replaced_indices)} original "
                       f"rows")
        else:
            percentage = (total_synthetic_anomalies / len(df_result) * 100)
            logger.info(f"({percentage:.2f}% synthetic)")
        
        # Log anomaly type mapping
        if hasattr(self, 'anomaly_type_mapping'):
            logger.info("Anomaly type mapping:")
            for anomaly_type, type_number in (
                    self.anomaly_type_mapping.items()):
                count = np.sum(self.anomaly_labels[anomaly_type])
                logger.info(f"  - {type_number}: {anomaly_type} "
                            f"({count} anomalies)")
        
        return df_result
    
    def _create_replacement_dataset(self, df_original: pd.DataFrame,
                                   all_anomalous_data: List[Dict],
                                   anomaly_types: List[str]) -> pd.DataFrame:
        """Create dataset by replacing original rows with anomalous ones."""
        
        # Start with original data
        df_result = df_original.copy()
        
        # Create anomaly type mapping
        anomaly_type_mapping = {anomaly_type: i + 1
                               for i, anomaly_type in enumerate(anomaly_types)}
        self.anomaly_type_mapping = anomaly_type_mapping
        
        # Initialize anomaly indicator
        df_result['is_synthetic_anomaly'] = 0
        
        # Initialize anomaly labels
        self.anomaly_labels = {}
        for anomaly_type in anomaly_types:
            self.anomaly_labels[anomaly_type] = np.zeros(len(df_result),
                                                        dtype=bool)
        
        # Replace rows with anomalous versions using vectorized operations
        if all_anomalous_data:
            # Extract data for batch processing
            indices_to_replace = []
            anomaly_rows_data = []
            anomaly_type_per_index = {}
            
            for anomaly_data in all_anomalous_data:
                row_data = anomaly_data['row_data']
                template_idx = anomaly_data['template_index']
                anomaly_type = anomaly_data['anomaly_type']
                
                indices_to_replace.append(template_idx)
                anomaly_rows_data.append(row_data)
                anomaly_type_per_index[template_idx] = anomaly_type
            
            # Create DataFrame from anomalous data
            if anomaly_rows_data:
                anomalous_df = pd.DataFrame(anomaly_rows_data, 
                                            index=indices_to_replace)
                
                # Update original dataframe with anomalous data
                # Only update columns that exist in both dataframes
                common_cols = [col for col in anomalous_df.columns 
                               if col in df_result.columns]
                df_result.loc[indices_to_replace, common_cols] = (
                    anomalous_df[common_cols])
                
                # Set anomaly indicators using vectorized operations
                for template_idx, anomaly_type in (
                        anomaly_type_per_index.items()):
                    if anomaly_type in anomaly_type_mapping:
                        type_number = anomaly_type_mapping[anomaly_type]
                        df_result.loc[template_idx, 'is_synthetic_anomaly'] = (
                            type_number)
                        self.anomaly_labels[anomaly_type][template_idx] = True
        
        return df_result
    
    def _create_additive_dataset(self, df_original: pd.DataFrame,
                                all_anomalous_data: List[Dict],
                                anomaly_types: List[str]) -> pd.DataFrame:
        """Create dataset by adding new anomalous rows."""
        
        # Extract just the row data for adding
        new_rows = [item['row_data'] for item in all_anomalous_data]
        
        # Combine original data with synthetic anomalies
        if new_rows:
            anomalous_df = pd.DataFrame(new_rows)
            df_combined = pd.concat([df_original, anomalous_df],
                                  ignore_index=True)
        else:
            df_combined = df_original.copy()
        
        # Create anomaly labels for the combined dataset
        n_original = len(df_original)
        n_total = len(df_combined)
        
        # Create anomaly type mapping
        anomaly_type_mapping = {anomaly_type: i + 1
                               for i, anomaly_type in enumerate(anomaly_types)}
        self.anomaly_type_mapping = anomaly_type_mapping
        
        # Initialize labels - original rows are False, synthetic rows will be
        # True
        self.anomaly_labels = {}
        for anomaly_type in anomaly_types:
            self.anomaly_labels[anomaly_type] = np.zeros(n_total, dtype=bool)
        
        # Add general anomaly indicator column
        df_combined['is_synthetic_anomaly'] = np.zeros(len(df_combined),
                                                      dtype=int)
        
        # Mark synthetic anomalies
        current_idx = n_original
        for anomaly_data in all_anomalous_data:
            row_data = anomaly_data['row_data']
            anomaly_type = anomaly_data['anomaly_type']
            
            if 'anomaly_type' in row_data:
                if anomaly_type in self.anomaly_labels:
                    self.anomaly_labels[anomaly_type][current_idx] = True
                    # Set the anomaly type number in the indicator column
                    if anomaly_type in anomaly_type_mapping:
                        type_number = anomaly_type_mapping[anomaly_type]
                        df_combined.loc[current_idx, 
                                       'is_synthetic_anomaly'] = type_number
            current_idx += 1
        
        return df_combined
    
    def _generate_single_bid_anomalies(self, df: pd.DataFrame, 
                                     n_anomalies: int, 
                                     anomaly_type: str) -> Tuple[List[Dict], 
                                                                List[int]]:
        """Generate new rows with single bid competitive anomalies.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find competitive procedures with more than 1 bid as templates
        competitive_procedures = ["Appel d'offres ouvert", 
                                "Appel d'offres restreint"]
        
        mask = (df['procedure'].isin(competitive_procedures) & 
                (df['offresRecues'] > 1) & 
                df['offresRecues'].notna())
        
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for "
                          "single_bid_competitive anomalies")
            return [], []
        
        # Randomly select template rows
        selected_rows = eligible_rows.sample(
            n=min(n_anomalies, len(eligible_rows)), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for idx, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            
            # Make it anomalous: set to exactly 1 bid
            new_row['offresRecues'] = 1
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} single bid competitive "
                   "anomaly rows")
        return new_rows, template_indices
    
    def _generate_price_inflation_anomalies(self, df: pd.DataFrame, 
                                          n_anomalies: int, 
                                          anomaly_type: str) -> Tuple[
                                              List[Dict], List[int]]:
        """Generate new rows with artificially inflated prices.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find contracts with valid amounts as templates
        mask = df['montant'].notna() & (df['montant'] > 0)
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for price inflation "
                          "anomalies")
            return [], []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(
            n=min(n_anomalies, len(eligible_rows)), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for idx, row in selected_rows.iterrows():
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
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} price inflation anomaly rows")
        return new_rows, template_indices
    
    def _generate_price_deflation_anomalies(self, df: pd.DataFrame, 
                                          n_anomalies: int, 
                                          anomaly_type: str) -> Tuple[
                                              List[Dict], List[int]]:
        """Generate new rows with artificially deflated prices.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find contracts with valid amounts as templates
        mask = df['montant'].notna() & (df['montant'] > 0)
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for price deflation "
                          "anomalies")
            return [], []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(
            n=min(n_anomalies, len(eligible_rows)), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for idx, row in selected_rows.iterrows():
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
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} price deflation anomaly rows")
        return new_rows, template_indices
    
    def _generate_procedure_manipulation_anomalies(self, df: pd.DataFrame, 
                                                  n_anomalies: int, 
                                                  anomaly_type: str) -> Tuple[
                                                      List[Dict], List[int]]:
        """Generate new rows with suspicious procedure manipulation.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find competitive procedures as templates
        competitive_procedures = ["Appel d'offres ouvert", 
                                "Appel d'offres restreint"]
        mask = df['procedure'].isin(competitive_procedures)
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for procedure "
                          "manipulation anomalies")
            return [], []
        
        # Non-competitive procedures to switch to
        non_competitive = ['Procédure adaptée', 
                          'Marché négocié sans publicité']
        
        # Select random template rows
        selected_rows = eligible_rows.sample(
            n=min(n_anomalies, len(eligible_rows)), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for idx, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            
            # Switch to non-competitive procedure
            new_row['procedure'] = random.choice(non_competitive)
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} procedure manipulation "
                   "anomaly rows")
        return new_rows, template_indices
    
    def _generate_suspicious_modification_anomalies(self, df: pd.DataFrame, 
                                                   n_anomalies: int, 
                                                   anomaly_type: str) -> Tuple[
                                                       List[Dict], List[int]]:
        """Generate new rows suggesting suspicious contract modifications.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find contracts with reasonable duration as templates
        mask = (df['dureeMois'].notna() & (df['dureeMois'] > 0) & 
                (df['dureeMois'] < 36))
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for suspicious "
                          "modification anomalies")
            return [], []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(
            n=min(n_anomalies, len(eligible_rows)), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for idx, row in selected_rows.iterrows():
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
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} suspicious modification "
                   "anomaly rows")
        return new_rows, template_indices
    
    def _generate_high_market_concentration_anomalies(self, df: pd.DataFrame, 
                                                      n_anomalies: int, 
                                                      anomaly_type: str) -> Tuple[
                                                          List[Dict], List[int]]:
        """Generate new rows with high market concentration anomalies.
        
        Creates anomalies where a single supplier dominates a buyer's contracts,
        indicating potential market concentration issues.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find buyer-CPV combinations with multiple suppliers as templates
        buyer_cpv_groups = df.groupby(['acheteur_id', 'codeCPV_3'])
        
        eligible_groups = []
        for (buyer_id, cpv), group in buyer_cpv_groups:
            if len(group) >= 3 and group['titulaire_id'].nunique() > 1:
                eligible_groups.append((buyer_id, cpv, group))
        
        if len(eligible_groups) == 0:
            logger.warning("No eligible buyer-CPV groups found for market "
                          "concentration anomalies")
            return [], []
        
        # Select random groups to create anomalies from
        selected_groups = random.sample(eligible_groups, 
                                       min(n_anomalies // 3, 
                                           len(eligible_groups)))
        
        new_rows = []
        template_indices = []
        
        for buyer_id, cpv, group in selected_groups:
            # Pick one supplier to dominate
            suppliers = group['titulaire_id'].unique()
            dominant_supplier = random.choice(suppliers)
            
            # Create new contracts for this dominant supplier
            template_rows = group.sample(n=min(3, len(group)), 
                                       random_state=self.random_seed)
            
            for idx, row in template_rows.iterrows():
                new_row = row.copy()
                new_row['titulaire_id'] = dominant_supplier
                
                # Add anomaly metadata
                new_row['anomaly_type'] = anomaly_type
                new_row['source_type'] = 'synthetic'
                
                new_rows.append(new_row.to_dict())
                template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} high market concentration "
                   "anomaly rows")
        return new_rows, template_indices
    
    def _generate_temporal_clustering_anomalies(self, df: pd.DataFrame, 
                                                n_anomalies: int, 
                                                anomaly_type: str) -> Tuple[
                                                    List[Dict], List[int]]:
        """Generate new rows with suspicious temporal clustering patterns.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find buyer-supplier pairs with multiple contracts as templates
        buyer_supplier_pairs = df.groupby(['acheteur_id', 
                                          'titulaire_id']).size()
        eligible_pairs = buyer_supplier_pairs[
            buyer_supplier_pairs >= 3].index.tolist()
        
        if len(eligible_pairs) == 0:
            logger.warning("No eligible buyer-supplier pairs found for "
                          "temporal clustering anomalies")
            return [], []
        
        # Select pairs to create clustered contracts from
        selected_pairs = random.sample(
            eligible_pairs,
            min(len(eligible_pairs), n_anomalies // 3))
        
        new_rows = []
        template_indices = []
        
        for buyer_id, supplier_id in selected_pairs:
            pair_contracts = df[(df['acheteur_id'] == buyer_id) & 
                              (df['titulaire_id'] == supplier_id)]
            
            # Pick template contracts and create clustered ones
            template_contracts = pair_contracts.sample(
                n=min(3, len(pair_contracts)), 
                random_state=self.random_seed)
            
            # Pick a random date and cluster contracts around it
            base_date = datetime(2023, random.randint(1, 12), 
                               random.randint(1, 28))
            
            for i, (idx, row) in enumerate(template_contracts.iterrows()):
                new_row = row.copy()
                
                # Create clustered date (within 2 weeks of each other)
                clustered_date = base_date + timedelta(
                    days=random.randint(0, 14))
                new_row['dateNotification'] = clustered_date.strftime(
                    '%Y-%m-%d')
                
                # Add anomaly metadata
                new_row['anomaly_type'] = anomaly_type
                new_row['source_type'] = 'synthetic'
                
                new_rows.append(new_row.to_dict())
                template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} temporal clustering "
                   "anomaly rows")
        return new_rows, template_indices
    
    def _generate_excessive_subcontracting_anomalies(self, df: pd.DataFrame, 
                                                      n_anomalies: int, 
                                                      anomaly_type: str) -> Tuple[
                                                          List[Dict], List[int]]:
        """Generate new rows with excessive subcontracting patterns.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find contracts that currently don't declare subcontracting as templates
        mask = (df['sousTraitanceDeclaree'].notna() & 
                (df['sousTraitanceDeclaree'] == 0))
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for excessive "
                          "subcontracting anomalies")
            return [], []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(
            n=min(n_anomalies, len(eligible_rows)), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for idx, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            
            # Make it anomalous: mark as having declared subcontracting
            new_row['sousTraitanceDeclaree'] = 1
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} excessive subcontracting "
                   "anomaly rows")
        return new_rows, template_indices
    
    def _generate_short_duration_anomalies(self, df: pd.DataFrame, 
                                           n_anomalies: int, 
                                           anomaly_type: str) -> Tuple[
                                               List[Dict], List[int]]:
        """Generate new rows with unusually short contract durations.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find contracts with valid duration (> 1 month) as templates
        mask = df['dureeMois'].notna() & (df['dureeMois'] > 1)
        eligible_rows = df[mask]
        
        if len(eligible_rows) == 0:
            logger.warning("No eligible contracts found for unusual duration "
                          "anomalies")
            return [], []
        
        # Select random template rows
        selected_rows = eligible_rows.sample(
            n=min(n_anomalies, len(eligible_rows)), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for idx, row in selected_rows.iterrows():
            # Create new anomalous row based on template
            new_row = row.copy()
            
            # Make it anomalous: set to very short duration (< 1 month)
            new_row['dureeMois'] = random.uniform(0.1, 0.9)
            
            # Add anomaly metadata
            new_row['anomaly_type'] = anomaly_type
            new_row['source_type'] = 'synthetic'
            
            new_rows.append(new_row.to_dict())
            template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} unusual short duration "
                   "anomaly rows")
        return new_rows, template_indices
    
    def _generate_suspicious_pairs_anomalies(self, df: pd.DataFrame, 
                                             n_anomalies: int, 
                                             anomaly_type: str) -> Tuple[
                                                 List[Dict], List[int]]:
        """Generate new rows with suspicious buyer-supplier relationship patterns.
        
        Returns:
            Tuple of (new_rows, template_indices)
        """
        
        # Find buyer-supplier pairs with multiple contracts and valid amounts
        buyer_supplier_amounts = df.groupby(['acheteur_id', 'titulaire_id']).agg({
            'montant': ['sum', 'count']
        }).reset_index()
        buyer_supplier_amounts.columns = ['acheteur_id', 'titulaire_id', 
                                         'sum_montant', 'count_contracts']
        
        # Filter for pairs with at least 2 contracts and valid amounts
        eligible_pairs = buyer_supplier_amounts[
            (buyer_supplier_amounts['count_contracts'] >= 2) & 
            (buyer_supplier_amounts['sum_montant'].notna()) &
            (buyer_supplier_amounts['sum_montant'] > 0)
        ]
        
        if len(eligible_pairs) == 0:
            logger.warning("No eligible buyer-supplier pairs found for "
                          "suspicious pairs anomalies")
            return [], []
        
        # Select pairs to create suspicious relationships from
        selected_pairs = eligible_pairs.sample(
            min(len(eligible_pairs), n_anomalies // 2), 
            random_state=self.random_seed)
        
        new_rows = []
        template_indices = []
        
        for _, row in selected_pairs.iterrows():
            buyer_id = row['acheteur_id']
            supplier_id = row['titulaire_id']
            
            # Find contracts for this pair as templates
            pair_contracts = df[(df['acheteur_id'] == buyer_id) & 
                              (df['titulaire_id'] == supplier_id)]
            
            # Create new contracts with inflated amounts for this pair
            template_contracts = pair_contracts.sample(
                n=min(2, len(pair_contracts)), 
                random_state=self.random_seed)
            
            for idx, template_row in template_contracts.iterrows():
                new_row = template_row.copy()
                
                # Make it anomalous: inflate the amount significantly
                original_amount = new_row['montant']
                if pd.notna(original_amount) and original_amount > 0:
                    new_row['montant'] = original_amount * random.uniform(1.5, 3.0)
                
                # Add anomaly metadata
                new_row['anomaly_type'] = anomaly_type
                new_row['source_type'] = 'synthetic'
                
                new_rows.append(new_row.to_dict())
                template_indices.append(idx)
        
        logger.info(f"Generated {len(new_rows)} suspicious buyer-supplier "
                   "pair anomaly rows")
        return new_rows, template_indices
    
    def get_anomaly_summary(self) -> pd.DataFrame:
        """Get a summary of generated anomalies."""
        
        summary_data = []
        for anomaly_type, labels in self.anomaly_labels.items():
            summary_data.append({
                'anomaly_type': anomaly_type,
                'count': np.sum(labels),
                'percentage': (np.sum(labels) / len(labels) * 100 
                             if len(labels) > 0 else 0)
            })
        
        return pd.DataFrame(summary_data).sort_values('count', 
                                                     ascending=False)
    
    def get_replaced_indices(self) -> set:
        """Get the indices of rows that were replaced (if replace mode was 
        used)."""
        return self.replaced_indices.copy()
    
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


def demonstrate_anomaly_generation(df: pd.DataFrame, 
                                   sample_size: int = 1000) -> None:
    """Demonstrate the anomaly generation functionality with replacement 
    option."""
    
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
    
    # Test both modes
    print("\n--- Mode 1: Adding new rows ---")
    df_with_added = generator.generate_anomalies(
        df_sample,
        anomaly_percentage=0.05,
        replace_rows=False,
        anomaly_types=['single_bid_competitive', 'price_inflation']
    )
    
    print(f"Original: {len(df_sample)} rows")
    print(f"With additions: {len(df_with_added)} rows")
    print(f"Added: {len(df_with_added) - len(df_sample)} rows")
    
    print("\n--- Mode 2: Replacing existing rows ---")
    generator2 = SyntheticAnomalyGenerator(random_seed=42)
    df_with_replaced = generator2.generate_anomalies(
        df_sample,
        anomaly_percentage=0.05,
        replace_rows=True,
        anomaly_types=['price_deflation', 'procedure_manipulation']
    )
    
    print(f"Original: {len(df_sample)} rows")
    print(f"With replacements: {len(df_with_replaced)} rows")
    print(f"Replaced indices: {len(generator2.get_replaced_indices())}")
    
    # Show summaries
    print("\nSummary for addition mode:")
    summary1 = generator.get_anomaly_summary()
    print(summary1.to_string(index=False))
    
    print("\nSummary for replacement mode:")
    summary2 = generator2.get_anomaly_summary()
    print(summary2.to_string(index=False))
    
    # Show example of replaced indices
    replaced_indices = generator2.get_replaced_indices()
    print(f"\nFirst 5 replaced indices: {list(replaced_indices)[:5]}")


if __name__ == "__main__":
    # Example usage
    print("Synthetic Anomaly Generator - Example Usage")
    print("This module creates synthetic anomalies by either adding new rows")
    print("or replacing existing rows in the dataset.")
    print("Each anomaly generation method returns both the new rows and the")
    print("indices of the template rows that were used.")
    print("See the demonstrate_anomaly_generation() function for usage "
          "examples.") 