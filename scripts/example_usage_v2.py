"""
Example Usage: Synthetic Anomaly Generation V2 (Adding New Rows)

This script demonstrates how to use the updated synthetic anomaly generator
that creates new rows instead of modifying existing data.

Author: RonanB400
Date: January 2025
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from synthetic_anomaly_generator_v2 import SyntheticAnomalyGenerator


def example_with_sample_data():
    """Create a simple example with synthetic procurement data."""
    
    print("="*70)
    print("SYNTHETIC ANOMALY GENERATOR V2 - ADDING NEW ROWS")
    print("="*70)
    
    # Create sample procurement data
    sample_data = {
        'acheteur_id': [f'BUYER_{i:03d}' for i in range(1, 101)],
        'titulaire_id': [f'SUPPLIER_{i:03d}' for i in range(1, 101)],
        'procedure': ['Appel d\'offres ouvert'] * 50 + ['Appel d\'offres restreint'] * 30 + ['Procédure adaptée'] * 20,
        'montant': np.random.lognormal(10, 1, 100),  # Log-normal distribution for realistic amounts
        'offresRecues': np.random.randint(1, 8, 100),
        'dureeMois': np.random.randint(1, 48, 100),
        'codeCPV': [f'CPV_{i:02d}' for i in np.random.randint(1, 21, 100)],
        'dateNotification': pd.date_range('2023-01-01', periods=100, freq='D'),
        'sousTraitanceDeclaree': np.random.choice([0, 1], 100, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(sample_data)
    
    print(f"Original dataset: {len(df)} rows")
    print("\nOriginal dataset overview:")
    print(df.describe())
    
    # Initialize the anomaly generator
    generator = SyntheticAnomalyGenerator(random_seed=42)
    
    # Generate synthetic anomalies by adding new rows
    print("\n" + "-"*50)
    print("GENERATING SYNTHETIC ANOMALIES")
    print("-"*50)
    
    df_with_anomalies, anomaly_labels = generator.generate_anomalies(
        df,
        anomaly_types=['single_bid_competitive', 'price_manipulation', 
                      'procedure_manipulation', 'suspicious_modifications'],
        anomaly_percentage=0.15  # Add 15% synthetic anomalies
    )
    
    # Show results
    print(f"\nFinal dataset: {len(df_with_anomalies)} rows")
    print(f"Original rows: {len(df)}")
    print(f"Added synthetic anomaly rows: {len(df_with_anomalies) - len(df)}")
    
    # Show anomaly summary
    print("\n" + "-"*50)
    print("ANOMALY SUMMARY")
    print("-"*50)
    
    summary = generator.get_anomaly_summary()
    print(summary.to_string(index=False))
    
    # Show examples of synthetic anomalies
    print("\n" + "-"*50)
    print("EXAMPLES OF SYNTHETIC ANOMALIES")
    print("-"*50)
    
    synthetic_rows = df_with_anomalies[df_with_anomalies['is_synthetic_anomaly'] == True]
    
    if len(synthetic_rows) > 0:
        print(f"\nFirst 5 synthetic anomaly rows:")
        relevant_cols = ['acheteur_id', 'titulaire_id', 'procedure', 'montant', 
                        'offresRecues', 'anomaly_type', 'source_type']
        available_cols = [col for col in relevant_cols if col in synthetic_rows.columns]
        print(synthetic_rows[available_cols].head().to_string(index=False))
        
        # Show specific anomaly types
        for anomaly_type in ['single_bid_competitive', 'price_manipulation']:
            type_rows = synthetic_rows[synthetic_rows['anomaly_type'] == anomaly_type]
            if len(type_rows) > 0:
                print(f"\n{anomaly_type.upper()} examples:")
                print(type_rows[available_cols].head(2).to_string(index=False))
    
    # Compare original vs synthetic data statistics
    print("\n" + "-"*50)
    print("DATA COMPARISON: ORIGINAL vs WITH ANOMALIES")
    print("-"*50)
    
    original_stats = df[['montant', 'offresRecues', 'dureeMois']].describe()
    combined_stats = df_with_anomalies[['montant', 'offresRecues', 'dureeMois']].describe()
    
    print("\nOriginal data statistics:")
    print(original_stats)
    
    print("\nCombined data statistics (with synthetic anomalies):")
    print(combined_stats)
    
    # Show the difference that anomalies make
    print("\n" + "-"*50)
    print("IMPACT OF SYNTHETIC ANOMALIES")
    print("-"*50)
    
    print(f"Original mean contract amount: €{df['montant'].mean():.2f}")
    print(f"Combined mean contract amount: €{df_with_anomalies['montant'].mean():.2f}")
    
    print(f"Original mean offers received: {df['offresRecues'].mean():.2f}")
    print(f"Combined mean offers received: {df_with_anomalies['offresRecues'].mean():.2f}")
    
    # Show procedure distribution
    print(f"\nOriginal procedure distribution:")
    print(df['procedure'].value_counts())
    
    print(f"\nCombined procedure distribution:")
    print(df_with_anomalies['procedure'].value_counts())
    
    return df_with_anomalies, anomaly_labels


def example_with_real_data(df_path: str):
    """Example with real procurement data file."""
    
    if not os.path.exists(df_path):
        print(f"File {df_path} not found. Using sample data instead.")
        return example_with_sample_data()
    
    print("="*70)
    print("REAL DATA EXAMPLE")
    print("="*70)
    
    # Load real data
    df = pd.read_csv(df_path)
    print(f"Loaded real dataset: {len(df)} rows")
    
    # Take a sample for demonstration
    if len(df) > 1000:
        df_sample = df.sample(n=1000, random_state=42)
        print(f"Using sample of 1000 rows for demonstration")
    else:
        df_sample = df.copy()
    
    # Generate anomalies
    generator = SyntheticAnomalyGenerator(random_seed=42)
    
    df_with_anomalies, anomaly_labels = generator.generate_anomalies(
        df_sample,
        anomaly_percentage=0.05  # Add 5% synthetic anomalies
    )
    
    print(f"\nOriginal sample: {len(df_sample)} rows")
    print(f"With anomalies: {len(df_with_anomalies)} rows")
    print(f"Added: {len(df_with_anomalies) - len(df_sample)} synthetic anomaly rows")
    
    # Show summary
    summary = generator.get_anomaly_summary()
    print("\nAnomaly Summary:")
    print(summary.to_string(index=False))
    
    return df_with_anomalies, anomaly_labels


if __name__ == "__main__":
    
    # Run the example with sample data
    df_result, labels = example_with_sample_data()
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nKey Benefits of Adding New Rows Approach:")
    print("- ✅ Original data is completely preserved")
    print("- ✅ Easy to identify synthetic vs real anomalies")
    print("- ✅ Can control exact number and type of anomalies")
    print("- ✅ Perfect for testing detection models")
    print("- ✅ Clear separation for evaluation metrics")
    
    print(f"\nFinal dataset shape: {df_result.shape}")
    print(f"Synthetic anomalies: {df_result['is_synthetic_anomaly'].sum()}")
    print(f"Original data: {(~df_result['is_synthetic_anomaly']).sum()}") 