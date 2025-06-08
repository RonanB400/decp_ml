"""
Quick Fix: Use the V2 Version that Already Works

The original synthetic_anomaly_generator.py has mixed method signatures.
Use synthetic_anomaly_generator_v2.py instead, which has all methods properly updated.

Author: RonanB400
Date: January 2025
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the V2 version which has all methods properly updated
from scripts.synthetic_anomaly_generator import SyntheticAnomalyGenerator

# Your existing code should work with this change:
def test_anomaly_generation(df):
    """Test the anomaly generation with your data."""
    
    # Initialize generator
    generator = SyntheticAnomalyGenerator(random_seed=42)
    
    # Use a sample of your data
    df_sample = df.copy()
    
    # Generate anomalies - this will now work correctly
    df_with_anomalies, anomaly_labels = generator.generate_anomalies(
        df_sample,
        anomaly_percentage=0.10,  # 10% anomalies
        anomaly_types=['single_bid_competitive', 'price_manipulation', 
                      'procedure_manipulation', 'suspicious_modifications']
    )
    
    print(f"Original dataset: {len(df_sample)} rows")
    print(f"With anomalies: {len(df_with_anomalies)} rows")
    print(f"Added synthetic anomalies: {len(df_with_anomalies) - len(df_sample)}")
    
    # Show summary
    summary = generator.get_anomaly_summary()
    print("\nAnomaly Summary:")
    print(summary)
    
    return df_with_anomalies, anomaly_labels

if __name__ == "__main__":
    print("Use synthetic_anomaly_generator_v2.py instead!")
    print("It has all methods properly updated to the new format.") 