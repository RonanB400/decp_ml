"""
Example Usage: Synthetic Anomaly Generation and Testing

This script demonstrates how to:
1. Load procurement data
2. Generate synthetic anomalies
3. Test GNN models against these known anomalies
4. Evaluate and visualize results

Author: RonanB400
Date: January 2025
"""

import pandas as pd
import numpy as np
import os
import sys

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from synthetic_anomaly_generator import SyntheticAnomalyGenerator, demonstrate_anomaly_generation
from test_gnn_with_synthetic_anomalies import GNNSyntheticAnomalyTester


def simple_anomaly_generation_example():
    """Simple example of generating synthetic anomalies."""
    
    print("="*60)
    print("SIMPLE SYNTHETIC ANOMALY GENERATION EXAMPLE")
    print("="*60)
    
    # Load sample data (replace with your actual data path)
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data')
    data_file = os.path.join(data_path, 'data_clean.csv')
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        print("Please ensure you have the cleaned data file in the correct location.")
        return
    
    # Load data
    df = pd.read_csv(data_file, encoding='utf-8')
    print(f"Loaded {len(df)} procurement contracts")
    
    # Take a sample for demonstration
    if len(df) > 2000:
        df_sample = df.sample(n=2000, random_state=42)
        print(f"Using sample of 2000 contracts for demonstration")
    else:
        df_sample = df.copy()
    
    # Initialize anomaly generator
    generator = SyntheticAnomalyGenerator(random_seed=42)
    
    # Generate specific types of anomalies
    anomaly_types_to_test = [
        'single_bid_competitive',    # Competitive procedures with only 1 bid
        'price_manipulation',        # Artificially inflated/deflated prices
        'high_market_concentration', # One supplier dominating buyer's contracts
        'procedure_manipulation'     # Switching to non-competitive procedures
    ]
    
    # Generate anomalies
    df_with_anomalies, anomaly_labels = generator.generate_anomalies(
        df_sample,
        anomaly_types=anomaly_types_to_test,
        anomaly_percentage=0.10  # 10% of data will be anomalous
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
    for anomaly_type in anomaly_types_to_test:
        if anomaly_type in anomaly_labels:
            anomalous_rows = df_with_anomalies[anomaly_labels[anomaly_type]]
            if len(anomalous_rows) > 0:
                print(f"\n{anomaly_type.upper()} example:")
                relevant_cols = ['acheteur_id', 'titulaire_id', 'procedure', 'montant', 'offresRecues']
                available_cols = [col for col in relevant_cols if col in anomalous_rows.columns]
                print(anomalous_rows[available_cols].head(1).to_string(index=False))
    
    return df_with_anomalies, anomaly_labels


def full_gnn_testing_example():
    """Full example of testing GNN models with synthetic anomalies."""
    
    print("\n" + "="*60)
    print("FULL GNN TESTING WITH SYNTHETIC ANOMALIES")
    print("="*60)
    
    # Configuration
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data')
    
    # Check if data exists
    data_file = os.path.join(data_path, 'data_clean.csv')
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        print("Please ensure you have the cleaned data file in the correct location.")
        return
    
    # Initialize tester
    tester = GNNSyntheticAnomalyTester(data_path)
    
    # Define which anomaly types to test
    test_anomaly_types = [
        'single_bid_competitive',    # Should be detectable by edge model
        'price_manipulation',        # Should be detectable by both models
        'high_market_concentration', # Should be detectable by node model
        'procedure_manipulation'     # Should be detectable by edge model
    ]
    
    print(f"Testing anomaly types: {test_anomaly_types}")
    
    # Run the test
    try:
        results = tester.run_synthetic_anomaly_test(
            anomaly_types=test_anomaly_types,
            anomaly_percentage=0.08,  # 8% anomalies
            test_sample_size=2000,    # Use 2000 contracts for faster testing
            epochs=20                 # Train for 20 epochs (faster for demo)
        )
        
        # Print detailed results
        tester.print_detailed_results()
        
        # Create visualizations
        save_dir = os.path.join(data_path, 'synthetic_test_results')
        tester.plot_comprehensive_results(save_dir=save_dir)
        
        print(f"\nResults and plots saved to: {save_dir}")
        
        return results
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("This might be due to missing dependencies or data issues.")
        return None


def quick_anomaly_demonstration():
    """Quick demonstration without full GNN training."""
    
    print("\n" + "="*60)
    print("QUICK ANOMALY DEMONSTRATION")
    print("="*60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data')
    data_file = os.path.join(data_path, 'data_clean.csv')
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        print("Creating sample data for demonstration...")
        
        # Create sample data for demonstration
        np.random.seed(42)
        sample_data = {
            'acheteur_id': [f'BUYER_{i:03d}' for i in range(100)] * 10,
            'titulaire_id': [f'SUPPLIER_{i:03d}' for i in range(50)] * 20,
            'procedure': np.random.choice(['Appel d\'offres ouvert', 'Procédure adaptée'], 1000),
            'montant': np.random.exponential(50000, 1000),
            'offresRecues': np.random.randint(1, 10, 1000),
            'dureeMois': np.random.randint(1, 60, 1000),
            'codeCPV': [f'{i:08d}-0' for i in np.random.randint(10000000, 99999999, 1000)],
            'sousTraitanceDeclaree': np.random.choice([0, 1], 1000)
        }
        df = pd.DataFrame(sample_data)
    else:
        df = pd.read_csv(data_file, encoding='utf-8')
    
    # Use the demonstrate function
    demonstrate_anomaly_generation(df, sample_size=500)


def main():
    """Main function to run examples."""
    
    print("Synthetic Anomaly Generation and Testing Examples")
    print("Please choose an option:")
    print("1. Quick demonstration with sample data")
    print("2. Simple anomaly generation example")
    print("3. Full GNN testing with synthetic anomalies")
    print("4. Run all examples")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        quick_anomaly_demonstration()
    elif choice == '2':
        simple_anomaly_generation_example()
    elif choice == '3':
        full_gnn_testing_example()
    elif choice == '4':
        quick_anomaly_demonstration()
        simple_anomaly_generation_example()
        full_gnn_testing_example()
    else:
        print("Invalid choice. Running quick demonstration...")
        quick_anomaly_demonstration()


if __name__ == "__main__":
    main()


# JUPYTER NOTEBOOK USAGE EXAMPLE:
"""
# In a Jupyter notebook, you can use it like this:

# 1. Import the modules
from scripts.synthetic_anomaly_generator import SyntheticAnomalyGenerator
from scripts.test_gnn_with_synthetic_anomalies import GNNSyntheticAnomalyTester
import pandas as pd

# 2. Load your data
df = pd.read_csv('data/data_clean.csv', encoding='utf-8')

# 3. Generate synthetic anomalies
generator = SyntheticAnomalyGenerator(random_seed=42)
df_with_anomalies, anomaly_labels = generator.generate_anomalies(
    df,
    anomaly_types=['single_bid_competitive', 'price_manipulation'],
    anomaly_percentage=0.05  # 5% anomalies
)

# 4. Test with GNN models
tester = GNNSyntheticAnomalyTester('data/')
results = tester.run_synthetic_anomaly_test(
    anomaly_types=['single_bid_competitive', 'price_manipulation'],
    anomaly_percentage=0.05,
    test_sample_size=1000,
    epochs=15
)

# 5. View results
tester.print_detailed_results()
tester.plot_comprehensive_results()
""" 