"""
Improved Anomaly Detection for DECP Data
This script provides several approaches to improve anomaly detection when the initial pipeline doesn't find any anomalies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

def test_contamination_values(X_train, X_test, preproc_baseline):
    """Test different contamination values to find optimal setting"""
    print("=== Testing Different Contamination Values ===")
    contamination_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    
    results = []
    for contamination in contamination_values:
        model_if = IsolationForest(
            n_estimators=100, 
            contamination=contamination, 
            random_state=0,
            max_samples='auto'
        )
        pipeline = make_pipeline(preproc_baseline, model_if)
        
        pipeline.fit(X_train)
        anomalies = pipeline.predict(X_test)
        
        n_anomalies = (anomalies == -1).sum()
        percentage = (n_anomalies / len(X_test)) * 100
        
        results.append({
            'contamination': contamination,
            'n_anomalies': n_anomalies,
            'percentage': percentage
        })
        
        print(f"Contamination: {contamination:.3f} | Anomalies: {n_anomalies:5d} ({percentage:.2f}%)")
    
    return pd.DataFrame(results)

def analyze_anomaly_scores(X_train, X_test, preproc_baseline, contamination=0.05):
    """Analyze anomaly scores distribution"""
    print(f"\n=== Analyzing Anomaly Scores (contamination={contamination}) ===")
    
    model_if = IsolationForest(
        n_estimators=100, 
        contamination=contamination, 
        random_state=0
    )
    pipeline = make_pipeline(preproc_baseline, model_if)
    
    pipeline.fit(X_train)
    
    # Get anomaly scores
    anomaly_scores = pipeline.decision_function(X_test)
    predictions = pipeline.predict(X_test)
    
    print(f"Anomaly Score Statistics:")
    print(f"Mean: {np.mean(anomaly_scores):.4f}")
    print(f"Std: {np.std(anomaly_scores):.4f}")
    print(f"Min: {np.min(anomaly_scores):.4f}")
    print(f"Max: {np.max(anomaly_scores):.4f}")
    
    # Show distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(anomaly_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Anomaly Scores')
    plt.axvline(x=0, color='red', linestyle='--', label='Decision Boundary')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.boxplot(anomaly_scores)
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Score Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Return most anomalous samples
    X_test_copy = X_test.copy()
    X_test_copy['anomaly_score'] = anomaly_scores
    X_test_copy['is_anomaly'] = predictions == -1
    
    return X_test_copy.sort_values('anomaly_score').head(20)

def try_alternative_algorithms(X_train, X_test, preproc_baseline):
    """Try alternative anomaly detection algorithms"""
    print("\n=== Testing Alternative Algorithms ===")
    
    # Fit preprocessing
    X_train_processed = preproc_baseline.fit_transform(X_train)
    X_test_processed = preproc_baseline.transform(X_test)
    
    algorithms = {
        'IsolationForest_0.05': IsolationForest(contamination=0.05, random_state=0),
        'IsolationForest_0.1': IsolationForest(contamination=0.1, random_state=0),
        'LocalOutlierFactor': LocalOutlierFactor(contamination=0.05, novelty=True),
        'OneClassSVM_rbf': OneClassSVM(kernel='rbf', gamma='scale', nu=0.05),
        'OneClassSVM_linear': OneClassSVM(kernel='linear', nu=0.05)
    }
    
    results = {}
    for name, model in algorithms.items():
        try:
            model.fit(X_train_processed)
            predictions = model.predict(X_test_processed)
            
            n_anomalies = (predictions == -1).sum()
            percentage = (n_anomalies / len(X_test)) * 100
            
            results[name] = {
                'n_anomalies': n_anomalies,
                'percentage': percentage
            }
            
            print(f"{name:20s} | Anomalies: {n_anomalies:5d} ({percentage:.2f}%)")
        except Exception as e:
            print(f"{name:20s} | Error: {str(e)}")
            results[name] = {'n_anomalies': 0, 'percentage': 0.0}
    
    return results

def feature_based_analysis(X_train, X_test):
    """Analyze potential anomalies based on individual features"""
    print("\n=== Feature-Based Anomaly Analysis ===")
    
    numerical_features = ['montant', 'dureeMois', 'tauxAvance']
    
    for feature in numerical_features:
        if feature in X_test.columns:
            # Calculate percentiles
            q1 = X_test[feature].quantile(0.25)
            q3 = X_test[feature].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Find outliers
            outliers = X_test[(X_test[feature] < lower_bound) | (X_test[feature] > upper_bound)]
            
            print(f"{feature}:")
            print(f"  Outliers: {len(outliers)} ({len(outliers)/len(X_test)*100:.2f}%)")
            print(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            if len(outliers) > 0:
                print(f"  Extreme values: {outliers[feature].nsmallest(3).tolist()} ... {outliers[feature].nlargest(3).tolist()}")
            print()

def improved_preprocessing_approach(X_train, X_test):
    """Try different preprocessing approaches"""
    print("\n=== Testing Different Preprocessing Approaches ===")
    
    # Approach 1: Standard Scaling instead of Robust Scaling
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    
    numerical_features = ['montant', 'dureeMois', 'tauxAvance']
    categorical_features = ['procedure', 'formePrix', 'typeGroupementOperateurs', 'codeCPV_3']
    binary_features = ['attributionAvance', 'sousTraitanceDeclaree']
    
    # Standard approach with StandardScaler
    preprocessor_standard = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numerical_features),
        ('cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False), categorical_features),
        ('bin', make_pipeline(SimpleImputer(strategy='constant', fill_value=0), StandardScaler()), binary_features)
    ])
    
    # Test with different scalers
    preprocessors = {
        'Standard_Scaler': preprocessor_standard,
        'No_Scaling': ColumnTransformer([
            ('num', SimpleImputer(strategy='median'), numerical_features),
            ('cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False), categorical_features),
            ('bin', SimpleImputer(strategy='constant', fill_value=0), binary_features)
        ])
    }
    
    for name, preprocessor in preprocessors.items():
        try:
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)
            
            model = IsolationForest(contamination=0.05, random_state=0)
            model.fit(X_train_proc)
            predictions = model.predict(X_test_proc)
            
            n_anomalies = (predictions == -1).sum()
            percentage = (n_anomalies / len(X_test)) * 100
            
            print(f"{name:15s} | Anomalies: {n_anomalies:5d} ({percentage:.2f}%)")
        except Exception as e:
            print(f"{name:15s} | Error: {str(e)}")

def dimensionality_reduction_approach(X_train, X_test, preproc_baseline):
    """Try anomaly detection with dimensionality reduction"""
    print("\n=== Testing with Dimensionality Reduction ===")
    
    # Preprocess data
    X_train_processed = preproc_baseline.fit_transform(X_train)
    X_test_processed = preproc_baseline.transform(X_test)
    
    # Apply PCA
    for n_components in [10, 20, 50]:
        try:
            pca = PCA(n_components=n_components, random_state=0)
            X_train_pca = pca.fit_transform(X_train_processed)
            X_test_pca = pca.transform(X_test_processed)
            
            model = IsolationForest(contamination=0.05, random_state=0)
            model.fit(X_train_pca)
            predictions = model.predict(X_test_pca)
            
            n_anomalies = (predictions == -1).sum()
            percentage = (n_anomalies / len(X_test)) * 100
            
            explained_var = pca.explained_variance_ratio_.sum()
            
            print(f"PCA {n_components:2d} components | Anomalies: {n_anomalies:5d} ({percentage:.2f}%) | Explained Var: {explained_var:.3f}")
        except Exception as e:
            print(f"PCA {n_components:2d} components | Error: {str(e)}")

# Example usage functions
def main_analysis(X_train, X_test, preproc_baseline):
    """Run all analysis functions"""
    print("Running comprehensive anomaly detection analysis...\n")
    
    # 1. Test different contamination values
    contamination_results = test_contamination_values(X_train, X_test, preproc_baseline)
    
    # 2. Analyze anomaly scores
    best_contamination = contamination_results[contamination_results['n_anomalies'] > 0]['contamination'].iloc[0] if len(contamination_results[contamination_results['n_anomalies'] > 0]) > 0 else 0.05
    most_anomalous = analyze_anomaly_scores(X_train, X_test, preproc_baseline, best_contamination)
    
    # 3. Try alternative algorithms
    algorithm_results = try_alternative_algorithms(X_train, X_test, preproc_baseline)
    
    # 4. Feature-based analysis
    feature_based_analysis(X_train, X_test)
    
    # 5. Different preprocessing approaches
    improved_preprocessing_approach(X_train, X_test)
    
    # 6. Dimensionality reduction
    dimensionality_reduction_approach(X_train, X_test, preproc_baseline)
    
    return {
        'contamination_results': contamination_results,
        'most_anomalous': most_anomalous,
        'algorithm_results': algorithm_results
    }

if __name__ == "__main__":
    print("Anomaly Detection Improvement Toolkit")
    print("Import this module and use the functions with your data:")
    print("results = main_analysis(X_train, X_test, preproc_baseline)") 