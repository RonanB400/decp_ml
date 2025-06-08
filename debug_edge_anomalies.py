# Debug script for edge anomaly detection issues

import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_edge_anomaly_detection(gnn_detector):
    """
    Debug function to identify issues with edge anomaly detection.
    
    Usage:
        # After training your edge model:
        from debug_edge_anomalies import debug_edge_anomaly_detection
        debug_edge_anomaly_detection(gnn_detector)
    """
    
    print("="*60)
    print("DEBUGGING EDGE ANOMALY DETECTION")
    print("="*60)
    
    # Step 1: Check if models exist
    print("\n1. Checking models...")
    if gnn_detector.edge_model is None:
        print("❌ Edge model is None - you need to build and train it first!")
        return
    else:
        print("✅ Edge model exists")
    
    # Step 2: Check if graph tensors exist
    print("\n2. Checking graph tensors...")
    if gnn_detector.graph_tensor_test is None:
        print("❌ Test graph tensor is None - you need to create it first!")
        return
    else:
        print("✅ Test graph tensor exists")
    
    # Step 3: Run diagnostics
    print("\n3. Running edge model diagnostics...")
    try:
        diagnostics = gnn_detector.diagnose_edge_model()
        
        print(f"📊 Edge features shape: {diagnostics.get('edge_features_shape', 'Unknown')}")
        print(f"📊 Edge features NaN count: {diagnostics.get('edge_features_nan_count', 'Unknown')}")
        print(f"📊 Edge features Inf count: {diagnostics.get('edge_features_inf_count', 'Unknown')}")
        print(f"📊 Edge features finite count: {diagnostics.get('edge_features_finite_count', 'Unknown')}")
        
        if 'prediction_error' in diagnostics:
            print(f"❌ Prediction error: {diagnostics['prediction_error']}")
        else:
            print(f"📊 Prediction shape: {diagnostics.get('prediction_shape', 'Unknown')}")
            print(f"📊 Prediction NaN count: {diagnostics.get('prediction_nan_count', 'Unknown')}")
            print(f"📊 Prediction Inf count: {diagnostics.get('prediction_inf_count', 'Unknown')}")
            
            if diagnostics.get('prediction_nan_count', 0) > 0:
                print("⚠️  WARNING: Model predictions contain NaN values!")
                
            if diagnostics.get('prediction_inf_count', 0) > 0:
                print("⚠️  WARNING: Model predictions contain infinite values!")
    
    except Exception as e:
        print(f"❌ Error during diagnostics: {str(e)}")
        return
    
    # Step 4: Test anomaly detection
    print("\n4. Testing anomaly detection...")
    try:
        edge_reconstruction_error, edge_threshold = gnn_detector.detect_edge_anomalies()
        
        nan_count = np.sum(np.isnan(edge_reconstruction_error))
        total_count = len(edge_reconstruction_error)
        nan_percentage = (nan_count / total_count) * 100
        
        print(f"📊 Reconstruction errors shape: {edge_reconstruction_error.shape}")
        print(f"📊 NaN errors: {nan_count}/{total_count} ({nan_percentage:.1f}%)")
        print(f"📊 Threshold: {edge_threshold}")
        
        if nan_percentage > 50:
            print("❌ PROBLEM: High percentage of NaN reconstruction errors!")
            print("   This suggests issues with:")
            print("   - Model training (model might not have converged)")
            print("   - Feature scaling (features might contain NaN/Inf)")
            print("   - Model architecture (gradients might be exploding/vanishing)")
        
    except Exception as e:
        print(f"❌ Error during anomaly detection: {str(e)}")
    
    # Step 5: Recommendations
    print("\n5. Recommendations...")
    print("To fix the issue, try:")
    print("1. Check your training history - did the model converge?")
    print("2. Verify feature scaling - are there NaN/Inf values in input features?")
    print("3. Check model architecture - try simpler architecture or different regularization")
    print("4. Verify data preprocessing - ensure no data corruption")
    print("5. Try training with different hyperparameters")
    
    print("\n" + "="*60)


def quick_feature_check(X_train_graph, X_test_graph):
    """
    Quick check of graph features for common issues.
    
    Usage:
        from debug_edge_anomalies import quick_feature_check
        quick_feature_check(X_train_graph, X_test_graph)
    """
    
    print("="*60)
    print("QUICK FEATURE CHECK")
    print("="*60)
    
    # Check training features
    print("\n📊 Training Graph:")
    train_edge_features = X_train_graph['edge_features']
    print(f"  Shape: {train_edge_features.shape}")
    print(f"  NaN count: {np.sum(np.isnan(train_edge_features))}")
    print(f"  Inf count: {np.sum(np.isinf(train_edge_features))}")
    print(f"  Min: {np.nanmin(train_edge_features):.6f}")
    print(f"  Max: {np.nanmax(train_edge_features):.6f}")
    print(f"  Mean: {np.nanmean(train_edge_features):.6f}")
    
    # Check test features
    print("\n📊 Test Graph:")
    test_edge_features = X_test_graph['edge_features']
    print(f"  Shape: {test_edge_features.shape}")
    print(f"  NaN count: {np.sum(np.isnan(test_edge_features))}")
    print(f"  Inf count: {np.sum(np.isinf(test_edge_features))}")
    print(f"  Min: {np.nanmin(test_edge_features):.6f}")
    print(f"  Max: {np.nanmax(test_edge_features):.6f}")
    print(f"  Mean: {np.nanmean(test_edge_features):.6f}")
    
    # Check for potential issues
    print("\n🔍 Potential Issues:")
    if np.sum(np.isnan(train_edge_features)) > 0:
        print("  ❌ Training features contain NaN values")
    if np.sum(np.isnan(test_edge_features)) > 0:
        print("  ❌ Test features contain NaN values")
    if np.sum(np.isinf(train_edge_features)) > 0:
        print("  ❌ Training features contain infinite values")
    if np.sum(np.isinf(test_edge_features)) > 0:
        print("  ❌ Test features contain infinite values")
    
    if (np.sum(np.isnan(train_edge_features)) == 0 and 
        np.sum(np.isnan(test_edge_features)) == 0 and
        np.sum(np.isinf(train_edge_features)) == 0 and 
        np.sum(np.isinf(test_edge_features)) == 0):
        print("  ✅ Features look clean (no NaN/Inf values)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("This is a debugging module. Import the functions in your main script:")
    print("from debug_edge_anomalies import debug_edge_anomaly_detection, quick_feature_check") 