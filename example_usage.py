#!/usr/bin/env python3
"""
Example usage of the updated GNN anomaly detection with preprocessed features
that include node columns in the preprocessed data
"""

import os
import numpy as np
from scripts.archive.gnn_anomaly_detection_onemodel import (
    ProcurementGraphBuilder, 
    GNNAnomalyDetector, 
    AnomalyAnalyzer
)

def main():
    # Initialize the graph builder
    graph_builder = ProcurementGraphBuilder()
    
    # Load and preprocess data
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
    df = graph_builder.load_data(data_path)
    X_train_preproc, X_test_preproc = graph_builder.preprocess_data(df)
    
    print("Original data shape:", df.shape)
    print("Preprocessed features shape:", X_train_preproc.shape)
    print("Feature columns:", list(X_train_preproc.columns))
    
    # Verify node columns are included
    node_cols = ['acheteur_nom', 'titulaire_nom']
    for col in node_cols:
        if col in X_train_preproc.columns:
            print(f"✓ {col} found in preprocessed data")
        else:
            print(f"✗ {col} missing from preprocessed data")
    
    # Create graph from preprocessed data (now includes node columns)
    graph_data = graph_builder.create_graph(X_train_preproc, df)
    
    print(f"\nGraph structure created:")
    print(f"- Number of nodes: {len(graph_data['nodes'])}")
    print(f"- Number of edges: {len(graph_data['edges'])}")
    print(f"- Node features shape: {graph_data['node_features'].shape}")
    print(f"- Edge features shape: {graph_data['edge_features'].shape}")
    print(f"- Number of feature columns: {len(graph_data['feature_columns'])}")
    
    # Show what features are being used for edges
    print(f"\nEdge feature columns (first 10):")
    for i, col in enumerate(graph_data['feature_columns'][:10]):
        print(f"  {i}: {col}")
    if len(graph_data['feature_columns']) > 10:
        print(f"  ... and {len(graph_data['feature_columns']) - 10} more")
    
    # Initialize GNN detector
    gnn_detector = GNNAnomalyDetector(hidden_dim=64, output_dim=32, num_layers=3)
    
    # Scale features before creating TensorFlow graph
    node_features = graph_data['node_features']
    edge_features = graph_data['edge_features']
    
    # Scale the features
    node_features_scaled = graph_builder.node_scaler.fit_transform(node_features)
    edge_features_scaled = graph_builder.edge_scaler.fit_transform(edge_features)
    
    print(f"\nScaled features:")
    print(f"- Node features scaled shape: {node_features_scaled.shape}")
    print(f"- Edge features scaled shape: {edge_features_scaled.shape}")
    
    # Create TensorFlow graph
    graph_tensor = gnn_detector.create_tensorflow_graph(
        graph_data, node_features_scaled, edge_features_scaled)
    
    # Build model
    gnn_detector.model = gnn_detector.build_model(
        node_features_scaled.shape[1], edge_features_scaled.shape[1])
    
    print(f"\nModel built:")
    print(f"- Input node feature dim: {node_features_scaled.shape[1]}")
    print(f"- Input edge feature dim: {edge_features_scaled.shape[1]}")
    
    # Train model (with fewer epochs for demonstration)
    print("\nTraining model...")
    history = gnn_detector.train(graph_tensor, epochs=10)
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    (node_reconstruction_error, edge_reconstruction_error, 
     node_threshold, edge_threshold) = gnn_detector.detect_anomalies()
    
    # Calculate anomaly masks
    node_anomalies = node_reconstruction_error > node_threshold
    edge_anomalies = edge_reconstruction_error > edge_threshold
    
    print(f"\nAnomaly detection results:")
    print(f"- Node anomalies detected: {np.sum(node_anomalies)} "
          f"({np.sum(node_anomalies)/len(node_anomalies)*100:.1f}%)")
    print(f"- Edge anomalies detected: {np.sum(edge_anomalies)} "
          f"({np.sum(edge_anomalies)/len(edge_anomalies)*100:.1f}%)")
    
    # Create results analysis
    analyzer = AnomalyAnalyzer()
    results_df = analyzer.create_results_dataframe(
        graph_data, node_reconstruction_error, node_anomalies)
    
    edge_results_df = analyzer.create_edge_results_dataframe(
        graph_data, edge_reconstruction_error, edge_anomalies)
    
    print(f"\nResults DataFrame columns:")
    print(list(results_df.columns))
    
    print(f"\nTop 5 most anomalous entities:")
    top_cols = ['entity_name', 'entity_type', 'node_reconstruction_error']
    if 'num_contracts' in results_df.columns:
        top_cols.append('num_contracts')
    if 'num_partners' in results_df.columns:
        top_cols.append('num_partners')
    
    top_anomalies = results_df.head(5)[top_cols]
    print(top_anomalies.to_string(index=False))
    
    print(f"\nTop 5 most anomalous contracts:")
    edge_cols = ['buyer_name', 'supplier_name', 'edge_reconstruction_error']
    if 'amount' in edge_results_df.columns:
        edge_cols.append('amount')
    
    top_edge_anomalies = edge_results_df.head(5)[edge_cols]
    print(top_edge_anomalies.to_string(index=False))
    
    # Test visualization (optional - will open in browser)
    try:
        print(f"\nGenerating network visualization...")
        graph_builder.visualize_procurement_graph(graph_data, 
            "Preprocessed French Public Procurement Network")
        print("Visualization saved and opened in browser!")
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    main() 