"""
Test GNN Anomaly Detection with Synthetic Anomalies

This module integrates synthetic anomaly generation with the GNN anomaly detection
pipeline to evaluate model performance on known anomalies.

Author: RonanB400
Date: January 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
import warnings

from scripts.synthetic_anomaly_generator import SyntheticAnomalyGenerator
from scripts.gnn_anomaly_detection_2 import ProcurementGraphBuilder, GNNAnomalyDetector, AnomalyAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNSyntheticAnomalyTester:
    """Test GNN models against synthetic anomalies."""
    
    def __init__(self, data_path: str, random_seed: int = 42):
        """Initialize the tester.
        
        Args:
            data_path: Path to the directory containing the data
            random_seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_seed = random_seed
        self.anomaly_generator = SyntheticAnomalyGenerator(random_seed)
        self.graph_builder = ProcurementGraphBuilder()
        self.gnn_detector = GNNAnomalyDetector(hidden_dim=64, output_dim=32, num_layers=3)
        self.analyzer = AnomalyAnalyzer()
        
        # Results storage
        self.test_results = {}
        
    def run_synthetic_anomaly_test(self, 
                                  anomaly_types: List[str] = None,
                                  anomaly_percentage: float = 0.05,
                                  test_sample_size: int = 5000,
                                  epochs: int = 30) -> Dict:
        """Run complete test pipeline with synthetic anomalies.
        
        Args:
            anomaly_types: Types of anomalies to generate and test
            anomaly_percentage: Percentage of data to make anomalous
            test_sample_size: Size of test dataset to use
            epochs: Number of training epochs
            
        Returns:
            Dictionary containing test results and metrics
        """
        
        logger.info("Starting GNN Synthetic Anomaly Test Pipeline...")
        
        # Step 1: Load and prepare data
        logger.info("Loading original data...")
        X_original = self.graph_builder.load_data(self.data_path)
        
        # Take a sample for testing (to speed up experiments)
        if len(X_original) > test_sample_size:
            X_sample = X_original.sample(n=test_sample_size, random_state=self.random_seed)
            logger.info(f"Using sample of {test_sample_size} contracts for testing")
        else:
            X_sample = X_original.copy()
            logger.info(f"Using full dataset of {len(X_original)} contracts")
        
        # Step 2: Generate synthetic anomalies
        logger.info("Generating synthetic anomalies...")
        X_with_anomalies, anomaly_labels = self.anomaly_generator.generate_anomalies(
            X_sample,
            anomaly_types=anomaly_types,
            anomaly_percentage=anomaly_percentage
        )
        
        # Step 3: Preprocess data (including anomalous data)
        logger.info("Preprocessing data with synthetic anomalies...")
        X_train_preproc, X_val_preproc, X_test_preproc, X_train, X_val, X_test = (
            self.graph_builder.preprocess_data(X_with_anomalies))
        
        # Step 4: Create graphs
        logger.info("Creating graph structures...")
        train_graph_data = self.graph_builder.create_graph(X_train_preproc, X_train, type='train')
        val_graph_data = self.graph_builder.create_graph(X_val_preproc, X_val, type='val')
        test_graph_data = self.graph_builder.create_graph(X_test_preproc, X_test, type='test')
        
        # Step 5: Scale features
        logger.info("Scaling features...")
        train_node_features_scaled = self.graph_builder.node_scaler.fit_transform(
            train_graph_data['node_features'])
        train_edge_features_scaled = self.graph_builder.edge_scaler.fit_transform(
            train_graph_data['edge_features'])
        
        val_node_features_scaled = self.graph_builder.node_scaler.transform(
            val_graph_data['node_features'])
        val_edge_features_scaled = self.graph_builder.edge_scaler.transform(
            val_graph_data['edge_features'])
        
        test_node_features_scaled = self.graph_builder.node_scaler.transform(
            test_graph_data['node_features'])
        test_edge_features_scaled = self.graph_builder.edge_scaler.transform(
            test_graph_data['edge_features'])
        
        # Step 6: Create TensorFlow graphs
        logger.info("Creating TensorFlow graph tensors...")
        train_graph_tensor = self.gnn_detector.create_tensorflow_graph(
            train_graph_data, train_node_features_scaled, train_edge_features_scaled)
        val_graph_tensor = self.gnn_detector.create_tensorflow_graph(
            val_graph_data, val_node_features_scaled, val_edge_features_scaled)
        test_graph_tensor = self.gnn_detector.create_tensorflow_graph(
            test_graph_data, test_node_features_scaled, test_edge_features_scaled)
        
        # Step 7: Build and train models
        logger.info("Building GNN models...")
        self.gnn_detector.node_model = self.gnn_detector.build_node_model(
            train_node_features_scaled.shape[1], train_edge_features_scaled.shape[1])
        self.gnn_detector.edge_model = self.gnn_detector.build_edge_model(
            train_node_features_scaled.shape[1], train_edge_features_scaled.shape[1])
        
        logger.info("Training models...")
        node_history = self.gnn_detector.train_node_model(
            train_graph_tensor, validation_graph_tensor=val_graph_tensor, epochs=epochs)
        edge_history = self.gnn_detector.train_edge_model(
            train_graph_tensor, validation_graph_tensor=val_graph_tensor, epochs=epochs)
        
        # Step 8: Detect anomalies on test set
        logger.info("Detecting anomalies on test set...")
        self.gnn_detector.graph_tensor_test = test_graph_tensor
        
        node_reconstruction_error, node_threshold = self.gnn_detector.detect_node_anomalies(
            threshold_percentile=95)  # Use 95th percentile for better detection
        edge_reconstruction_error, edge_threshold = self.gnn_detector.detect_edge_anomalies(
            threshold_percentile=95)
        
        # Step 9: Map predictions back to original indices and evaluate
        logger.info("Evaluating performance against synthetic anomaly labels...")
        
        # Get the test set indices to map back to original anomaly labels
        test_indices = X_test.index
        
        # Create ground truth labels for test set
        node_ground_truth = self._create_node_ground_truth(
            X_test, anomaly_labels, test_graph_data)
        edge_ground_truth = self._create_edge_ground_truth(
            X_test, anomaly_labels)
        
        # Calculate predictions
        node_predictions = node_reconstruction_error > node_threshold
        edge_predictions = edge_reconstruction_error > edge_threshold
        
        # Step 10: Calculate metrics
        results = self._calculate_comprehensive_metrics(
            node_ground_truth, node_predictions, node_reconstruction_error,
            edge_ground_truth, edge_predictions, edge_reconstruction_error,
            anomaly_labels, X_test, test_graph_data
        )
        
        # Store additional information
        results.update({
            'training_history': {
                'node_history': node_history,
                'edge_history': edge_history
            },
            'anomaly_summary': self.anomaly_generator.get_anomaly_summary(),
            'test_data_info': {
                'total_contracts': len(X_test),
                'total_entities': len(test_graph_data['nodes']),
                'anomaly_percentage_target': anomaly_percentage,
                'actual_anomaly_percentage': np.sum(X_test['is_synthetic_anomaly']) / len(X_test) * 100
            }
        })
        
        self.test_results = results
        logger.info("Synthetic anomaly testing completed!")
        
        return results
    
    def _create_node_ground_truth(self, X_test: pd.DataFrame, 
                                 anomaly_labels: Dict[str, np.ndarray],
                                 test_graph_data: Dict) -> np.ndarray:
        """Create ground truth labels for nodes based on synthetic anomalies."""
        
        # Map contract-level anomalies to node-level anomalies
        node_ground_truth = np.zeros(len(test_graph_data['nodes']), dtype=bool)
        
        # Get buyer and supplier mappings
        buyer_to_id = test_graph_data['buyer_to_id']
        supplier_to_id = test_graph_data['supplier_to_id']
        
        # Find which entities (buyers/suppliers) are involved in anomalous contracts
        anomalous_entities = set()
        
        for contract_idx, contract in X_test.iterrows():
            # Check if this contract has any synthetic anomaly
            is_anomalous = contract.get('is_synthetic_anomaly', False)
            
            if is_anomalous:
                # Add both buyer and supplier as potentially anomalous
                buyer_id = contract.get('acheteur_id')
                supplier_id = contract.get('titulaire_id')
                
                if pd.notna(buyer_id) and buyer_id in buyer_to_id:
                    anomalous_entities.add(buyer_to_id[buyer_id])
                if pd.notna(supplier_id) and supplier_id in supplier_to_id:
                    anomalous_entities.add(supplier_to_id[supplier_id])
        
        # Mark these entities as anomalous
        for entity_idx in anomalous_entities:
            if entity_idx < len(node_ground_truth):
                node_ground_truth[entity_idx] = True
        
        logger.info(f"Created node ground truth: {np.sum(node_ground_truth)} anomalous entities")
        return node_ground_truth
    
    def _create_edge_ground_truth(self, X_test: pd.DataFrame, 
                                 anomaly_labels: Dict[str, np.ndarray]) -> np.ndarray:
        """Create ground truth labels for edges based on synthetic anomalies."""
        
        # Edge ground truth is directly the contract-level anomaly labels
        edge_ground_truth = X_test['is_synthetic_anomaly'].values.astype(bool)
        
        logger.info(f"Created edge ground truth: {np.sum(edge_ground_truth)} anomalous contracts")
        return edge_ground_truth
    
    def _calculate_comprehensive_metrics(self, 
                                       node_gt: np.ndarray, node_pred: np.ndarray, node_scores: np.ndarray,
                                       edge_gt: np.ndarray, edge_pred: np.ndarray, edge_scores: np.ndarray,
                                       anomaly_labels: Dict[str, np.ndarray],
                                       X_test: pd.DataFrame,
                                       test_graph_data: Dict) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        
        results = {}
        
        # Node-level metrics
        if len(node_gt) > 0 and np.sum(node_gt) > 0:
            results['node_metrics'] = {
                'classification_report': classification_report(node_gt, node_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(node_gt, node_pred).tolist(),
                'auc_roc': roc_auc_score(node_gt, node_scores) if np.sum(node_gt) > 0 and np.sum(node_gt) < len(node_gt) else None,
                'num_anomalies_detected': int(np.sum(node_pred)),
                'num_true_anomalies': int(np.sum(node_gt)),
                'detection_rate': float(np.sum(node_pred & node_gt) / np.sum(node_gt)) if np.sum(node_gt) > 0 else 0.0
            }
        else:
            results['node_metrics'] = {'error': 'No node anomalies in test set'}
        
        # Edge-level metrics
        if len(edge_gt) > 0 and np.sum(edge_gt) > 0:
            results['edge_metrics'] = {
                'classification_report': classification_report(edge_gt, edge_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(edge_gt, edge_pred).tolist(),
                'auc_roc': roc_auc_score(edge_gt, edge_scores) if np.sum(edge_gt) > 0 and np.sum(edge_gt) < len(edge_gt) else None,
                'num_anomalies_detected': int(np.sum(edge_pred)),
                'num_true_anomalies': int(np.sum(edge_gt)),
                'detection_rate': float(np.sum(edge_pred & edge_gt) / np.sum(edge_gt)) if np.sum(edge_gt) > 0 else 0.0
            }
        else:
            results['edge_metrics'] = {'error': 'No edge anomalies in test set'}
        
        # Per-anomaly-type metrics (only for edges/contracts)
        results['per_anomaly_type_metrics'] = {}
        
        # Map test indices back to original indices for anomaly label lookup
        test_original_indices = X_test.index
        
        for anomaly_type, original_labels in anomaly_labels.items():
            if anomaly_type == 'is_synthetic_anomaly':
                continue
                
            # Get labels for test set only
            test_type_labels = np.zeros(len(X_test), dtype=bool)
            for i, original_idx in enumerate(test_original_indices):
                if original_idx < len(original_labels):
                    test_type_labels[i] = original_labels[original_idx]
            
            if np.sum(test_type_labels) > 0:  # Only if we have this type of anomaly in test set
                type_detection_rate = float(np.sum(edge_pred & test_type_labels) / np.sum(test_type_labels))
                results['per_anomaly_type_metrics'][anomaly_type] = {
                    'num_true_anomalies': int(np.sum(test_type_labels)),
                    'num_detected': int(np.sum(edge_pred & test_type_labels)),
                    'detection_rate': type_detection_rate,
                    'false_positive_rate': float(np.sum(edge_pred & ~test_type_labels) / np.sum(~test_type_labels)) if np.sum(~test_type_labels) > 0 else 0.0
                }
        
        return results
    
    def plot_comprehensive_results(self, save_dir: str = None):
        """Create comprehensive visualizations of test results."""
        
        if not self.test_results:
            logger.error("No test results available. Run test first.")
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Plot 1: Training histories
        ax1 = plt.subplot(3, 3, 1)
        self._plot_training_history(ax1)
        
        # Plot 2: Confusion matrices
        ax2 = plt.subplot(3, 3, 2)
        self._plot_confusion_matrices(ax2)
        
        # Plot 3: ROC curves
        ax3 = plt.subplot(3, 3, 3)
        self._plot_roc_curves(ax3)
        
        # Plot 4: Detection rates by anomaly type
        ax4 = plt.subplot(3, 3, 4)
        self._plot_detection_rates_by_type(ax4)
        
        # Plot 5: Score distributions
        ax5 = plt.subplot(3, 3, 5)
        self._plot_score_distributions(ax5)
        
        # Plot 6: Precision-Recall curves
        ax6 = plt.subplot(3, 3, 6)
        self._plot_precision_recall_curves(ax6)
        
        # Plot 7: Summary metrics table
        ax7 = plt.subplot(3, 3, 7)
        self._plot_summary_table(ax7)
        
        # Plot 8: Anomaly type breakdown
        ax8 = plt.subplot(3, 3, 8)
        self._plot_anomaly_breakdown(ax8)
        
        # Plot 9: Model performance comparison
        ax9 = plt.subplot(3, 3, 9)
        self._plot_model_comparison(ax9)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'synthetic_anomaly_test_results.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to {save_path}")
        
        plt.show()
    
    def _plot_training_history(self, ax):
        """Plot training loss histories."""
        node_history = self.test_results['training_history']['node_history']
        edge_history = self.test_results['training_history']['edge_history']
        
        epochs = range(1, len(node_history['loss']) + 1)
        
        ax.plot(epochs, node_history['loss'], 'b-', label='Node Model', linewidth=2)
        ax.plot(epochs, edge_history['loss'], 'r-', label='Edge Model', linewidth=2)
        
        if 'val_loss' in node_history:
            ax.plot(epochs, node_history['val_loss'], 'b--', label='Node Val', alpha=0.7)
        if 'val_loss' in edge_history:
            ax.plot(epochs, edge_history['val_loss'], 'r--', label='Edge Val', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_confusion_matrices(self, ax):
        """Plot confusion matrices for both models."""
        if 'node_metrics' in self.test_results and 'confusion_matrix' in self.test_results['node_metrics']:
            node_cm = np.array(self.test_results['node_metrics']['confusion_matrix'])
            
            # Simple text-based confusion matrix display
            ax.text(0.1, 0.8, "Node Model Confusion Matrix:", fontsize=12, fontweight='bold', transform=ax.transAxes)
            ax.text(0.1, 0.7, f"TN: {node_cm[0,0]}, FP: {node_cm[0,1]}", fontsize=10, transform=ax.transAxes)
            ax.text(0.1, 0.6, f"FN: {node_cm[1,0]}, TP: {node_cm[1,1]}", fontsize=10, transform=ax.transAxes)
        
        if 'edge_metrics' in self.test_results and 'confusion_matrix' in self.test_results['edge_metrics']:
            edge_cm = np.array(self.test_results['edge_metrics']['confusion_matrix'])
            
            ax.text(0.1, 0.4, "Edge Model Confusion Matrix:", fontsize=12, fontweight='bold', transform=ax.transAxes)
            ax.text(0.1, 0.3, f"TN: {edge_cm[0,0]}, FP: {edge_cm[0,1]}", fontsize=10, transform=ax.transAxes)
            ax.text(0.1, 0.2, f"FN: {edge_cm[1,0]}, TP: {edge_cm[1,1]}", fontsize=10, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Confusion Matrices')
        ax.axis('off')
    
    def _plot_roc_curves(self, ax):
        """Plot ROC curves."""
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        
        if 'node_metrics' in self.test_results and self.test_results['node_metrics'].get('auc_roc'):
            auc = self.test_results['node_metrics']['auc_roc']
            ax.text(0.6, 0.2, f'Node AUC: {auc:.3f}', fontsize=10, transform=ax.transAxes)
        
        if 'edge_metrics' in self.test_results and self.test_results['edge_metrics'].get('auc_roc'):
            auc = self.test_results['edge_metrics']['auc_roc']
            ax.text(0.6, 0.1, f'Edge AUC: {auc:.3f}', fontsize=10, transform=ax.transAxes)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_detection_rates_by_type(self, ax):
        """Plot detection rates by anomaly type."""
        per_type_metrics = self.test_results['per_anomaly_type_metrics']
        
        if per_type_metrics:
            anomaly_types = list(per_type_metrics.keys())
            detection_rates = [per_type_metrics[t]['detection_rate'] for t in anomaly_types]
            
            bars = ax.bar(range(len(anomaly_types)), detection_rates, alpha=0.7)
            ax.set_xticks(range(len(anomaly_types)))
            ax.set_xticklabels([t.replace('_', '\n') for t in anomaly_types], rotation=45, ha='right')
            ax.set_ylabel('Detection Rate')
            ax.set_title('Detection Rate by Anomaly Type')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, rate in zip(bars, detection_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{rate:.2f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No per-type metrics available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Detection Rate by Anomaly Type')
    
    def _plot_score_distributions(self, ax):
        """Plot anomaly score distributions."""
        # This would require access to the actual scores, which we'd need to store
        ax.text(0.5, 0.5, 'Score distributions\n(implementation pending)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Anomaly Score Distributions')
    
    def _plot_precision_recall_curves(self, ax):
        """Plot precision-recall curves."""
        ax.text(0.5, 0.5, 'Precision-Recall curves\n(implementation pending)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Precision-Recall Curves')
    
    def _plot_summary_table(self, ax):
        """Plot summary metrics table."""
        summary_text = "SUMMARY METRICS\n\n"
        
        if 'node_metrics' in self.test_results and 'error' not in self.test_results['node_metrics']:
            node_metrics = self.test_results['node_metrics']
            f1 = node_metrics['classification_report']['weighted avg']['f1-score']
            precision = node_metrics['classification_report']['weighted avg']['precision']
            recall = node_metrics['classification_report']['weighted avg']['recall']
            
            summary_text += f"NODE MODEL:\n"
            summary_text += f"F1-Score: {f1:.3f}\n"
            summary_text += f"Precision: {precision:.3f}\n"
            summary_text += f"Recall: {recall:.3f}\n\n"
        
        if 'edge_metrics' in self.test_results and 'error' not in self.test_results['edge_metrics']:
            edge_metrics = self.test_results['edge_metrics']
            f1 = edge_metrics['classification_report']['weighted avg']['f1-score']
            precision = edge_metrics['classification_report']['weighted avg']['precision']
            recall = edge_metrics['classification_report']['weighted avg']['recall']
            
            summary_text += f"EDGE MODEL:\n"
            summary_text += f"F1-Score: {f1:.3f}\n"
            summary_text += f"Precision: {precision:.3f}\n"
            summary_text += f"Recall: {recall:.3f}\n"
        
        ax.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top', transform=ax.transAxes)
        ax.set_title('Summary Metrics')
        ax.axis('off')
    
    def _plot_anomaly_breakdown(self, ax):
        """Plot breakdown of anomaly types."""
        anomaly_summary = self.test_results['anomaly_summary']
        
        if len(anomaly_summary) > 0:
            ax.pie(anomaly_summary['count'], labels=anomaly_summary['anomaly_type'], autopct='%1.1f%%')
            ax.set_title('Anomaly Type Distribution')
        else:
            ax.text(0.5, 0.5, 'No anomaly data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Anomaly Type Distribution')
    
    def _plot_model_comparison(self, ax):
        """Plot comparison between node and edge models."""
        models = []
        f1_scores = []
        
        if 'node_metrics' in self.test_results and 'error' not in self.test_results['node_metrics']:
            models.append('Node Model')
            f1_scores.append(self.test_results['node_metrics']['classification_report']['weighted avg']['f1-score'])
        
        if 'edge_metrics' in self.test_results and 'error' not in self.test_results['edge_metrics']:
            models.append('Edge Model')
            f1_scores.append(self.test_results['edge_metrics']['classification_report']['weighted avg']['f1-score'])
        
        if models:
            bars = ax.bar(models, f1_scores, alpha=0.7, color=['skyblue', 'lightcoral'])
            ax.set_ylabel('F1-Score')
            ax.set_title('Model Performance Comparison')
            ax.set_ylim(0, 1)
            
            for bar, score in zip(bars, f1_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No model comparison data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Performance Comparison')
    
    def print_detailed_results(self):
        """Print detailed test results."""
        
        if not self.test_results:
            print("No test results available. Run test first.")
            return
        
        print("\n" + "="*80)
        print("SYNTHETIC ANOMALY DETECTION TEST RESULTS")
        print("="*80)
        
        # Test data info
        test_info = self.test_results['test_data_info']
        print(f"\nTest Data Information:")
        print(f"- Total contracts: {test_info['total_contracts']}")
        print(f"- Total entities: {test_info['total_entities']}")
        print(f"- Target anomaly percentage: {test_info['anomaly_percentage_target']*100:.1f}%")
        print(f"- Actual anomaly percentage: {test_info['actual_anomaly_percentage']:.1f}%")
        
        # Anomaly summary
        print(f"\nSynthetic Anomalies Generated:")
        anomaly_summary = self.test_results['anomaly_summary']
        for _, row in anomaly_summary.iterrows():
            print(f"- {row['anomaly_type']}: {row['count']} ({row['percentage']:.1f}%)")
        
        # Node model results
        if 'node_metrics' in self.test_results and 'error' not in self.test_results['node_metrics']:
            print(f"\nNODE MODEL RESULTS:")
            node_metrics = self.test_results['node_metrics']
            print(f"- True anomalies: {node_metrics['num_true_anomalies']}")
            print(f"- Detected anomalies: {node_metrics['num_anomalies_detected']}")
            print(f"- Detection rate: {node_metrics['detection_rate']:.3f}")
            if node_metrics['auc_roc']:
                print(f"- AUC-ROC: {node_metrics['auc_roc']:.3f}")
            
            print(f"\nClassification Report (Node Model):")
            cr = node_metrics['classification_report']
            print(f"- Precision: {cr['weighted avg']['precision']:.3f}")
            print(f"- Recall: {cr['weighted avg']['recall']:.3f}")
            print(f"- F1-Score: {cr['weighted avg']['f1-score']:.3f}")
        else:
            print(f"\nNODE MODEL RESULTS: {self.test_results.get('node_metrics', {}).get('error', 'Not available')}")
        
        # Edge model results
        if 'edge_metrics' in self.test_results and 'error' not in self.test_results['edge_metrics']:
            print(f"\nEDGE MODEL RESULTS:")
            edge_metrics = self.test_results['edge_metrics']
            print(f"- True anomalies: {edge_metrics['num_true_anomalies']}")
            print(f"- Detected anomalies: {edge_metrics['num_anomalies_detected']}")
            print(f"- Detection rate: {edge_metrics['detection_rate']:.3f}")
            if edge_metrics['auc_roc']:
                print(f"- AUC-ROC: {edge_metrics['auc_roc']:.3f}")
            
            print(f"\nClassification Report (Edge Model):")
            cr = edge_metrics['classification_report']
            print(f"- Precision: {cr['weighted avg']['precision']:.3f}")
            print(f"- Recall: {cr['weighted avg']['recall']:.3f}")
            print(f"- F1-Score: {cr['weighted avg']['f1-score']:.3f}")
        else:
            print(f"\nEDGE MODEL RESULTS: {self.test_results.get('edge_metrics', {}).get('error', 'Not available')}")
        
        # Per-anomaly-type results
        if self.test_results['per_anomaly_type_metrics']:
            print(f"\nPER-ANOMALY-TYPE DETECTION RATES:")
            for anomaly_type, metrics in self.test_results['per_anomaly_type_metrics'].items():
                print(f"- {anomaly_type}: {metrics['detection_rate']:.3f} "
                      f"({metrics['num_detected']}/{metrics['num_true_anomalies']})")
        
        print("\n" + "="*80)


def main():
    """Example usage of the synthetic anomaly tester."""
    
    # Configuration
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data')
    
    # Initialize tester
    tester = GNNSyntheticAnomalyTester(DATA_PATH)
    
    # Run test with specific anomaly types
    test_anomaly_types = [
        'single_bid_competitive',
        'price_manipulation', 
        'high_market_concentration',
        'procedure_manipulation'
    ]
    
    # Run the test
    results = tester.run_synthetic_anomaly_test(
        anomaly_types=test_anomaly_types,
        anomaly_percentage=0.08,  # 8% anomalies
        test_sample_size=3000,    # Use 3000 contracts for testing
        epochs=25                 # Train for 25 epochs
    )
    
    # Print detailed results
    tester.print_detailed_results()
    
    # Create visualizations
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data', 'results')
    tester.plot_comprehensive_results(save_dir=save_dir)


if __name__ == "__main__":
    main()