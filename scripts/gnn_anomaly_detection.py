"""
Graph Neural Networks for Anomaly Detection in Public Procurement

This module implements Graph Neural Networks using tensorflow_gnn to
analyze buyer-supplier relationships and detect potential anomalies
in public procurement data.

Author: RonanB400
Date: January 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import logging
from typing import Dict, Tuple, List

import tensorflow as tf
import tensorflow_gnn as tfgnn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import networkx as nx

from scripts.preprocess_pipeline import create_pipeline
from scripts.data_cleaner import filter_top_cpv_categories

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcurementGraphBuilder:
    """Build graph structures from procurement data."""
    
    def __init__(self):
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load procurement data from CSV files."""
        logger.info(f"Loading data from {data_path}")

        X = pd.read_csv(os.path.join(data_path, 'data_clean.csv'), encoding='utf-8')
        # Basic data validation
        required_columns = ['acheteur_id', 'titulaire_id', 'montant',
                            'dateNotification']
        missing_cols = [col for col in required_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return X
    
    def preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        logger.info("Preprocessing data...")
        
        # Fill missing values
        X = filter_top_cpv_categories(X, top_n=60, cpv_column='codeCPV_3')

        # Preprocess pipeline
        numerical_columns = ['montant', 'dureeMois', 'offresRecues']

        binary_columns = ['sousTraitanceDeclaree', 'origineFrance', 
                          'marcheInnovant', 'idAccordCadre']
        
        categorical_columns = ['procedure', 'nature', 'formePrix', 'ccag',
                               'typeGroupementOperateurs', 'tauxAvance_cat',
                               'codeCPV_3']
        
        nodes_columns = ['acheteur_id', 'titulaire_id']
        
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=0, stratify=X['codeCPV_3'])
        
        preproc_pipeline = create_pipeline(numerical_columns, binary_columns, categorical_columns)

        X_train_preproc = preproc_pipeline.fit_transform(X_train)
        X_train_preproc.index = X_train.index
        X_train_preproc = pd.concat([X_train_preproc, X_train[nodes_columns]], axis=1)

        X_test_preproc = preproc_pipeline.transform(X_test)
        X_test_preproc.index = X_test.index
        X_test_preproc = pd.concat([X_test_preproc, X_test[nodes_columns]], axis=1)
        
        return X_train_preproc, X_test_preproc, X_train, X_test
    

    
    def create_graph(self, X_processed: pd.DataFrame,
                     X_original: pd.DataFrame = None, type: str = 'train') -> Dict:
        """Transform preprocessed procurement data into graph structure.
        
        Args:
            X_processed: Preprocessed dataframe with encoded features
            X_original: Original dataframe for metadata (optional)
        """
        logger.info("Creating graph structure from preprocessed data...")
        
        # Remove rows with NaN buyer or supplier names
        valid_mask = (X_processed['acheteur_id'].notna() & 
                     X_processed['titulaire_id'].notna())
        X_processed = X_processed[valid_mask].copy()
        
        if X_original is not None:
            X_original = X_original[valid_mask].copy()
        
        logger.info(f"Filtered to {len(X_processed)} valid contracts "
                   f"(removed {(~valid_mask).sum()} contracts with missing names)")
        
        # Create unique identifiers for buyers and suppliers
        buyers = X_processed['acheteur_id'].unique()
        suppliers = X_processed['titulaire_id'].unique()
        
        # Create node mappings
        # buyer_to_id / supplier_to_id is a dictionary that maps each buyer / supplier to a unique integer
        buyer_to_id = {buyer: i for i, buyer in enumerate(buyers)}
        supplier_to_id = {supplier: i + len(buyers)
                          for i, supplier in enumerate(suppliers)}
        
        # Combine all nodes
        all_nodes = list(buyers) + list(suppliers)
        
        # OPTIMIZATION 1: Vectorized edge creation - ONE EDGE PER CONTRACT
        logger.info("Creating edges and edge features from preprocessed data...")
        
        # Map buyer and supplier names to IDs using vectorized operations
        # buyer_ids / supplier_ids is a list of integers that correspond to the unique integer ID of each buyer / supplier
        buyer_ids = X_processed['acheteur_id'].map(buyer_to_id).values.astype(np.int32)
        supplier_ids = X_processed['titulaire_id'].map(supplier_to_id).values.astype(np.int32)
        edges = np.column_stack([buyer_ids, supplier_ids]).astype(np.int32)
        
        # Extract all feature columns (excluding entity names)
        feature_columns = [col for col in X_processed.columns
                           if col not in ['acheteur_id', 'titulaire_id']]
        
        # Create edge features from all preprocessed features
        edge_features = X_processed[feature_columns].values.astype(np.float32)
        
        # Store contract IDs for edge-level analysis
        # contract_ids
        contract_ids = X_processed.index.tolist()
        
        # OPTIMIZATION 2: Bulk computation of node features from preprocessed data
        logger.info("Computing acheteur features from preprocessed data...")

        # Pre-compute aggregations for buyers
        buyer_stats = self._compute_bulk_node_features_preprocessed(
            X_processed, feature_columns, 'acheteur_id', 'titulaire_id')
        
        logger.info("Computing titulaire features from preprocessed data...")
        # Pre-compute aggregations for suppliers  
        supplier_stats = self._compute_bulk_node_features_preprocessed(
            X_processed, feature_columns, 'titulaire_id', 'acheteur_id')
        
        # Build node features arrays
        node_features = []
        node_types = []
        
        # Buyer features
        for buyer in buyers:
            features = buyer_stats[buyer]
            node_features.append(features)
            node_types.append(0)  # Buyer
        
        # Supplier features
        for supplier in suppliers:
            features = supplier_stats[supplier]
            node_features.append(features)
            node_types.append(1)  # Supplier
        
        # Create contract data for analysis (use original data if available)
        if X_original is not None:
            contract_data = X_original[['acheteur_id', 'titulaire_id',
                                        'montant', 'codeCPV_3', 'procedure',
                                        'dureeMois']].copy()
        else:
            # Create minimal contract data from preprocessed features
            contract_data = pd.DataFrame({
                'acheteur_id': X_processed['acheteur_id'],
                'titulaire_id': X_processed['titulaire_id'],
                'montant': X_processed['other_num_pipeline__montant'],
                'codeCPV_3': 'preprocessed',  # Placeholder
                'procedure': 'preprocessed',  # Placeholder
                'dureeMois': 'preprocessed'  # Placeholder
            })
        
        graph_data = {
            'nodes': all_nodes,
            'edges': edges,
            'node_features': np.array(node_features, dtype=np.float32),
            'edge_features': edge_features,
            'node_types': np.array(node_types, dtype=np.int32),
            'buyer_to_id': buyer_to_id,
            'supplier_to_id': supplier_to_id,
            'contract_ids': contract_ids,
            'contract_data': contract_data,
            'feature_columns': feature_columns  # Store feature column names
        }

        # Save the graph data to a pickle file
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data')
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, f'graph_data_{type}.pkl'), 'wb') as f:
            pickle.dump(graph_data, f)
        
        return graph_data
    
    def _compute_bulk_node_features_preprocessed(self, X_processed: pd.DataFrame,
                                                feature_columns: List[str],
                                                entity_col: str, 
                                                partner_col: str) -> Dict[str, List[float]]:
        """Compute node features for all entities using preprocessed data."""
        
        # Group by entity and compute basic stats on preprocessed features
        entity_groups = X_processed.groupby(entity_col)
        
        entity_features = {}
        for entity, group in entity_groups:
            # Basic contract count and partner count
            num_contracts = len(group)
            num_partners = group[partner_col].nunique()
            
            # Aggregate the preprocessed features
            feature_stats = []
            
            # Mean of all preprocessed features
            feature_means = group[feature_columns].mean().values
            feature_stats.extend(feature_means)
            
            # Standard deviation of numerical features
            numerical_features = [col for col in feature_columns 
                                if col.startswith(('other_num_pipeline__', 
                                                  'offres_recues_pipeline__'))]
            if numerical_features:
                feature_stds = group[numerical_features].std().fillna(0).values
                feature_stats.extend(feature_stds)
            
            # Add basic entity statistics
            feature_stats.extend([
                num_contracts,
                num_partners,
                num_contracts / max(num_partners, 1)  # Contracts per partner
            ])
            
            entity_features[entity] = feature_stats
            
        return entity_features
    
    def visualize_procurement_graph(self, graph_data: Dict, title: str = "Procurement Network"):
        """Create an interactive visualization of the full procurement graph.
        
        Args:
            graph_data: Dictionary containing the graph data from create_graph
            title: Title for the visualization
        """
        from pyvis.network import Network
        import webbrowser
        import os
        
        # Create a new network
        net = Network(height="900px", width="100%", bgcolor="#ffffff",
                    font_color="black", notebook=False)
        
        # Add nodes
        for i, (name, node_type) in enumerate(zip(
            graph_data['nodes'], graph_data['node_types'])):
            
            # Calculate node size and color based on available features
            node_features = graph_data['node_features'][i]
            
            # Try to get meaningful metrics from node features
            if len(node_features) >= 3:
                # Last 3 features are: num_contracts, num_partners, contracts_per_partner
                num_contracts = int(node_features[-3]) if not np.isnan(node_features[-3]) else 1
                node_size = min(50 + num_contracts * 2, 100)
                
                # Use mean of feature values for color
                if len(node_features) > 3:
                    valid_features = node_features[:-3]
                    valid_features = valid_features[~np.isnan(valid_features)]
                    feature_value = float(np.mean(valid_features)) if len(valid_features) > 0 else 0.0
                else:
                    feature_value = float(node_features[-1]) if not np.isnan(node_features[-1]) else 0.0
            else:
                # Fallback
                num_contracts = 1
                node_size = 50
                feature_value = float(node_features[0]) if len(node_features) > 0 and not np.isnan(node_features[0]) else 0.0
            
            # Normalize feature value to a color scale (blue to red)
            # Use percentile-based normalization for better color distribution
            all_feature_values = []
            for nf in graph_data['node_features']:
                if len(nf) > 3:
                    valid_features = nf[:-3]
                    valid_features = valid_features[~np.isnan(valid_features)]
                    if len(valid_features) > 0:
                        all_feature_values.append(float(np.mean(valid_features)))
                    else:
                        all_feature_values.append(0.0)
                elif len(nf) >= 1:
                    val = float(nf[-1]) if not np.isnan(nf[-1]) else 0.0
                    all_feature_values.append(val)
                else:
                    all_feature_values.append(0.0)
            
            percentile_90 = np.percentile(all_feature_values, 90) if len(all_feature_values) > 0 else 1.0
            if np.isnan(percentile_90) or percentile_90 <= 0:
                percentile_90 = 1.0
            if np.isnan(feature_value):
                feature_value = 0.0
                
            feature_ratio = min(feature_value / max(percentile_90, 1), 1.0)
            if np.isnan(feature_ratio):
                feature_ratio = 0.0
                
            color = f"rgb({int(255 * feature_ratio)}, 0, {int(255 * (1 - feature_ratio))})"
            
            # Create tooltip with available information
            num_partners = int(node_features[-2]) if len(node_features) >= 2 and not np.isnan(node_features[-2]) else 0
            tooltip = f"Type: {'Buyer' if node_type == 0 else 'Supplier'}\n"
            tooltip += f"Contracts: {num_contracts}\n"
            tooltip += f"Partners: {num_partners}\n"
            tooltip += f"Feature Value: {feature_value:.3f}"
            
            # Add node with properties
            net.add_node(
                int(i),  # Convert to Python int
                label=str(name),  # Convert to Python string
                title=tooltip,
                color=color,
                size=node_size,
                shape="diamond" if node_type == 0 else "dot"
            )
        
        # Add edges with weights based on contract amounts
        for i, edge in enumerate(graph_data['edges']):
            # Get contract amount from contract_data for edge width
            if 'contract_data' in graph_data and i < len(graph_data['contract_data']):
                try:
                    contract_amount = float(graph_data['contract_data'].iloc[i]['montant'])
                except (KeyError, IndexError, ValueError):
                    # Fallback: try to get amount from preprocessed features
                    edge_features = graph_data['edge_features'][i]
                    if 'feature_columns' in graph_data:
                        amount_col_idx = None
                        for j, col in enumerate(graph_data['feature_columns']):
                            if 'montant' in col:
                                amount_col_idx = j
                                break
                        if amount_col_idx is not None:
                            contract_amount = float(edge_features[amount_col_idx])
                        else:
                            contract_amount = 100000  # Default fallback
                    else:
                        contract_amount = 100000  # Default fallback
            else:
                contract_amount = 100000  # Default fallback
            
            # Scale edge width based on contract amount
            edge_width = min(1 + contract_amount / 1e5, 5)  # Scale but cap at 5
            
            net.add_edge(
                int(edge[0]),  # Convert to Python int
                int(edge[1]),  # Convert to Python int
                width=edge_width,
                title=f"Amount: {contract_amount:,.2f}"
            )
        
        # Configure physics layout for initial spreading, then disable for static view
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.08,
                    "damping": 0.9,  // Higher damping for faster settling
                    "avoidOverlap": 1
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                    "enabled": true,
                    "iterations": 300,  // Lower for faster stop
                    "updateInterval": 25
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "hideEdgesOnDrag": true,
                "navigationButtons": true
            }
        }
        """)
        net.toggle_physics(False)
        
        # Add title
        net.set_title(title)
        
        # Save and open in browser from the data folder
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, 'procurement_graph.html')
        net.save_graph(output_path)
        webbrowser.open('file://' + output_path)


class GNNAnomalyDetector:
    """Graph Neural Network for anomaly detection."""
    
    def __init__(self, hidden_dim: int = 64, output_dim: int = 32,
                 num_layers: int = 3):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.model = None
        self.graph_tensor_train = None
        self.graph_tensor_test = None
        self.schema = None
        
    def create_tensorflow_graph(self, graph_data: Dict,
                                node_features_scaled: np.ndarray,
                                edge_features_scaled: np.ndarray
                                ) -> tfgnn.GraphTensor:
        """Create a TensorFlow GNN graph from our data."""
        logger.info("Creating TensorFlow GNN graph...")
        
        # Create the graph tensor with single node set
        graph_tensor = tfgnn.GraphTensor.from_pieces(
            node_sets={
                "entities": tfgnn.NodeSet.from_fields(
                    features={
                        "features": tf.constant(node_features_scaled,
                                                dtype=tf.float32),
                        "node_type": tf.constant(np.array(graph_data['node_types'], dtype=np.int32),
                                               dtype=tf.int32)
                    },
                    sizes=tf.constant([len(node_features_scaled)],
                                      dtype=tf.int32)
                )
            },
            edge_sets={
                "contracts": tfgnn.EdgeSet.from_fields(
                    features={
                        "features": tf.constant(edge_features_scaled,
                                                dtype=tf.float32)
                    },
                    sizes=tf.constant([len(edge_features_scaled)],
                                      dtype=tf.int32),
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("entities",
                                tf.constant(graph_data['edges'][:, 0].astype(np.int32),
                                            dtype=tf.int32)),
                        target=("entities",
                                tf.constant(graph_data['edges'][:, 1].astype(np.int32),
                                            dtype=tf.int32))
                    )
                )
            }
        )
        
        return graph_tensor
    
    def build_model(self, node_feature_dim: int,
                    edge_feature_dim: int,
                    l2_regularization: float = 5e-4,
                    dropout_rate: float = 0.3) -> tf.keras.Model:
        """Build the GNN model for anomaly detection with proper regularization."""
        logger.info("Building GNN model with node and edge anomaly detection...")
        
        # Helper function for regularized dense layers
        def dense_with_regularization(units, activation="relu"):
            """Dense layer with L2 regularization and dropout."""
            regularizer = tf.keras.regularizers.l2(l2_regularization)
            return tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units,
                    activation=activation,
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer),
                tf.keras.layers.Dropout(dropout_rate)
            ])
        
        # Create input spec with single node set using correct API
        input_spec = tfgnn.GraphTensorSpec.from_piece_specs(
            node_sets_spec={
                "entities": tfgnn.NodeSetSpec.from_field_specs(
                    features_spec={
                        "features": tf.TensorSpec(
                            shape=(None, node_feature_dim),
                            dtype=tf.float32),
                        "node_type": tf.TensorSpec(shape=(None,),
                                                   dtype=tf.int32)
                    },
                    sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32)
                )
            },
            edge_sets_spec={
                "contracts": tfgnn.EdgeSetSpec.from_field_specs(
                    features_spec={
                        "features": tf.TensorSpec(
                            shape=(None, edge_feature_dim),
                            dtype=tf.float32)
                    },
                    sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32),
                    adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                        source_node_set="entities",
                        target_node_set="entities"
                    )
                )
            }
        )
        
        # Input layer
        input_graph = tf.keras.layers.Input(type_spec=input_spec)
        graph = input_graph
        
        # Initialize hidden states for both nodes and edges
        def set_initial_node_state(node_set, *, node_set_name):
            return tf.keras.layers.Dense(self.hidden_dim)(node_set["features"])
            
        def set_initial_edge_state(edge_set, *, edge_set_name):
            return tf.keras.layers.Dense(self.hidden_dim)(edge_set["features"])
            
        graph = tfgnn.keras.layers.MapFeatures(
            node_sets_fn=set_initial_node_state,
            edge_sets_fn=set_initial_edge_state
        )(graph)
        
        # GNN message passing layers with regularization
        for i in range(self.num_layers):
            graph = tfgnn.keras.layers.GraphUpdate(
                node_sets={
                    "entities": tfgnn.keras.layers.NodeSetUpdate(
                        {"contracts": tfgnn.keras.layers.SimpleConv(
                            sender_edge_feature=tfgnn.HIDDEN_STATE,
                            message_fn=dense_with_regularization(self.hidden_dim),
                            reduce_type="sum",
                            receiver_tag=tfgnn.TARGET)},
                        tfgnn.keras.layers.NextStateFromConcat(
                            dense_with_regularization(self.hidden_dim)))}
            )(graph)
        
        # Extract final node and edge representations
        node_features = graph.node_sets["entities"][tfgnn.HIDDEN_STATE]
        edge_features = graph.edge_sets["contracts"][tfgnn.HIDDEN_STATE]
        
        # Create node embeddings with regularization
        node_embeddings = dense_with_regularization(
            self.output_dim, activation="tanh")(node_features)
        node_embeddings = tf.keras.layers.Lambda(
            lambda x: x, name="node_embeddings")(node_embeddings)
        
        # Create edge embeddings with regularization
        edge_embeddings = dense_with_regularization(
            self.output_dim, activation="tanh")(edge_features)
        edge_embeddings = tf.keras.layers.Lambda(
            lambda x: x, name="edge_embeddings")(edge_embeddings)
        
        # Node reconstruction pathway for anomaly detection
        node_reconstructed = dense_with_regularization(
            self.hidden_dim, activation="relu")(node_embeddings)
        # Final reconstruction layer without dropout to preserve quality
        node_reconstructed = tf.keras.layers.Dense(
            node_feature_dim, 
            kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
            name="node_reconstructed")(node_reconstructed)
        
        # Edge reconstruction pathway for anomaly detection
        edge_reconstructed = dense_with_regularization(
            self.hidden_dim, activation="relu")(edge_embeddings)
        # Final reconstruction layer without dropout to preserve quality
        edge_reconstructed = tf.keras.layers.Dense(
            edge_feature_dim, 
            kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
            name="edge_reconstructed")(edge_reconstructed)
        
        model = tf.keras.Model(
            inputs=input_graph,
            outputs={
                'node_embeddings': node_embeddings,
                'edge_embeddings': edge_embeddings,
                'node_reconstructed': node_reconstructed,
                'edge_reconstructed': edge_reconstructed
            }
        )
        
        return model
    
    def train(self, graph_tensor: tfgnn.GraphTensor,
             epochs: int = 50) -> Dict:
        """Train the GNN model."""
        logger.info(f"Training GNN model for {epochs} epochs...")
        
        self.graph_tensor_train = graph_tensor
        
        # Get node and edge features for reconstruction targets
        node_target_features = graph_tensor.node_sets['entities']['features']
        edge_target_features = graph_tensor.edge_sets['contracts']['features']
        
        # Create dummy targets for embeddings (we focus on reconstruction loss)
        num_nodes = tf.shape(node_target_features)[0]
        num_edges = tf.shape(edge_target_features)[0]
        dummy_node_embeddings = tf.zeros((num_nodes, self.output_dim))
        dummy_edge_embeddings = tf.zeros((num_edges, self.output_dim))
        
        # Create a dataset from the single graph
        def data_generator():
            yield (graph_tensor, {
                'node_embeddings': dummy_node_embeddings,
                'edge_embeddings': dummy_edge_embeddings,
                'node_reconstructed': node_target_features,
                'edge_reconstructed': edge_target_features
            })
        
        # Create dataset with proper output signature
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                graph_tensor.spec,
                {
                    'node_embeddings': tf.TensorSpec(
                        shape=(None, self.output_dim), dtype=tf.float32),
                    'edge_embeddings': tf.TensorSpec(
                        shape=(None, self.output_dim), dtype=tf.float32),
                    'node_reconstructed': tf.TensorSpec(
                        shape=(None, node_target_features.shape[1]), 
                        dtype=tf.float32),
                    'edge_reconstructed': tf.TensorSpec(
                        shape=(None, edge_target_features.shape[1]), 
                        dtype=tf.float32)
                }
            )
        )
        
        # Repeat for multiple epochs
        dataset = dataset.repeat()
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'node_embeddings': tf.keras.losses.MeanSquaredError(),
                'edge_embeddings': tf.keras.losses.MeanSquaredError(),
                'node_reconstructed': tf.keras.losses.MeanSquaredError(),
                'edge_reconstructed': tf.keras.losses.MeanSquaredError()
            },
            loss_weights={'node_embeddings': 0.05, 'edge_embeddings': 0.05,
                         'node_reconstructed': 0.45, 'edge_reconstructed': 0.45}
        )
        
        # Train using model.fit
        history = self.model.fit(
            dataset,
            steps_per_epoch=1,  # One step per epoch since we have one graph
            epochs=epochs,
            verbose=1
        )
        
        return history.history
    
    def detect_anomalies(self, threshold_percentile: float = 95
                         ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Detect anomalies based on reconstruction error."""
        logger.info("Detecting node and edge anomalies...")
        
        if self.model is None or self.graph_tensor is None:
            raise ValueError("Model must be trained before detecting "
                             "anomalies")
        
        # Get predictions
        predictions = self.model.predict(self.graph_tensor)
        node_reconstructed = predictions['node_reconstructed']
        edge_reconstructed = predictions['edge_reconstructed']
        
        # Calculate reconstruction errors
        original_node_features = (self.graph_tensor.node_sets['entities']
                              ['features'].numpy())
        original_edge_features = (self.graph_tensor.edge_sets['contracts']
                              ['features'].numpy())
        
        node_reconstruction_error = np.mean((original_node_features - 
                                           node_reconstructed) ** 2, axis=1)
        edge_reconstruction_error = np.mean((original_edge_features - 
                                           edge_reconstructed) ** 2, axis=1)
        
        # Determine thresholds and anomalies
        node_threshold = np.percentile(node_reconstruction_error, 
                                      threshold_percentile)
        edge_threshold = np.percentile(edge_reconstruction_error, 
                                      threshold_percentile)
        node_anomalies = node_reconstruction_error > node_threshold
        edge_anomalies = edge_reconstruction_error > edge_threshold
        
        logger.info(f"Detected {np.sum(node_anomalies)} node anomalies "
                    f"({np.sum(node_anomalies)/len(node_anomalies)*100:.1f}%)")
        logger.info(f"Detected {np.sum(edge_anomalies)} edge anomalies "
                    f"({np.sum(edge_anomalies)/len(edge_anomalies)*100:.1f}%)")
        
        return (node_reconstruction_error, edge_reconstruction_error, 
                node_threshold, edge_threshold)


class AnomalyAnalyzer:
    """Analyze and visualize anomaly detection results."""
    
    def __init__(self):
        pass
    
    def create_results_dataframe(self, graph_data: Dict,
                                 node_reconstruction_error: np.ndarray,
                                 node_anomalies: np.ndarray) -> pd.DataFrame:
        """Create a comprehensive results DataFrame for nodes."""
        # Create basic results dataframe
        results = {
            'entity_name': graph_data['nodes'],
            'entity_type': ['Buyer' if t == 0 else 'Supplier'
                           for t in graph_data['node_types']],
            'node_reconstruction_error': node_reconstruction_error,
            'is_node_anomaly': node_anomalies
        }
        
        # Add node features if available (handle variable structure)
        node_features = graph_data['node_features']
        if node_features.shape[1] >= 3:
            # Extract basic statistics from the end of feature vector
            # These are added by the preprocessing function
            results.update({
                'num_contracts': node_features[:, -3],
                'num_partners': node_features[:, -2], 
                'contracts_per_partner': node_features[:, -1]
            })
            
            # If we have more features, add some aggregated measures
            if node_features.shape[1] > 3:
                results.update({
                    'mean_feature_value': np.mean(node_features[:, :-3], axis=1),
                    'std_feature_value': np.std(node_features[:, :-3], axis=1)
                })
        
        return pd.DataFrame(results).sort_values('node_reconstruction_error', 
                                                ascending=False)
    
    def create_edge_results_dataframe(self, graph_data: Dict,
                                     edge_reconstruction_error: np.ndarray,
                                     edge_anomalies: np.ndarray) -> pd.DataFrame:
        """Create a comprehensive results DataFrame for edges (contracts)."""
        contract_data = graph_data['contract_data']
        
        return pd.DataFrame({
            'contract_id': graph_data['contract_ids'],
            'buyer_name': contract_data['acheteur_id'].values,
            'supplier_name': contract_data['titulaire_id'].values,
            'amount': contract_data['montant'].values,
            'cpv_code': contract_data['codeCPV_3'].values,
            'procedure': contract_data['procedure'].values,
            'notification_date': contract_data['dateNotification'].values,
            'edge_reconstruction_error': edge_reconstruction_error,
            'is_edge_anomaly': edge_anomalies,
            'log_amount': graph_data['edge_features'][:, 0],
            'cpv_hash': graph_data['edge_features'][:, 1],
            'procedure_hash': graph_data['edge_features'][:, 2],
            'duration_months': graph_data['edge_features'][:, 3]
        }).sort_values('edge_reconstruction_error', ascending=False)
    
    def plot_results(self, results_df: pd.DataFrame,
                    node_reconstruction_error: np.ndarray,
                    edge_reconstruction_error: np.ndarray,
                    node_threshold: float, edge_threshold: float):
        """Create comprehensive visualization of results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Node reconstruction error distribution
        axes[0, 0].hist(node_reconstruction_error, bins=50, alpha=0.7,
                       color='skyblue')
        axes[0, 0].axvline(node_threshold, color='red', linestyle='--',
                          label=f'Threshold ({node_threshold:.4f})')
        axes[0, 0].set_xlabel('Node Reconstruction Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Node Reconstruction Errors')
        axes[0, 0].legend()
        
        # Plot 2: Edge reconstruction error distribution
        axes[0, 1].hist(edge_reconstruction_error, bins=50, alpha=0.7,
                       color='skyblue')
        axes[0, 1].axvline(edge_threshold, color='red', linestyle='--',
                          label=f'Threshold ({edge_threshold:.4f})')
        axes[0, 1].set_xlabel('Edge Reconstruction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Edge Reconstruction Errors')
        axes[0, 1].legend()
        
        # Plot 3: Error by entity type
        buyer_errors = node_reconstruction_error[
            results_df['entity_type'] == 'Buyer']
        supplier_errors = node_reconstruction_error[
            results_df['entity_type'] == 'Supplier']
        axes[0, 2].boxplot([buyer_errors, supplier_errors],
                          labels=['Buyers', 'Suppliers'])
        axes[0, 2].set_ylabel('Node Reconstruction Error')
        axes[0, 2].set_title('Node Reconstruction Error by Entity Type')
        
        # Plot 4: Scatter plot - contracts vs feature value
        normal_mask = ~results_df['is_node_anomaly']
        if 'num_contracts' in results_df.columns and 'mean_feature_value' in results_df.columns:
            y_col = 'mean_feature_value'
            y_label = 'Mean Feature Value'
        elif 'num_contracts' in results_df.columns:
            y_col = 'contracts_per_partner'
            y_label = 'Contracts per Partner'
        else:
            # Fallback: use reconstruction error
            y_col = 'node_reconstruction_error'
            y_label = 'Reconstruction Error'
            
        x_col = 'num_contracts' if 'num_contracts' in results_df.columns else 'node_reconstruction_error'
        x_label = 'Number of Contracts' if x_col == 'num_contracts' else 'Reconstruction Error'
        
        axes[1, 0].scatter(results_df[normal_mask][x_col],
                          results_df[normal_mask][y_col],
                          alpha=0.6, s=30, label='Normal', color='lightblue')
        axes[1, 0].scatter(results_df[~normal_mask][x_col],
                          results_df[~normal_mask][y_col],
                          alpha=0.8, s=60, label='Anomaly', color='red',
                          marker='x')
        axes[1, 0].set_xlabel(x_label)
        axes[1, 0].set_ylabel(y_label)
        axes[1, 0].set_title(f'{x_label} vs {y_label}')
        axes[1, 0].legend()
        
        # Plot 5: Partners vs contracts
        axes[1, 1].scatter(results_df[normal_mask]['num_partners'],
                          results_df[normal_mask]['num_contracts'],
                          alpha=0.6, s=30, label='Normal', color='lightblue')
        axes[1, 1].scatter(results_df[~normal_mask]['num_partners'],
                          results_df[~normal_mask]['num_contracts'],
                          alpha=0.8, s=60, label='Anomaly', color='red',
                          marker='x')
        axes[1, 1].set_xlabel('Number of Partners')
        axes[1, 1].set_ylabel('Number of Contracts')
        axes[1, 1].set_title('Partners vs Contracts')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def analyze_anomalous_communities(self, graph_data: Dict,
                                     results_df: pd.DataFrame,
                                     node_embeddings: np.ndarray,
                                     edge_embeddings: np.ndarray) -> Dict:
        """Analyze communities among anomalous entities."""
        logger.info("Analyzing anomalous communities...")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for i, (name, node_type, is_anom) in enumerate(zip(
            graph_data['nodes'], graph_data['node_types'],
            results_df['is_node_anomaly'])):
            G.add_node(i, name=name,
                      type='buyer' if node_type == 0 else 'supplier',
                      anomaly=is_anom)
        
        # Add edges
        for edge in graph_data['edges']:
            G.add_edge(edge[0], edge[1])
        
        # Extract anomalous subgraph
        anomalous_nodes = results_df[results_df['is_node_anomaly']].index.tolist()
        anomalous_neighbors = set(anomalous_nodes)
        
        for node in anomalous_nodes:
            anomalous_neighbors.update(G.neighbors(node))
        
        subgraph = G.subgraph(anomalous_neighbors)
        
        # Community detection using embeddings
        communities_info = {}
        if len(anomalous_neighbors) > 3:
            anomalous_node_embeddings = node_embeddings[list(anomalous_neighbors)]
            anomalous_edge_embeddings = edge_embeddings[list(anomalous_neighbors)]
            
            # Use DBSCAN for community detection
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            node_communities = dbscan.fit_predict(anomalous_node_embeddings)
            edge_communities = dbscan.fit_predict(anomalous_edge_embeddings)
            
            # Add community information to nodes
            for node, community in zip(anomalous_neighbors, node_communities):
                G.nodes[node]['community'] = int(community)
            
            communities_info = {
                'num_communities': len(set(node_communities)) - \
                    (1 if -1 in node_communities else 0),
                'noise_points': np.sum(node_communities == -1),
                'subgraph_size': subgraph.number_of_nodes(),
                'subgraph_edges': subgraph.number_of_edges()
            }
            
            # Create interactive visualization
            self.visualize_graph(G, anomalous_nodes, communities_info)
        
        return communities_info

    def visualize_graph(self, G: nx.Graph, anomalous_nodes: List[int],
                       communities_info: Dict):
        """Create an interactive visualization of the graph."""
        try:
            from pyvis.network import Network
            import webbrowser
            from tempfile import NamedTemporaryFile
            
            # Create a new network
            net = Network(height="750px", width="100%", bgcolor="#ffffff",
                        font_color="black")
            
            # Add nodes
            for node in G.nodes():
                node_data = G.nodes[node]
                color = "red" if node in anomalous_nodes else "blue"
                shape = "diamond" if node_data['type'] == 'buyer' else "dot"
                
                # Add node with properties
                net.add_node(
                    node,
                    label=node_data['name'],
                    title=f"Type: {node_data['type']}\n"
                          f"Anomaly: {node_data['anomaly']}\n"
                          f"Community: {node_data.get('community', 'N/A')}",
                    color=color,
                    shape=shape
                )
            
            # Add edges
            for edge in G.edges():
                net.add_edge(edge[0], edge[1])
            
            # Configure physics layout
            net.set_options("""
            {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 100,
                        "springConstant": 0.08
                    },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "timestep": 0.35,
                    "stabilization": {
                        "enabled": true,
                        "iterations": 1000
                    }
                }
            }
            """)
            
            # Save and open in browser
            with NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                net.save_graph(tmp.name)
                webbrowser.open('file://' + tmp.name)
                
        except ImportError:
            logger.warning("pyvis not installed. Install with: pip install pyvis")
            # Fallback to static visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=[n for n in G.nodes()
                                         if n in anomalous_nodes],
                                 node_color='red',
                                 node_size=100,
                                 alpha=0.6)
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=[n for n in G.nodes()
                                         if n not in anomalous_nodes],
                                 node_color='blue',
                                 node_size=50,
                                 alpha=0.4)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, alpha=0.2)
            
            # Add labels
            labels = {n: G.nodes[n]['name'] for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
            
            plt.title("Anomalous Entities Network\n"
                     f"Communities: {communities_info.get('num_communities', 0)}")
            plt.axis('off')
            plt.show()


def main():
    """Main execution function."""
    logger.info("Starting GNN Anomaly Detection Pipeline...")
    
    # Configuration
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                            'data')
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                             'models', 'anomalies')
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Initialize components
    graph_builder = ProcurementGraphBuilder()
    gnn_detector = GNNAnomalyDetector(hidden_dim=64, output_dim=32,
                                     num_layers=3)
    analyzer = AnomalyAnalyzer()
    
    try:
        # Load and preprocess data
        X = graph_builder.load_data(DATA_PATH)
        X_train_preproc, X_test_preproc, X_train, X_test = graph_builder.preprocess_data(X)
        
        # Create graph from TRAINING data for model training
        train_graph_data = graph_builder.create_graph(X_train_preproc, X_train)
        
        # Scale features using training data
        train_node_features = train_graph_data['node_features']
        train_edge_features = train_graph_data['edge_features']
        
        # Fit scalers on training data
        train_node_features_scaled = graph_builder.node_scaler.fit_transform(
            train_node_features)
        train_edge_features_scaled = graph_builder.edge_scaler.fit_transform(
            train_edge_features)
        
        # Create TensorFlow graph for training
        train_graph_tensor = gnn_detector.create_tensorflow_graph(
            train_graph_data, train_node_features_scaled, train_edge_features_scaled)
        
        # Build model
        gnn_detector.model = gnn_detector.build_model(
            train_node_features_scaled.shape[1], train_edge_features_scaled.shape[1])
        
        # Train model on training data
        history = gnn_detector.train(train_graph_tensor, epochs=50)
        
        # Create graph from TEST data for anomaly detection
        test_graph_data = graph_builder.create_graph(X_test_preproc, X_test)
        
        # Scale test features using training data scalers (transform only, no fit)
        test_node_features = test_graph_data['node_features']
        test_edge_features = test_graph_data['edge_features']
        test_node_features_scaled = graph_builder.node_scaler.transform(
            test_node_features)
        test_edge_features_scaled = graph_builder.edge_scaler.transform(
            test_edge_features)
        
        # Create TensorFlow graph for testing
        test_graph_tensor = gnn_detector.create_tensorflow_graph(
            test_graph_data, test_node_features_scaled, test_edge_features_scaled)
        
        # Store test graph for anomaly detection
        gnn_detector.graph_tensor = test_graph_tensor
        
        # Detect anomalies on TEST data
        (node_reconstruction_error, edge_reconstruction_error, 
         node_threshold, edge_threshold) = gnn_detector.detect_anomalies()
        
        # Calculate anomaly masks
        node_anomalies = node_reconstruction_error > node_threshold
        edge_anomalies = edge_reconstruction_error > edge_threshold
        
        # OPTIONAL: Also evaluate on training data for comparison
        logger.info("Evaluating model on training data for comparison...")
        gnn_detector.graph_tensor = train_graph_tensor
        (train_node_error, train_edge_error, _, _) = gnn_detector.detect_anomalies(
            threshold_percentile=95)
        train_node_anomalies = train_node_error > node_threshold
        train_edge_anomalies = train_edge_error > edge_threshold
        
        print(f"Training data anomalies: {np.sum(train_node_anomalies)} nodes "
              f"({np.sum(train_node_anomalies)/len(train_node_anomalies)*100:.1f}%), "
              f"{np.sum(train_edge_anomalies)} edges "
              f"({np.sum(train_edge_anomalies)/len(train_edge_anomalies)*100:.1f}%)")
        
        # Reset to test data for final analysis
        gnn_detector.graph_tensor = test_graph_tensor
        
        # Get embeddings for analysis from TEST data
        predictions = gnn_detector.model.predict(test_graph_tensor)
        node_embeddings = predictions['node_embeddings']
        edge_embeddings = predictions['edge_embeddings']
        
        # Create results DataFrames using TEST data
        results_df = analyzer.create_results_dataframe(
            test_graph_data, node_reconstruction_error, node_anomalies)
        
        edge_results_df = analyzer.create_edge_results_dataframe(
            test_graph_data, edge_reconstruction_error, edge_anomalies)
        
        # Analyze communities
        communities_info = analyzer.analyze_anomalous_communities(
            test_graph_data, results_df, node_embeddings, edge_embeddings)
        
        # Visualize results
        analyzer.plot_results(results_df, node_reconstruction_error, 
                             edge_reconstruction_error,
                             node_threshold, edge_threshold)
        
        # Save results
        gnn_detector.model.save(os.path.join(MODEL_PATH, 'gnn_anomaly_model'))
        results_df.to_csv(os.path.join(MODEL_PATH, 'gnn_node_anomaly_results.csv'),
                         index=False)
        edge_results_df.to_csv(os.path.join(MODEL_PATH, 'gnn_edge_anomaly_results.csv'),
                              index=False)
        np.save(os.path.join(MODEL_PATH, 'gnn_node_embeddings.npy'),
               node_embeddings)
        np.save(os.path.join(MODEL_PATH, 'gnn_edge_embeddings.npy'),
               edge_embeddings)
        
        # Print summary
        print("\n" + "="*60)
        print("GNN ANOMALY DETECTION SUMMARY")
        print("="*60)
        print(f"Training entities: {len(train_graph_data['nodes'])}")
        print(f"- Training buyers: {np.sum(train_graph_data['node_types'] == 0)}")
        print(f"- Training suppliers: {np.sum(train_graph_data['node_types'] == 1)}")
        print(f"Training contracts: {len(train_graph_data['edges'])}")
        print(f"\nTest entities analyzed: {len(test_graph_data['nodes'])}")
        print(f"- Test buyers: {np.sum(test_graph_data['node_types'] == 0)}")
        print(f"- Test suppliers: {np.sum(test_graph_data['node_types'] == 1)}")
        print(f"Test contracts analyzed: {len(test_graph_data['edges'])}")
        
        print(f"\nNode anomalies detected: {np.sum(node_anomalies)} "
              f"({np.sum(node_anomalies)/len(node_anomalies)*100:.1f}%)")
        print(f"Edge anomalies detected: {np.sum(edge_anomalies)} "
              f"({np.sum(edge_anomalies)/len(edge_anomalies)*100:.1f}%)")
        
        anomalous_buyers = results_df[
            (results_df['is_node_anomaly']) &
            (results_df['entity_type'] == 'Buyer')]
        anomalous_suppliers = results_df[
            (results_df['is_node_anomaly']) &
            (results_df['entity_type'] == 'Supplier')]
        
        print(f"- Anomalous buyers: {len(anomalous_buyers)}")
        print(f"- Anomalous suppliers: {len(anomalous_suppliers)}")
        print(f"\nModel performance:")
        print(f"- Final node reconstruction loss: "
              f"{history['node_reconstructed_loss'][-1]:.4f}")
        print(f"- Final edge reconstruction loss: "
              f"{history['edge_reconstructed_loss'][-1]:.4f}")
        print(f"- Node anomaly threshold: {node_threshold:.4f}")
        print(f"- Edge anomaly threshold: {edge_threshold:.4f}")
        
        if communities_info:
            print(f"\nCommunity analysis:")
            print(f"- Communities detected: "
                  f"{communities_info['num_communities']}")
            print(f"- Anomalous subgraph size: "
                  f"{communities_info['subgraph_size']} nodes")
        
        print(f"\nTop 5 most anomalous entities:")
        top_anomalies = results_df.head(5)[
            ['entity_name', 'entity_type', 'node_reconstruction_error']]
        print(top_anomalies.to_string(index=False))
        
        print(f"\nTop 5 most anomalous contracts:")
        top_edge_anomalies = edge_results_df.head(5)[
            ['buyer_name', 'supplier_name', 'amount', 'edge_reconstruction_error']]
        print(top_edge_anomalies.to_string(index=False))
        
        # Visualize the test procurement graph
        graph_builder.visualize_procurement_graph(test_graph_data, 
            "French Public Procurement Network (Test Set)")
        
        logger.info("GNN Anomaly Detection completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in GNN pipeline: {str(e)}")
        raise


def build_model_standalone(node_feature_dim: int, edge_feature_dim: int,
                          hidden_dim: int = 64, output_dim: int = 32,
                          num_layers: int = 3, l2_regularization: float = 5e-4,
                          dropout_rate: float = 0.3) -> tf.keras.Model:
    """Standalone version of build_model for direct use in notebooks."""
    logger.info("Building GNN model...")
    
    # Helper function for regularized dense layers
    def dense_with_regularization(units, activation="relu"):
        """Dense layer with L2 regularization and dropout."""
        regularizer = tf.keras.regularizers.l2(l2_regularization)
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer),
            tf.keras.layers.Dropout(dropout_rate)
        ])
    
    # Create input spec for batched graphs
    input_spec = tfgnn.GraphTensorSpec.from_piece_specs(
        node_sets_spec={
            "entities": tfgnn.NodeSetSpec.from_field_specs(
                features_spec={
                    "features": tf.TensorSpec(
                        shape=(None, node_feature_dim),
                        dtype=tf.float32),
                    "node_type": tf.TensorSpec(shape=(None,),
                                               dtype=tf.int32)
                },
                sizes_spec=tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        },
        edge_sets_spec={
            "contracts": tfgnn.EdgeSetSpec.from_field_specs(
                features_spec={
                    "features": tf.TensorSpec(
                        shape=(None, edge_feature_dim),
                        dtype=tf.float32)
                },
                sizes_spec=tf.TensorSpec(shape=(None,), dtype=tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                    source_node_set="entities",
                    target_node_set="entities"
                )
            )
        }
    )
    
    # Input layer for batched GraphTensor
    input_graph = tf.keras.layers.Input(type_spec=input_spec)
    
    # IMPORTANT: Merge batch to components - this is the key for proper batching
    graph = input_graph.merge_batch_to_components()
    
    # Initialize hidden states for both nodes and edges
    def set_initial_node_state(node_set, *, node_set_name):
        return tf.keras.layers.Dense(hidden_dim)(node_set["features"])
        
    def set_initial_edge_state(edge_set, *, edge_set_name):
        return tf.keras.layers.Dense(hidden_dim)(edge_set["features"])
        
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=set_initial_node_state,
        edge_sets_fn=set_initial_edge_state
    )(graph)
    
    # GNN message passing layers with regularization
    for i in range(num_layers):
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "entities": tfgnn.keras.layers.NodeSetUpdate(
                    {"contracts": tfgnn.keras.layers.SimpleConv(
                        sender_edge_feature=tfgnn.HIDDEN_STATE,
                        message_fn=dense_with_regularization(hidden_dim),
                        reduce_type="sum",
                        receiver_tag=tfgnn.TARGET)},
                    tfgnn.keras.layers.NextStateFromConcat(
                        dense_with_regularization(hidden_dim)))}
        )(graph)
    
    # Extract final node representations using context pooling
    # Pool node states back to the original batch structure
    node_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", node_set_name="entities")(graph)
    
    # Since we're doing reconstruction, we need to unpool back to nodes
    # For simplicity, we'll use a different approach - extract node features directly
    # Note: This requires the graph to maintain node indexing
    node_features = graph.node_sets["entities"][tfgnn.HIDDEN_STATE]
    
    # Create embeddings with regularization
    embeddings = dense_with_regularization(
        output_dim, activation="tanh")(node_features)
    embeddings = tf.keras.layers.Lambda(
        lambda x: x, name="embeddings")(embeddings)
    
    # Reconstruction pathway for anomaly detection
    reconstructed = dense_with_regularization(
        hidden_dim, activation="relu")(embeddings)
    # Final reconstruction layer without dropout to preserve quality
    reconstructed = tf.keras.layers.Dense(
        node_feature_dim, 
        kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
        name="reconstructed")(reconstructed)
    
    model = tf.keras.Model(
        inputs=input_graph,
        outputs={
            'embeddings': embeddings,
            'reconstructed': reconstructed
        }
    )
    
    return model


def train_model_standalone(model: tf.keras.Model, 
                          graph_tensor: tfgnn.GraphTensor,
                          output_dim: int = 32,
                          epochs: int = 50) -> Dict:
    """Standalone training function for direct use in notebooks."""
    logger.info(f"Training GNN model for {epochs} epochs...")
    
    # Get node features for reconstruction target
    target_features = graph_tensor.node_sets['entities']['features']
    num_nodes = tf.shape(target_features)[0]
    dummy_embeddings = tf.zeros((num_nodes, output_dim))
    
    # Create a dataset from the single graph
    # The key is to create a dataset that yields (graph, targets) pairs
    def data_generator():
        yield (graph_tensor, {
            'embeddings': dummy_embeddings,
            'reconstructed': target_features
        })
    
    # Create dataset with proper output signature
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            graph_tensor.spec,
            {
                'embeddings': tf.TensorSpec(shape=(None, output_dim), dtype=tf.float32),
                'reconstructed': tf.TensorSpec(shape=(None, target_features.shape[1]), dtype=tf.float32)
            }
        )
    )
    
    # Batch the dataset (batch size = 1 for single graph)
    dataset = dataset.batch(1)
    
    # Repeat for multiple epochs (model.fit will handle epochs, but this ensures data availability)
    dataset = dataset.repeat()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'embeddings': tf.keras.losses.MeanSquaredError(),
            'reconstructed': tf.keras.losses.MeanSquaredError()
        },
        loss_weights={'embeddings': 0.1, 'reconstructed': 0.9}
    )
    
    # Train using model.fit
    history = model.fit(
        dataset,
        steps_per_epoch=1,  # One step per epoch since we have one graph
        epochs=epochs,
        verbose=1
    )
    
    return history.history


if __name__ == "__main__":
    main()


# USAGE EXAMPLE FOR NOTEBOOK:
"""
# Correct sequence for using this module in a notebook:

# 1. Initialize components
graph_builder = ProcurementGraphBuilder()
gnn_detector = GNNAnomalyDetector(hidden_dim=64, output_dim=32, num_layers=3)

# 2. Load and preprocess data
X = graph_builder.load_data(DATA_PATH)  # Replace DATA_PATH with your path
X_train_preproc, X_test_preproc, X_train, X_test = graph_builder.preprocess_data(X)

# 3. Create graph from preprocessed data - THIS CREATES THE NODE AND EDGE FEATURES
graph_data = graph_builder.create_graph(X_train_preproc, X_train)

# 4. NOW you can access the features from graph_data
node_features = graph_data['node_features']
edge_features = graph_data['edge_features']

# 5. Scale the features
node_features_scaled = graph_builder.node_scaler.fit_transform(node_features)
edge_features_scaled = graph_builder.edge_scaler.fit_transform(edge_features)

# 6. Create TensorFlow graph
graph_tensor = gnn_detector.create_tensorflow_graph(
    graph_data, node_features_scaled, edge_features_scaled)

# 7. Build and train model
gnn_detector.model = gnn_detector.build_model(
    node_features_scaled.shape[1], edge_features_scaled.shape[1])

# 8. Train
history = gnn_detector.train(graph_tensor, epochs=50)

# 9. Detect anomalies
(node_reconstruction_error, edge_reconstruction_error, 
 node_threshold, edge_threshold) = gnn_detector.detect_anomalies()
""" 