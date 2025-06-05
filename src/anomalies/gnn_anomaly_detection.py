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
import sqlite3
import logging
from typing import Dict, Tuple, List

import tensorflow as tf
import tensorflow_gnn as tfgnn

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcurementGraphBuilder:
    """Build graph structures from procurement data."""
    
    def __init__(self):
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess procurement data."""
        logger.info("Loading procurement data...")
        
        # Try loading from CSV first, then SQLite if CSV doesn't exist
        csv_path = os.path.join(data_path, 'data_cpv.csv')
        sqlite_path = os.path.join(data_path, 'datalab.sqlite')
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} records from CSV")
        elif os.path.exists(sqlite_path):
            conn = sqlite3.connect(sqlite_path)
            query = """
            SELECT 
                acheteur_nom,
                titulaire_nom,
                montant,
                cpv_division_libelle,
                procedure,
                date_notification,
                departement_code,
                nature,
                dureeMois,
                formePrix
            FROM marches_publics 
            WHERE montant IS NOT NULL 
            AND acheteur_nom IS NOT NULL 
            AND titulaire_nom IS NOT NULL
            LIMIT 10000
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(f"Loaded {len(df)} records from SQLite")
        else:
            raise FileNotFoundError(f"No data found in {data_path}")
            
        return self._preprocess_data(df)
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for graph construction."""
        logger.info("Preprocessing data...")
        
        # Basic cleaning
        df = df.dropna(subset=['montant', 'acheteur_nom', 'titulaire_nom'])
        
        # Filter out extreme outliers in montant
        Q1 = df['montant'].quantile(0.25)
        Q3 = df['montant'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df['montant'] >= lower_bound) &
                (df['montant'] <= upper_bound)]
        
        # Keep only entities with minimum activity
        buyer_counts = df['acheteur_nom'].value_counts()
        supplier_counts = df['titulaire_nom'].value_counts()
        
        active_buyers = buyer_counts[buyer_counts >= 2].index
        active_suppliers = supplier_counts[supplier_counts >= 2].index
        
        df = df[df['acheteur_nom'].isin(active_buyers) &
                df['titulaire_nom'].isin(active_suppliers)]
        
        logger.info(f"After preprocessing: {len(df)} contracts, "
                    f"{df['acheteur_nom'].nunique()} buyers, "
                    f"{df['titulaire_nom'].nunique()} suppliers")
        
        return df
    
    def create_graph(self, df: pd.DataFrame) -> Dict:
        """Transform procurement data into a graph structure."""
        logger.info("Creating graph structure...")
        
        # Create unique identifiers for buyers and suppliers
        buyers = df['acheteur_nom'].unique()
        suppliers = df['titulaire_nom'].unique()
        
        # Create node mappings
        buyer_to_id = {buyer: i for i, buyer in enumerate(buyers)}
        supplier_to_id = {supplier: i + len(buyers)
                          for i, supplier in enumerate(suppliers)}
        
        # Combine all nodes
        all_nodes = list(buyers) + list(suppliers)
        
        # Create edges (contracts) and edge features
        edges = []
        edge_features = []
        
        for _, row in df.iterrows():
            buyer_id = buyer_to_id[row['acheteur_nom']]
            supplier_id = supplier_to_id[row['titulaire_nom']]
            
            edges.append([buyer_id, supplier_id])
            
            # Edge features: contract information
            cpv_hash = hash(str(row.get('cpv_division_libelle', ''))) % 1000
            proc_hash = hash(str(row.get('procedure', ''))) % 100
            dept_code = int(row.get('departement_code', 0)) \
                if pd.notna(row.get('departement_code')) else 0
            duree = row.get('dureeMois', 0) \
                if pd.notna(row.get('dureeMois')) else 0
            
            edge_features.append([
                np.log1p(row['montant']),  # Log-transformed amount
                cpv_hash,  # CPV category
                proc_hash,  # Procedure type
                dept_code,
                duree
            ])
        
        # Create node features
        node_features = []
        node_types = []  # 0 for buyers, 1 for suppliers
        
        # Buyer features
        for buyer in buyers:
            buyer_data = df[df['acheteur_nom'] == buyer]
            features = self._calculate_node_features(buyer_data,
                                                   'titulaire_nom')
            node_features.append(features)
            node_types.append(0)  # Buyer
        
        # Supplier features
        for supplier in suppliers:
            supplier_data = df[df['titulaire_nom'] == supplier]
            features = self._calculate_node_features(supplier_data,
                                                   'acheteur_nom')
            node_features.append(features)
            node_types.append(1)  # Supplier
        
        return {
            'nodes': all_nodes,
            'edges': np.array(edges),
            'node_features': np.array(node_features, dtype=np.float32),
            'edge_features': np.array(edge_features, dtype=np.float32),
            'node_types': np.array(node_types),
            'buyer_to_id': buyer_to_id,
            'supplier_to_id': supplier_to_id
        }
    
    def _calculate_node_features(self, entity_data: pd.DataFrame,
                               partner_col: str) -> List[float]:
        """Calculate features for a node (buyer or supplier)."""
        n_partners = entity_data[partner_col].nunique()
        contracts_per_partner = len(entity_data) / n_partners \
            if n_partners > 0 else 0
            
        # Calculate contracts per partner distribution
        partner_contract_counts = entity_data[partner_col].value_counts()
        max_contracts_per_partner = (partner_contract_counts.max() 
                                     if not partner_contract_counts.empty 
                                     else 0)
        min_contracts_per_partner = (partner_contract_counts.min() 
                                     if not partner_contract_counts.empty 
                                     else 0)
        std_contracts_per_partner = (partner_contract_counts.std() 
                                     if len(partner_contract_counts) > 1 
                                     else 0)
            
        features = [
            len(entity_data),  # Number of contracts
            entity_data['montant'].sum(),  # Total amount
            entity_data['montant'].mean(),  # Average amount
            entity_data['montant'].std() if len(entity_data) > 1 else 0,
            n_partners,  # Number of unique partners
            entity_data['montant'].max(),  # Maximum contract amount
            entity_data['montant'].min(),  # Minimum contract amount
            contracts_per_partner,  # Average contracts per partner
            max_contracts_per_partner,  # Max contracts with single partner
            min_contracts_per_partner,  # Min contracts with single partner
            std_contracts_per_partner  # Std dev of contracts per partner
        ]
        return features
    
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
            
            # Calculate node size based on number of contracts
            num_contracts = int(graph_data['node_features'][i][0])
            node_size = min(50 + num_contracts * 2, 100)  # Scale size but cap it
            
            # Calculate node color based on total amount
            total_amount = float(graph_data['node_features'][i][1])
            # Normalize amount to a color scale (blue to red)
            amount_ratio = min(total_amount / 1e6, 1.0)  # Cap at 1M
            color = f"rgb({int(255 * amount_ratio)}, 0, {int(255 * (1 - amount_ratio))})"
            
            # Add node with properties
            net.add_node(
                int(i),  # Convert to Python int
                label=str(name),  # Convert to Python string
                title=f"Type: {'Buyer' if node_type == 0 else 'Supplier'}\n"
                      f"Contracts: {num_contracts}\n"
                      f"Total Amount: {total_amount:,.2f}\n"
                      f"Avg Amount: {float(graph_data['node_features'][i][2]):,.2f}\n"
                      f"Partners: {int(graph_data['node_features'][i][4])}",
                color=color,
                size=node_size,
                shape="diamond" if node_type == 0 else "dot"
            )
        
        # Add edges with weights based on contract amounts
        for i, edge in enumerate(graph_data['edges']):
            # Get edge features
            edge_features = graph_data['edge_features'][i]
            contract_amount = float(np.exp(edge_features[0]))  # Convert back from log
            
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
        self.graph_tensor = None
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
                        "node_type": tf.constant(graph_data['node_types'],
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
                                tf.constant(graph_data['edges'][:, 0],
                                            dtype=tf.int32)),
                        target=("entities",
                                tf.constant(graph_data['edges'][:, 1],
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
        
        # Extract final node representations
        node_features = graph.node_sets["entities"][tfgnn.HIDDEN_STATE]
        
        # Create embeddings with regularization
        embeddings = dense_with_regularization(
            self.output_dim, activation="tanh")(node_features)
        embeddings = tf.keras.layers.Lambda(
            lambda x: x, name="embeddings")(embeddings)
        
        # Reconstruction pathway for anomaly detection
        reconstructed = dense_with_regularization(
            self.hidden_dim, activation="relu")(embeddings)
        # Final reconstruction layer without dropout to preserve reconstruction quality
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
    
    def train(self, graph_tensor: tfgnn.GraphTensor,
             epochs: int = 50) -> Dict:
        """Train the GNN model."""
        logger.info(f"Training GNN model for {epochs} epochs...")
        
        self.graph_tensor = graph_tensor
        
        # Get node features for reconstruction target
        target_features = graph_tensor.node_sets['entities']['features']
        
        # Create dummy target for embeddings (we focus on reconstruction loss)
        num_nodes = tf.shape(target_features)[0]
        dummy_embeddings = tf.zeros((num_nodes, self.output_dim))
        
        # Add batch dimension to targets
        target_features_batched = tf.expand_dims(target_features, 0)
        dummy_embeddings_batched = tf.expand_dims(dummy_embeddings, 0)
        
        # Add batch dimension to graph tensor properly
        batched_graph = tfgnn.GraphTensor.from_pieces(
            node_sets={
                "entities": tfgnn.NodeSet.from_fields(
                    features={
                        "features": tf.expand_dims(
                            graph_tensor.node_sets["entities"]["features"], 0),
                        "node_type": tf.expand_dims(
                            graph_tensor.node_sets["entities"]["node_type"], 0)
                    },
                    sizes=graph_tensor.node_sets["entities"].sizes
                )
            },
            edge_sets={
                "contracts": tfgnn.EdgeSet.from_fields(
                    features={
                        "features": tf.expand_dims(
                            graph_tensor.edge_sets["contracts"]["features"], 0)
                    },
                    sizes=graph_tensor.edge_sets["contracts"].sizes,
                    adjacency=graph_tensor.edge_sets["contracts"].adjacency
                )
            }
        )
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'embeddings': tf.keras.losses.MeanSquaredError(),
                'reconstructed': tf.keras.losses.MeanSquaredError()
            },
            loss_weights={'embeddings': 0.1, 'reconstructed': 0.9}
        )
        
        # Train
        history = self.model.fit(
            batched_graph,
            {'embeddings': dummy_embeddings_batched,
             'reconstructed': target_features_batched},
            epochs=epochs,
            verbose=1
        )
        
        return history.history
    
    def detect_anomalies(self, threshold_percentile: float = 95
                         ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Detect anomalies based on reconstruction error."""
        logger.info("Detecting anomalies...")
        
        if self.model is None or self.graph_tensor is None:
            raise ValueError("Model must be trained before detecting "
                             "anomalies")
        
        # Get predictions
        predictions = self.model.predict(self.graph_tensor)
        reconstructed = predictions['reconstructed']
        
        # Calculate reconstruction error
        original_features = (self.graph_tensor.node_sets['entities']
                              ['features'].numpy())
        
        reconstruction_error = np.mean((original_features - reconstructed) ** 2,
                                       axis=1)
        
        # Determine threshold and anomalies
        threshold = np.percentile(reconstruction_error, threshold_percentile)
        anomalies = reconstruction_error > threshold
        
        logger.info(f"Detected {np.sum(anomalies)} anomalies "
                    f"({np.sum(anomalies)/len(anomalies)*100:.1f}%)")
        
        return reconstruction_error, anomalies, threshold


class AnomalyAnalyzer:
    """Analyze and visualize anomaly detection results."""
    
    def __init__(self):
        pass
    
    def create_results_dataframe(self, graph_data: Dict,
                                reconstruction_error: np.ndarray,
                                anomalies: np.ndarray) -> pd.DataFrame:
        """Create a comprehensive results DataFrame."""
        return pd.DataFrame({
            'entity_name': graph_data['nodes'],
            'entity_type': ['Buyer' if t == 0 else 'Supplier'
                           for t in graph_data['node_types']],
            'reconstruction_error': reconstruction_error,
            'is_anomaly': anomalies,
            'num_contracts': graph_data['node_features'][:, 0],
            'total_amount': graph_data['node_features'][:, 1],
            'avg_amount': graph_data['node_features'][:, 2],
            'amount_std': graph_data['node_features'][:, 3],
            'num_partners': graph_data['node_features'][:, 4],
            'max_amount': graph_data['node_features'][:, 5],
            'min_amount': graph_data['node_features'][:, 6],
            'contracts_per_partner': graph_data['node_features'][:, 7],
            'max_contracts_per_partner': graph_data['node_features'][:, 8],
            'min_contracts_per_partner': graph_data['node_features'][:, 9],
            'std_contracts_per_partner': graph_data['node_features'][:, 10]
        }).sort_values('reconstruction_error', ascending=False)
    
    def plot_results(self, results_df: pd.DataFrame,
                    reconstruction_error: np.ndarray,
                    threshold: float, embeddings: np.ndarray = None):
        """Create comprehensive visualization of results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Reconstruction error distribution
        axes[0, 0].hist(reconstruction_error, bins=50, alpha=0.7,
                       color='skyblue')
        axes[0, 0].axvline(threshold, color='red', linestyle='--',
                          label=f'Threshold ({threshold:.4f})')
        axes[0, 0].set_xlabel('Reconstruction Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Reconstruction Errors')
        axes[0, 0].legend()
        
        # Plot 2: Error by entity type
        buyer_errors = reconstruction_error[
            results_df['entity_type'] == 'Buyer']
        supplier_errors = reconstruction_error[
            results_df['entity_type'] == 'Supplier']
        axes[0, 1].boxplot([buyer_errors, supplier_errors],
                          labels=['Buyers', 'Suppliers'])
        axes[0, 1].set_ylabel('Reconstruction Error')
        axes[0, 1].set_title('Reconstruction Error by Entity Type')
        
        # Plot 3: Anomalies by type
        anomaly_counts = results_df[results_df['is_anomaly']][
            'entity_type'].value_counts()
        if len(anomaly_counts) > 0:
            axes[0, 2].bar(anomaly_counts.index, anomaly_counts.values,
                          color=['lightcoral', 'lightgreen'])
        axes[0, 2].set_title('Number of Anomalies by Entity Type')
        axes[0, 2].set_ylabel('Count')
        
        # Plot 4: Scatter plot - contracts vs amount
        normal_mask = ~results_df['is_anomaly']
        axes[1, 0].scatter(results_df[normal_mask]['num_contracts'],
                          results_df[normal_mask]['total_amount'],
                          alpha=0.6, s=30, label='Normal', color='lightblue')
        axes[1, 0].scatter(results_df[~normal_mask]['num_contracts'],
                          results_df[~normal_mask]['total_amount'],
                          alpha=0.8, s=60, label='Anomaly', color='red',
                          marker='x')
        axes[1, 0].set_xlabel('Number of Contracts')
        axes[1, 0].set_ylabel('Total Amount')
        axes[1, 0].set_title('Contracts vs Total Amount')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        
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
        
        # Plot 6: Embeddings visualization (if provided)
        if embeddings is not None:
            # Reduce dimensionality for visualization
            if embeddings.shape[1] > 2:
                pca = PCA(n_components=min(50, embeddings.shape[1]))
                embeddings_pca = pca.fit_transform(embeddings)
                tsne = TSNE(n_components=2, random_state=42,
                           perplexity=min(30, len(embeddings_pca)//4))
                embeddings_2d = tsne.fit_transform(embeddings_pca)
            else:
                embeddings_2d = embeddings
            
            normal_mask_emb = ~results_df['is_anomaly'].values
            axes[1, 2].scatter(embeddings_2d[normal_mask_emb, 0],
                              embeddings_2d[normal_mask_emb, 1],
                              alpha=0.6, s=30, label='Normal',
                              color='lightblue')
            axes[1, 2].scatter(embeddings_2d[~normal_mask_emb, 0],
                              embeddings_2d[~normal_mask_emb, 1],
                              alpha=0.8, s=60, label='Anomaly', color='red',
                              marker='x')
            axes[1, 2].set_title('Node Embeddings (t-SNE)')
            axes[1, 2].set_xlabel('t-SNE 1')
            axes[1, 2].set_ylabel('t-SNE 2')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def analyze_anomalous_communities(self, graph_data: Dict,
                                     results_df: pd.DataFrame,
                                     embeddings: np.ndarray) -> Dict:
        """Analyze communities among anomalous entities."""
        logger.info("Analyzing anomalous communities...")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for i, (name, node_type, is_anom) in enumerate(zip(
            graph_data['nodes'], graph_data['node_types'],
            results_df['is_anomaly'])):
            G.add_node(i, name=name,
                      type='buyer' if node_type == 0 else 'supplier',
                      anomaly=is_anom)
        
        # Add edges
        for edge in graph_data['edges']:
            G.add_edge(edge[0], edge[1])
        
        # Extract anomalous subgraph
        anomalous_nodes = results_df[results_df['is_anomaly']].index.tolist()
        anomalous_neighbors = set(anomalous_nodes)
        
        for node in anomalous_nodes:
            anomalous_neighbors.update(G.neighbors(node))
        
        subgraph = G.subgraph(anomalous_neighbors)
        
        # Community detection using embeddings
        communities_info = {}
        if len(anomalous_neighbors) > 3:
            anomalous_embeddings = embeddings[list(anomalous_neighbors)]
            
            # Use DBSCAN for community detection
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            communities = dbscan.fit_predict(anomalous_embeddings)
            
            # Add community information to nodes
            for node, community in zip(anomalous_neighbors, communities):
                G.nodes[node]['community'] = int(community)
            
            communities_info = {
                'num_communities': len(set(communities)) - \
                    (1 if -1 in communities else 0),
                'noise_points': np.sum(communities == -1),
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
        df = graph_builder.load_data(DATA_PATH)
        
        # Create graph
        graph_data = graph_builder.create_graph(df)
        
        # Scale features before creating TensorFlow graph
        node_features = graph_data['node_features']
        edge_features = graph_data['edge_features']
        
        # Scale the features
        node_features_scaled = graph_builder.node_scaler.fit_transform(
            node_features)
        edge_features_scaled = graph_builder.edge_scaler.fit_transform(
            edge_features)
        
        # Create TensorFlow graph
        graph_tensor = gnn_detector.create_tensorflow_graph(
            graph_data, node_features_scaled, edge_features_scaled)
        
        # Build model
        gnn_detector.model = gnn_detector.build_model(
            node_features_scaled.shape[1], edge_features_scaled.shape[1])
        
        # Train model
        history = gnn_detector.train(graph_tensor, epochs=50)
        
        # Detect anomalies
        reconstruction_error, anomalies, threshold = \
            gnn_detector.detect_anomalies()
        
        # Get embeddings for analysis
        predictions = gnn_detector.model.predict(graph_tensor)
        embeddings = predictions['embeddings']
        
        # Create results DataFrame
        results_df = analyzer.create_results_dataframe(
            graph_data, reconstruction_error, anomalies)
        
        # Analyze communities
        communities_info = analyzer.analyze_anomalous_communities(
            graph_data, results_df, embeddings)
        
        # Visualize results
        analyzer.plot_results(results_df, reconstruction_error, threshold,
                             embeddings)
        
        # Save results
        gnn_detector.model.save(os.path.join(MODEL_PATH, 'gnn_anomaly_model'))
        results_df.to_csv(os.path.join(MODEL_PATH, 'gnn_anomaly_results.csv'),
                         index=False)
        np.save(os.path.join(MODEL_PATH, 'gnn_node_embeddings.npy'),
               embeddings)
        
        # Print summary
        print("\n" + "="*60)
        print("GNN ANOMALY DETECTION SUMMARY")
        print("="*60)
        print(f"Total entities analyzed: {len(graph_data['nodes'])}")
        print(f"- Buyers: {np.sum(graph_data['node_types'] == 0)}")
        print(f"- Suppliers: {np.sum(graph_data['node_types'] == 1)}")
        print(f"\nAnomalies detected: {np.sum(anomalies)} "
              f"({np.sum(anomalies)/len(anomalies)*100:.1f}%)")
        
        anomalous_buyers = results_df[
            (results_df['is_anomaly']) &
            (results_df['entity_type'] == 'Buyer')]
        anomalous_suppliers = results_df[
            (results_df['is_anomaly']) &
            (results_df['entity_type'] == 'Supplier')]
        
        print(f"- Anomalous buyers: {len(anomalous_buyers)}")
        print(f"- Anomalous suppliers: {len(anomalous_suppliers)}")
        print(f"\nModel performance:")
        print(f"- Final reconstruction loss: "
              f"{history['reconstructed_loss'][-1]:.4f}")
        print(f"- Anomaly threshold: {threshold:.4f}")
        
        if communities_info:
            print(f"\nCommunity analysis:")
            print(f"- Communities detected: "
                  f"{communities_info['num_communities']}")
            print(f"- Anomalous subgraph size: "
                  f"{communities_info['subgraph_size']} nodes")
        
        print(f"\nTop 5 most anomalous entities:")
        top_anomalies = results_df.head(5)[
            ['entity_name', 'entity_type', 'reconstruction_error']]
        print(top_anomalies.to_string(index=False))
        
        # Visualize the full procurement graph
        graph_builder.visualize_procurement_graph(graph_data, 
            "French Public Procurement Network")
        
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