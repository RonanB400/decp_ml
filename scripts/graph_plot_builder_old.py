import pandas as pd
import os
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphPlotBuilder:
    """Build graph structures from procurement data."""
    
    def __init__(self):
        pass
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load procurement data from CSV files."""
        logger.info(f"Loading data from {data_path}")

        data_file_path = os.path.join(data_path, 'data_clean.csv')
        X = pd.read_csv(data_file_path, encoding='utf-8')
        # Basic data validation
        selected_columns = ['dateNotification', 'acheteur_nom',
                            'titulaire_nom', 'montant', 'dureeMois',
                            'codeCPV', 'procedure', 'objet']
        
        X = X[selected_columns]
        
        return X
    

    def create_graph(self, X: pd.DataFrame) -> dict:
        """Transform procurement data into graph structure.
        
        Args:
            X: DataFrame from load_data with original columns
            type: Type of data (train/test/val) for saving
        """
        logger.info("Creating graph structure from procurement data...")
        
        # Remove rows with NaN buyer or supplier names
        valid_mask = (X['acheteur_nom'].notna() &
                      X['titulaire_nom'].notna())
        X_filtered = X[valid_mask].copy()
        
        logger.info(f"Filtered to {len(X_filtered)} valid contracts "
                    f"(removed {(~valid_mask).sum()} contracts with "
                    f"missing names)")
        
        # Create unique identifiers for buyers and suppliers
        buyers = X_filtered['acheteur_nom'].unique()
        suppliers = X_filtered['titulaire_nom'].unique()
        
        # Create node mappings
        buyer_to_id = {buyer: i for i, buyer in enumerate(buyers)}
        supplier_to_id = {supplier: i + len(buyers)
                          for i, supplier in enumerate(suppliers)}
        
        # Combine all nodes
        all_nodes = list(buyers) + list(suppliers)
        
        logger.info("Creating edges from procurement data...")
        
        # Map buyer and supplier names to IDs
        buyer_ids = X_filtered['acheteur_nom'].map(buyer_to_id).values
        supplier_ids = X_filtered['titulaire_nom'].map(supplier_to_id).values
        edges = list(zip(buyer_ids, supplier_ids))
        
        # Create edge features from available columns
        edge_features = []
        for _, row in X_filtered.iterrows():
            features = [
                row['montant'] if pd.notna(row['montant']) else 0,
                row['dureeMois'] if pd.notna(row['dureeMois']) else 0,
            ]
            edge_features.append(features)
        
        # Store contract IDs for analysis
        contract_ids = X_filtered.index.tolist()
        
        logger.info("Computing node features...")
        
        # Compute node features for buyers
        buyer_features = self._compute_node_features(
            X_filtered, 'acheteur_nom', buyers)
        
        # Compute node features for suppliers  
        supplier_features = self._compute_node_features(
            X_filtered, 'titulaire_nom', suppliers)
        
        # Build node features arrays
        node_features = []
        node_types = []
        
        # Buyer features
        for buyer in buyers:
            features = buyer_features[buyer]
            node_features.append(features)
            node_types.append(0)  # Buyer
        
        # Supplier features
        for supplier in suppliers:
            features = supplier_features[supplier]
            node_features.append(features)
            node_types.append(1)  # Supplier
        
        # Create contract data for analysis
        contract_data = X_filtered[['acheteur_nom', 'titulaire_nom',
                                    'montant', 'codeCPV', 'procedure',
                                    'dureeMois']].copy()
        
        graph_data = {
            'nodes': all_nodes,
            'edges': edges,
            'node_features': node_features,
            'edge_features': edge_features,
            'node_types': node_types,
            'buyer_to_id': buyer_to_id,
            'supplier_to_id': supplier_to_id,
            'contract_ids': contract_ids,
            'contract_data': contract_data
        }

        # Save the graph data to a pickle file
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data')
        os.makedirs(data_dir, exist_ok=True)
        graph_file_path = os.path.join(data_dir, f'graph_data_clean.pkl')
        with open(graph_file_path, 'wb') as f:
            pickle.dump(graph_data, f)
        
        return graph_data

    def _compute_node_features(self, data: pd.DataFrame,
                               entity_col: str, entities: list) -> dict:
        """Compute features for nodes based on their contracts."""
        features = {}
        
        for entity in entities:
            entity_data = data[data[entity_col] == entity]
            
            # Basic statistics
            contract_count = len(entity_data)
            total_amount = (entity_data['montant'].sum()
                            if len(entity_data) > 0 else 0)
            avg_amount = (entity_data['montant'].mean()
                          if len(entity_data) > 0 else 0)
            avg_duration = (entity_data['dureeMois'].mean()
                            if len(entity_data) > 0 else 0)
            
            features[entity] = [
                contract_count,
                total_amount,
                avg_amount,
                avg_duration
            ]
        
        return features
