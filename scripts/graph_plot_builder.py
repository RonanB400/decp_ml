import pandas as pd
import os
import pickle
import logging
from typing import Optional, Union
from pyvis.network import Network
import networkx as nx
import random

try:
    import graphistry
    from dotenv import load_dotenv
    GRAPHISTRY_AVAILABLE = True
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    GRAPHISTRY_AVAILABLE = False
    print("Warning: graphistry or python-dotenv not available")

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

    def create_graph(self, X: pd.DataFrame,
                     min_contract_amount: float = 20_000) -> dict:
        """Transform procurement data into graph structure.
        
        Args:
            X: DataFrame from load_data with original columns
            min_contract_amount: Minimum contract amount to include
        """
        logger.info("Creating graph structure from procurement data...")
        
        # Remove rows with NaN buyer or supplier names
        valid_mask = (X['acheteur_nom'].notna() &
                      X['titulaire_nom'].notna())
        X_filtered = X[valid_mask].copy()
        
        # Filter by minimum contract amount if specified
        if min_contract_amount > 0:
            amount_mask = X_filtered['montant'] >= min_contract_amount
            X_filtered = X_filtered[amount_mask].copy()
            logger.info(f"Filtered contracts with amount >= "
                        f"{min_contract_amount:,.2f}€")
        
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
        
        # OPTIMIZATION: Vectorized edge creation
        buyer_ids = X_filtered['acheteur_nom'].map(buyer_to_id).values
        supplier_ids = X_filtered['titulaire_nom'].map(supplier_to_id).values
        edges = list(zip(buyer_ids, supplier_ids))
        
        # OPTIMIZATION: Vectorized edge features creation
        edge_features = (X_filtered[['montant', 'dureeMois']]
                         .fillna(0).values.tolist())
        
        # Store contract IDs for analysis
        contract_ids = X_filtered.index.tolist()
        
        logger.info("Computing node features with vectorized operations...")
        
        # OPTIMIZATION: Bulk computation using groupby
        buyer_features, supplier_features = (
            self._compute_bulk_node_features(X_filtered))
        
        # Build node features arrays
        node_features = []
        node_types = []
        
        # Buyer features
        for buyer in buyers:
            features = buyer_features.get(buyer, [0, 0, 0, 0])
            node_features.append(features)
            node_types.append(0)  # Buyer
        
        # Supplier features
        for supplier in suppliers:
            features = supplier_features.get(supplier, [0, 0, 0, 0])
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
        graph_file_path = os.path.join(data_dir, 'graph_data_clean.pkl')
        with open(graph_file_path, 'wb') as f:
            pickle.dump(graph_data, f)
        
        return graph_data

    def _compute_bulk_node_features(self, data: pd.DataFrame) -> tuple:
        """Compute features for all nodes using vectorized operations."""
        
        # Compute buyer features using groupby (much faster)
        buyer_stats = data.groupby('acheteur_nom').agg({
            'montant': ['count', 'sum', 'mean'],
            'dureeMois': 'mean'
        }).round(2)
        
        # Flatten column names
        buyer_stats.columns = ['contract_count', 'total_amount',
                               'avg_amount', 'avg_duration']
        buyer_stats = buyer_stats.fillna(0)
        
        # Convert to dictionary format
        buyer_features = {}
        for buyer in buyer_stats.index:
            row = buyer_stats.loc[buyer]
            buyer_features[buyer] = [
                int(row['contract_count']),
                float(row['total_amount']),
                float(row['avg_amount']),
                float(row['avg_duration'])
            ]
        
        # Compute supplier features using groupby
        supplier_stats = data.groupby('titulaire_nom').agg({
            'montant': ['count', 'sum', 'mean'],
            'dureeMois': 'mean'
        }).round(2)
        
        # Flatten column names
        supplier_stats.columns = ['contract_count', 'total_amount',
                                  'avg_amount', 'avg_duration']
        supplier_stats = supplier_stats.fillna(0)
        
        # Convert to dictionary format
        supplier_features = {}
        for supplier in supplier_stats.index:
            row = supplier_stats.loc[supplier]
            supplier_features[supplier] = [
                int(row['contract_count']),
                float(row['total_amount']),
                float(row['avg_amount']),
                float(row['avg_duration'])
            ]
        
        return buyer_features, supplier_features

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

    def plot_graph(self, graph_data: dict,
                   output_path: str = "graph_visualization.html",
                   focus_node: Optional[Union[str, int]] = None,
                   max_nodes: int = 100,
                   max_edges: int = 200,
                   physics_enabled: bool = True) -> None:
        """Visualize the graph using pyvis network.
        
        Args:
            graph_data: Graph data dictionary from create_graph
            output_path: Path to save the HTML visualization
            focus_node: Node name or ID to zoom/focus on
            max_nodes: Maximum number of nodes to display
            max_edges: Maximum number of edges to display
            physics_enabled: Whether to enable physics simulation
        """
        logger.info("Creating graph visualization with pyvis...")
        
        # Get graph components
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        node_features = graph_data['node_features']
        node_types = graph_data['node_types']
        edge_features = graph_data['edge_features']
        
        # Limit nodes if necessary
        if len(nodes) > max_nodes:
            logger.info(f"Limiting display to {max_nodes} nodes "
                        f"(total: {len(nodes)})")
            # Get top nodes by contract count
            node_importance = [features[0] for features in node_features]
            top_indices = sorted(range(len(node_importance)),
                                 key=lambda i: node_importance[i],
                                 reverse=True)[:max_nodes]
            
            # Filter nodes and features
            nodes = [nodes[i] for i in top_indices]
            node_features = [node_features[i] for i in top_indices]
            node_types = [node_types[i] for i in top_indices]
            
            # Create mapping for filtered nodes
            old_to_new = {old_idx: new_idx
                          for new_idx, old_idx in enumerate(top_indices)}
            
            # Filter edges to only include nodes in our subset
            filtered_edges = []
            filtered_edge_features = []
            for i, edge in enumerate(edges):
                if edge[0] in old_to_new and edge[1] in old_to_new:
                    filtered_edges.append((old_to_new[edge[0]],
                                           old_to_new[edge[1]]))
                    filtered_edge_features.append(edge_features[i])
            edges = filtered_edges
            edge_features = filtered_edge_features
        
        # Limit total edges if needed
        if len(edges) > max_edges:
            logger.info(f"Limiting edges to {max_edges} (total: {len(edges)})")
            
            # Create list of (edge, features, amount) for sorting
            edge_data = []
            for i, (edge, features) in enumerate(zip(edges, edge_features)):
                amount = features[0] if len(features) > 0 else 0
                edge_data.append((edge, features, amount))
            
            # Sort by contract amount (descending) and limit
            edge_data.sort(key=lambda x: x[2], reverse=True)
            edge_data = edge_data[:max_edges]
            
            # Extract filtered edges and features
            edges = [item[0] for item in edge_data]
            edge_features = [item[1] for item in edge_data]
            
            logger.info(f"Using {len(edges)} edges after filtering")
        
        # Create network with larger size
        net = Network(height="1000px", width="100%", directed=False)

        # net = Network(height="600px", width="100%", directed=False,
        #                select_menu=True, filter_menu=True)
        
        # Create initial positions to avoid center clustering
        logger.info("Computing initial node positions...")
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        G.add_edges_from(edges)
        
        # Use spring layout for initial positioning with more spacing
        try:
            pos = nx.spring_layout(G, k=5, iterations=50, seed=42)
        except Exception:
            # Fallback to circular layout if spring layout fails
            pos = nx.circular_layout(G)
        
        # Scale positions to fill the canvas better with much larger spread
        scale_factor = 3000  # Dramatically increased for much larger spacing
        for node_id in pos:
            pos[node_id] = (pos[node_id][0] * scale_factor,
                            pos[node_id][1] * scale_factor)
        
        # Add nodes to the network with initial positions
        for i, (node_name, features, node_type) in enumerate(
                zip(nodes, node_features, node_types)):
            
            # Set node properties based on type
            if node_type == 0:  # Buyer
                color = "#ff9999"  # Light red
                shape = "box"
                type_label = "Acheteur"
            else:  # Supplier
                color = "#99ccff"  # Light blue
                shape = "circle"
                type_label = "Titulaire"
            
            # Scale node size based on contract count
            size = min(15 + features[0] * 3, 60)
            
            # Create tooltip with node information
            title = (f"{type_label}: {node_name}\n"
                     f"Contracts: {features[0]}\n"
                     f"Total Amount: {features[1]:,.2f}€\n"
                     f"Avg Amount: {features[2]:,.2f}€\n"
                     f"Avg Duration: {features[3]:.1f} months")
            
            # Highlight focus node if specified
            if focus_node is not None:
                if ((isinstance(focus_node, str) and
                     node_name == focus_node) or
                        (isinstance(focus_node, int) and i == focus_node)):
                    color = "#ffff00"  # Yellow for focus
                    size *= 1.5
            
            # Get initial position for this node with wider range
            x, y = pos.get(i, (random.uniform(-1500, 1500),
                               random.uniform(-1500, 1500)))
            
            net.add_node(i, label=str(node_name)[:20], color=color,
                         size=size, shape=shape, title=title,
                         x=x, y=y, fixed=False)
        
        # Add edges to the network with variable width
        for i, edge in enumerate(edges):
            # Scale edge width based on contract amount
            if i < len(edge_features):
                features_list = edge_features[i]
                amount = features_list[0] if len(features_list) > 0 else 1
                width = min(1 + amount / 50000, 5)  # Scale width
            else:
                width = 1
            net.add_edge(edge[0], edge[1], width=width)
        
        # Configure physics for stability
        if physics_enabled:
            net.set_options("""
            {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 50},
                "barnesHut": {
                  "gravitationalConstant": -8000,
                  "centralGravity": 0.01,
                  "springLength": 300,
                  "springConstant": 0.005,
                  "damping": 0.3,
                  "avoidOverlap": 1.0
                }
              },
              "interaction": {
                "dragNodes": true,
                "dragView": true,
                "zoomView": true
              }
            }
            """)
        else:
            net.set_options('{"physics": {"enabled": false}}')
        
        # Save the visualization
        net.save_graph(output_path)
        logger.info(f"Graph visualization saved to {output_path}")
        logger.info(f"Visualization created with {len(nodes)} nodes "
                    f"and {len(edges)} edges")
        if focus_node is not None:
            logger.info(f"Focused on node: {focus_node}")

    def plot_graph_graphistry(self, graph_data: dict,
                              max_nodes: int = 500,
                              max_edges: int = 1000,
                              focus_node: Optional[Union[str, int]] = None,
                              node_size_factor: float = 1.0) -> None:
        """Visualize the graph using Graphistry (GPU-accelerated).
        
        Args:
            graph_data: Graph data dictionary from create_graph
            max_nodes: Maximum number of nodes to display
            max_edges: Maximum number of edges to display
            focus_node: Node name or ID to highlight
            node_size_factor: Multiplier for node sizes
        """
        if not GRAPHISTRY_AVAILABLE:
            logger.error("Graphistry not available. Install with: "
                         "pip install graphistry python-dotenv")
            return
        
        logger.info("Creating Graphistry visualization...")
        
        # Get API key from environment
        api_key = os.getenv('GRAPHISTRY_API_KEY')
        if not api_key:
            logger.error("GRAPHISTRY_API_KEY not found in "
                         "environment variables")
            return
        
        # Configure Graphistrys
        try:
            # Try different authentication methods
            if hasattr(graphistry, 'register'):
                graphistry.register(api=3, key=api_key)
            else:
                graphistry.login(api_key)
        except Exception as e:
            logger.error(f"Failed to authenticate with Graphistry: {e}")
            logger.info("Trying alternative authentication method...")
            try:
                # Alternative method
                graphistry.register(api=3, username='', password='',
                                    key=api_key)
            except Exception as e2:
                logger.error(f"Alternative authentication also failed: {e2}")
                return
        
        # Get graph components
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        node_features = graph_data['node_features']
        node_types = graph_data['node_types']
        edge_features = graph_data['edge_features']
        
        # Filter nodes if necessary
        if len(nodes) > max_nodes:
            logger.info(f"Limiting display to {max_nodes} nodes "
                        f"(total: {len(nodes)})")
            # Get top nodes by contract count
            node_importance = [features[0] for features in node_features]
            top_indices = sorted(range(len(node_importance)),
                                 key=lambda i: node_importance[i],
                                 reverse=True)[:max_nodes]
            
            # Filter nodes and features
            nodes = [nodes[i] for i in top_indices]
            node_features = [node_features[i] for i in top_indices]
            node_types = [node_types[i] for i in top_indices]
            
            # Create mapping for filtered nodes
            old_to_new = {old_idx: new_idx
                          for new_idx, old_idx in enumerate(top_indices)}
            
            # Filter edges to only include nodes in our subset
            filtered_edges = []
            filtered_edge_features = []
            for i, edge in enumerate(edges):
                if edge[0] in old_to_new and edge[1] in old_to_new:
                    filtered_edges.append((old_to_new[edge[0]],
                                           old_to_new[edge[1]]))
                    filtered_edge_features.append(edge_features[i])
            edges = filtered_edges
            edge_features = filtered_edge_features
        
        # Limit edges if necessary
        if len(edges) > max_edges:
            logger.info(f"Limiting edges to {max_edges} (total: {len(edges)})")
            edge_data = []
            for i, (edge, features) in enumerate(zip(edges, edge_features)):
                amount = features[0] if len(features) > 0 else 0
                edge_data.append((edge, features, amount))
            
            # Sort by contract amount and limit
            edge_data.sort(key=lambda x: x[2], reverse=True)
            edge_data = edge_data[:max_edges]
            
            edges = [item[0] for item in edge_data]
            edge_features = [item[1] for item in edge_data]
        
        # Create nodes DataFrame
        node_size = [min(10 + node_features[i][0] * node_size_factor, 50)
                     for i in range(len(nodes))]
        node_color = ['#ff6b6b' if node_types[i] == 0 else '#4ecdc4'
                      for i in range(len(nodes))]
        
        nodes_df = pd.DataFrame({
            'node_id': range(len(nodes)),
            'name': nodes,
            'type': ['Acheteur' if node_types[i] == 0 else 'Titulaire'
                     for i in range(len(nodes))],
            'contracts': [node_features[i][0] for i in range(len(nodes))],
            'total_amount': [node_features[i][1] for i in range(len(nodes))],
            'avg_amount': [node_features[i][2] for i in range(len(nodes))],
            'avg_duration': [node_features[i][3] for i in range(len(nodes))],
            'size': node_size,
            'color': node_color
        })
        
        # Create edges DataFrame
        edge_amounts = [edge_features[i][0] if i < len(edge_features) else 0
                        for i in range(len(edges))]
        edge_durations = [edge_features[i][1] if i < len(edge_features) else 0
                          for i in range(len(edges))]
        edge_widths = [min(1 + (edge_features[i][0] / 50000
                                if i < len(edge_features) else 0), 5)
                       for i in range(len(edges))]
        
        edges_df = pd.DataFrame({
            'source': [edge[0] for edge in edges],
            'target': [edge[1] for edge in edges],
            'amount': edge_amounts,
            'duration': edge_durations,
            'width': edge_widths
        })
        
        # Highlight focus node if specified
        if focus_node is not None:
            focus_id = None
            if isinstance(focus_node, str):
                try:
                    focus_id = nodes.index(focus_node)
                    focus_mask = nodes_df['node_id'] == focus_id
                    nodes_df.loc[focus_mask, 'color'] = '#ffff00'
                    nodes_df.loc[focus_mask, 'size'] *= 1.5
                except ValueError:
                    logger.warning(f"Focus node '{focus_node}' not found")
            elif isinstance(focus_node, int) and 0 <= focus_node < len(nodes):
                focus_mask = nodes_df['node_id'] == focus_node
                nodes_df.loc[focus_mask, 'color'] = '#ffff00'
                nodes_df.loc[focus_mask, 'size'] *= 1.5
        
        # Create Graphistry plot
        try:
            g = graphistry.edges(edges_df, 'source', 'target').nodes(
                nodes_df, 'node_id')
            
            # Configure visual settings
            g2 = (g.bind(point_color='color', point_size='size',
                         edge_weight='width')
                  .settings(url_params={'play': 0,  # Start paused
                                        'splashAfter': 0,  # No splash screen
                                        'info': True,  # Show info panel
                                        'showArrows': False,  # Undirected
                                        'backgroundColor': '#1a1a1a'}))
            
            # Plot and return URL
            url = g2.plot(render=False)
            logger.info(f"Graphistry visualization created: {url}")
            logger.info(f"Visualization has {len(nodes)} nodes and "
                        f"{len(edges)} edges")
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to create Graphistry visualization: {e}")
            return None 