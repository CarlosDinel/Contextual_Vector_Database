"""
Main DaPGRaC class for the DaPGRaC system.

This is the primary interface for the DaPGRaC core module, which orchestrates
all the components to provide dynamic relationship formation and evolution.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Union, Optional, Any

from .shooting_edge_velocity import ShootingEdgeVelocity
from .relationship_magnetism import RelationshipMagnetism
from .relationship_decay import RelationshipDecay
from .relationship_strength import RelationshipStrength
from .relationship_edges import RelationshipEdges


class DaPGRaC:
    """
    Feed-forward network layer for relationship formation and evolution.
    Processes contextual observations from the Contexter to form, update, or decay relationships.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DaPGRaC feed-forward network.
        
        Args:
            config: Optional configuration dictionary with parameters.
        """
        self.config = config or {}
        
        # Initialize feed-forward components
        self.sev_calculator = ShootingEdgeVelocity(self.config.get('sev', {}))
        self.magnetism_calculator = RelationshipMagnetism(self.config.get('magnetism', {}))
        self.decay_calculator = RelationshipDecay(self.config.get('decay', {}))
        self.strength_calculator = RelationshipStrength(self.config.get('strength', {}))
        self.edge_manager = RelationshipEdges(self.config.get('edges', {}))
        
        # Set feed-forward parameters
        self.sev_threshold = self.config.get('sev_threshold', 0.1)
        self.relationship_threshold = self.config.get('relationship_threshold', 0.01)
        self.max_iterations = self.config.get('max_iterations', 100)
        self.hidden_layers = self.config.get('hidden_layers', [64, 32])  # Feed-forward hidden layers
    
    def process_vectors(self, vectors: List[Dict], context: Optional[Dict] = None) -> List[Dict]:
        """
        Process vectors through the DaPGRaC feed-forward network.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'data', etc.
            context: Optional context information from the Contexter
            
        Returns:
            Updated list of vector dictionaries with relationship information.
        """
        # Make a copy of vectors to avoid modifying the input
        updated_vectors = [v.copy() for v in vectors]
        
        # Initialize relationships if not present
        for vector in updated_vectors:
            if 'relationships' not in vector:
                vector['relationships'] = {}
        
        # Process through feed-forward layers
        self._process_feed_forward(updated_vectors, context)
        
        return updated_vectors
    
    def _process_feed_forward(self, vectors: List[Dict], context: Optional[Dict] = None) -> None:
        """
        Process vectors through the feed-forward network layers.
        
        Args:
            vectors: List of vector dictionaries
            context: Optional context information from the Contexter
        """
        # Process each vector through hidden layers
        for vector in vectors:
            # Get vector data and context
            data = np.array(vector['data'])
            if context and vector['id'] in context:
                vector_context = context[vector['id']]
            else:
                vector_context = []
            
            # Process through hidden layers
            for layer_size in self.hidden_layers:
                # Apply feed-forward transformation
                transformed = self._apply_feed_forward(data, layer_size)
                
                # Update vector data
                vector['data'] = transformed.tolist()
                
                # Update relationships based on transformed data
                self.update_relationships(vectors)
    
    def _apply_feed_forward(self, data: np.ndarray, layer_size: int) -> np.ndarray:
        """
        Apply feed-forward transformation to vector data.
        
        Args:
            data: Input vector data
            layer_size: Size of the hidden layer
            
        Returns:
            Transformed vector data
        """
        # Initialize weights and biases
        weights = np.random.randn(data.shape[0], layer_size) * 0.01
        biases = np.zeros(layer_size)
        
        # Apply feed-forward transformation
        transformed = np.dot(data, weights) + biases
        
        # Apply ReLU activation
        transformed = np.maximum(0, transformed)
        
        return transformed
    
    def update_relationships(self, vectors: List[Dict]) -> None:
        """
        Update relationships between vectors.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'data', etc.
            
        Returns:
            None (updates vectors in place).
        """
        # Process each pair of vectors
        for i, vector_i in enumerate(vectors):
            for j, vector_j in enumerate(vectors):
                if i == j:  # Skip self
                    continue
                
                # Get vector IDs
                vector_i_id = vector_i.get('id', str(i))
                vector_j_id = vector_j.get('id', str(j))
                
                # Calculate relationship magnetism
                magnetism = self.magnetism_calculator.calculate_with_vectors(vector_i, vector_j)
                
                # Calculate SEV
                sev = self.sev_calculator.calculate_with_vectors(vector_i, vector_j, self.magnetism_calculator)
                
                # Check if relationship should form or update
                if sev >= self.sev_threshold:
                    # Get existing relationship or create new one
                    if vector_j_id in vector_i.get('relationships', {}):
                        relationship = vector_i['relationships'][vector_j_id].copy()
                    else:
                        relationship = {
                            'source_id': vector_i_id,
                            'target_id': vector_j_id,
                            'strength': 0.0,
                            'edge_type': 'global',
                            'created_at': time.time(),
                            'last_updated': time.time()
                        }
                    
                    # Apply decay to existing strength
                    decayed_relationship = self.decay_calculator.calculate_with_relationship(relationship)
                    
                    # Update strength based on SEV
                    updated_relationship = self.strength_calculator.calculate_with_relationship(decayed_relationship, sev)
                    
                    # Calculate edges (global and segmental)
                    # Automatically identify segments
                    segments = self.edge_manager.identify_segments(np.array(vector_i['data']))
                    
                    # Calculate edges
                    edges = self.edge_manager.calculate_with_vectors(vector_i, vector_j, segments)
                    
                    # Update relationship with edge information
                    updated_relationship['global_edge'] = edges['global_edge']
                    updated_relationship['segmental_edges'] = edges['segmental_edges']
                    updated_relationship['combined_edge'] = edges['combined_edge']
                    
                    # Store updated relationship if significant
                    if updated_relationship['strength'] >= self.relationship_threshold:
                        vector_i['relationships'][vector_j_id] = updated_relationship
                    elif vector_j_id in vector_i.get('relationships', {}):
                        # Remove relationship if below threshold
                        del vector_i['relationships'][vector_j_id]
    
    def get_strongest_relationships(self, vector: Dict, top_n: int = 5) -> List[Dict]:
        """
        Get the strongest relationships for a vector.
        
        Args:
            vector: Vector dictionary.
            top_n: Number of top relationships to return.
            
        Returns:
            List of strongest relationship dictionaries.
        """
        # Get relationships
        relationships = vector.get('relationships', {}).values()
        
        # Sort by strength
        sorted_relationships = sorted(relationships, key=lambda r: r.get('strength', 0.0), reverse=True)
        
        # Return top N
        return sorted_relationships[:top_n]
    
    def create_vector(self, data: List[float], id: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict:
        """
        Create a new vector with default attributes.
        
        Args:
            data: Vector data as a list of floats.
            id: Optional vector ID.
            metadata: Optional metadata dictionary.
            
        Returns:
            Vector dictionary with default attributes.
        """
        # Generate ID if not provided
        if id is None:
            id = f"vector_{int(time.time() * 1000)}"
        
        # Create vector dictionary
        vector = {
            'id': id,
            'data': data,
            'impact_radius': 1.0,  # Default impact radius
            'created_at': time.time(),
            'last_updated': time.time(),
            'metadata': metadata or {},
            'relationships': {}
        }
        
        return vector
    
    def get_relationship_graph(self, vectors: List[Dict], threshold: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """
        Get the relationship graph between vectors.
        
        Args:
            vectors: List of vector dictionaries.
            threshold: Optional strength threshold for including relationships.
            
        Returns:
            Dictionary mapping vector IDs to dictionaries mapping target IDs to strengths.
        """
        # Use default threshold if not provided
        if threshold is None:
            threshold = self.relationship_threshold
        
        # Initialize graph
        graph = {}
        
        # Build graph
        for vector in vectors:
            vector_id = vector.get('id')
            if vector_id is None:
                continue
                
            graph[vector_id] = {}
            
            for target_id, relationship in vector.get('relationships', {}).items():
                strength = relationship.get('strength', 0.0)
                
                if strength >= threshold:
                    graph[vector_id][target_id] = strength
        
        return graph
