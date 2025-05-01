"""
Main DaPGRaC class for the DaPGRaC system.

This is the primary interface for the DaPGRaC core module, which orchestrates
all the components to provide dynamic relationship formation and evolution.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass

from .relationship_executor import RelationshipExecutor
from .decay_controller import DecayController
from .filters import RelationshipFilter
from .edge_dynamics import EdgeDynamics

@dataclass
class DaPGRaCConfig:
    """Configuration for the DaPGRaC system."""
    hidden_layers: List[int] = [64, 32]
    relationship_threshold: float = 0.01
    max_iterations: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    decay_rate: float = 0.95
    edge_threshold: float = 0.1

class DaPGRaC:
    """
    Feed-forward network layer for relationship formation and evolution.
    Processes contextual observations from the Contexter to form, update, or decay relationships.
    """
    
    def __init__(self, config: Optional[DaPGRaCConfig] = None):
        """
        Initialize the DaPGRaC feed-forward network.
        
        Args:
            config: Optional configuration object with parameters.
        """
        self.config = config or DaPGRaCConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.relationship_executor = RelationshipExecutor(
            threshold=self.config.relationship_threshold,
            batch_size=self.config.batch_size
        )
        self.decay_controller = DecayController(
            decay_rate=self.config.decay_rate
        )
        self.relationship_filter = RelationshipFilter()
        self.edge_dynamics = EdgeDynamics(
            threshold=self.config.edge_threshold
        )
        
        # Initialize weights for feed-forward network
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights for the feed-forward network."""
        self.weights = []
        self.biases = []
        
        # Initialize weights for each layer
        for i in range(len(self.config.hidden_layers)):
            if i == 0:
                # First layer weights
                self.weights.append(
                    np.random.randn(self.config.hidden_layers[i], self.config.hidden_layers[i]) * 0.01
                )
            else:
                # Subsequent layer weights
                self.weights.append(
                    np.random.randn(self.config.hidden_layers[i-1], self.config.hidden_layers[i]) * 0.01
                )
            self.biases.append(np.zeros(self.config.hidden_layers[i]))
    
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
        
        # Process through feed-forward layers
        self._process_feed_forward(updated_vectors, context)
        
        # Update relationships
        self._update_relationships(updated_vectors)
        
        return updated_vectors
    
    def _process_feed_forward(self, vectors: List[Dict], context: Optional[Dict] = None) -> None:
        """
        Process vectors through the feed-forward network layers.
        
        Args:
            vectors: List of vector dictionaries
            context: Optional context information from the Contexter
        """
        for vector in vectors:
            # Get vector data
            data = np.array(vector['data'])
            
            # Process through each layer
            for i, (weights, biases) in enumerate(zip(self.weights, self.biases)):
                # Apply feed-forward transformation
                transformed = np.dot(data, weights) + biases
                
                # Apply ReLU activation
                transformed = np.maximum(0, transformed)
                
                # Update vector data
                vector['data'] = transformed.tolist()
                
                # Apply edge dynamics
                if context and vector['id'] in context:
                    self.edge_dynamics.update_edges(vector, context[vector['id']])
    
    def _update_relationships(self, vectors: List[Dict]) -> None:
        """
        Update relationships between vectors using the new relationship execution system.
        
        Args:
            vectors: List of vector dictionaries
        """
        # Execute relationship updates
        updated_relationships = self.relationship_executor.execute_updates(vectors)
        
        # Apply decay to relationships
        self.decay_controller.update_relationship_strengths(vectors)
        
        # Filter relationships
        for vector in vectors:
            vector['relationships'] = self.relationship_filter.filter_relationships(
                vector.get('relationships', {})
            )
    
    def get_strongest_relationships(self, vector: Dict, top_n: int = 5) -> List[Dict]:
        """
        Get the strongest relationships for a vector.
        
        Args:
            vector: Vector dictionary.
            top_n: Number of top relationships to return.
            
        Returns:
            List of strongest relationship dictionaries.
        """
        relationships = vector.get('relationships', {}).values()
        sorted_relationships = sorted(
            relationships,
            key=lambda r: r.get('strength', 0.0),
            reverse=True
        )
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
        if id is None:
            id = f"vector_{int(time.time() * 1000)}"
        
        return {
            'id': id,
            'data': data,
            'created_at': time.time(),
            'last_updated': time.time(),
            'metadata': metadata or {},
            'relationships': {}
        }
    
    def get_relationship_graph(self, vectors: List[Dict], threshold: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """
        Get the relationship graph between vectors.
        
        Args:
            vectors: List of vector dictionaries.
            threshold: Optional strength threshold for including relationships.
            
        Returns:
            Dictionary mapping vector IDs to dictionaries mapping target IDs to strengths.
        """
        threshold = threshold or self.config.relationship_threshold
        graph = {}
        
        for vector in vectors:
            vector_id = vector['id']
            graph[vector_id] = {}
            
            for target_id, relationship in vector.get('relationships', {}).items():
                if relationship['strength'] >= threshold:
                    graph[vector_id][target_id] = relationship['strength']
        
        return graph
