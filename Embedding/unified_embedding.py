""" Unified Embedding Module for Contextual Vector Database

This module implements the unified embedding model that coordinates various
specialized embedders for generating comprehensive vector representations.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
import os
import sys

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import specialized embedders
try:
    from .text_embedding import TextEmbedding
    from .transaction_embedding import TransactionEmbedding
    from .time_stamp_embedding import TimeStampEmbedding
    from .data_object_embedding import DataObjectEmbedding
    from .multimodal_embedding import MultimodalEmbedding
except ImportError:
    # Fallback for direct script execution
    from text_embedding import TextEmbedding
    from transaction_embedding import TransactionEmbedding
    from time_stamp_embedding import TimeStampEmbedding
    from data_object_embedding import DataObjectEmbedding
    from multimodal_embedding import MultimodalEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedEmbedding:
    """
    Main class for unified embedding, coordinating various specialized embedders.
    """
    def __init__(self, 
                 max_vector_size: int = 1024,
                 normalize: bool = True):
        """
        Initialize the unified embedder.
        
        Args:
            max_vector_size: Maximum size of the combined vector
            normalize: Whether to normalize vectors
        """
        self.max_vector_size = max_vector_size
        self.normalize = normalize
        
        # Initialize specialized embedders
        self.text_embedder = TextEmbedding()
        self.transaction_embedder = TransactionEmbedding()
        self.timestamp_embedder = TimeStampEmbedding()
        self.data_object_embedder = DataObjectEmbedding()
        self.multimodal_embedder = MultimodalEmbedding()
        
        # Initialize storage
        self.vectors = {}
        self.metadata = {}
        
    def embed_data(self, 
                  data: Union[str, Dict[str, Any]],
                  data_type: str,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Embed data using the appropriate specialized embedder.
        
        Args:
            data: The data to embed
            data_type: Type of data ('text', 'transaction', 'timestamp', 'data_object', 'multimodal')
            metadata: Additional metadata about the data
            
        Returns:
            str: ID of the created vector
        """
        # Get the appropriate embedder
        embedder = self._get_embedder(data_type)
        if not embedder:
            raise ValueError(f"Unsupported data type: {data_type}")
            
        # Embed the data
        vector = embedder.embed(data)
        
        # Normalize if requested
        if self.normalize:
            vector = self._normalize_vector(vector)
            
        # Generate vector ID
        vector_id = self._generate_vector_id()
        
        # Store vector and metadata
        self.vectors[vector_id] = vector
        self.metadata[vector_id] = metadata or {}
        
        return vector_id
    
    def _get_embedder(self, data_type: str) -> Optional[Any]:
        """Get the appropriate embedder for the data type."""
        embedders = {
            'text': self.text_embedder,
            'transaction': self.transaction_embedder,
            'timestamp': self.timestamp_embedder,
            'data_object': self.data_object_embedder,
            'multimodal': self.multimodal_embedder
        }
        return embedders.get(data_type)
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def _generate_vector_id(self) -> str:
        """Generate a unique vector ID."""
        return f"vec_{datetime.now().timestamp()}_{len(self.vectors)}"
    
    def combine_vectors(self, 
                       vector_ids: List[str],
                       weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Combine multiple vectors with optional weights.
        
        Args:
            vector_ids: List of vector IDs to combine
            weights: Optional list of weights for each vector
            
        Returns:
            np.ndarray: Combined vector
        """
        if not vector_ids:
            raise ValueError("No vectors to combine")
            
        # Get vectors
        vectors = [self.vectors[vid] for vid in vector_ids]
        
        # Use uniform weights if none provided
        if weights is None:
            weights = [1.0 / len(vectors)] * len(vectors)
            
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Combine vectors
        combined = np.zeros_like(vectors[0])
        for vector, weight in zip(vectors, weights):
            combined += vector * weight
            
        if self.normalize:
            combined = self._normalize_vector(combined)
            
        return combined
    
    def update_vector(self, 
                     vector_id: str,
                     new_data: Union[str, Dict[str, Any]],
                     data_type: str,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Update an existing vector with new data.
        
        Args:
            vector_id: ID of the vector to update
            new_data: New data to incorporate
            data_type: Type of the new data
            metadata: Additional metadata
            
        Returns:
            str: ID of the updated vector
        """
        # Get existing vector and metadata
        existing_vector = self.vectors[vector_id]
        existing_metadata = self.metadata[vector_id]
        
        # Create new vector
        new_vector_id = self.embed_data(
            new_data,
            data_type,
            metadata={**existing_metadata, **(metadata or {})}
        )
        
        # Combine with existing vector
        combined_vector = self.combine_vectors(
            [vector_id, new_vector_id],
            weights=[0.7, 0.3]  # Give more weight to existing vector
        )
        
        # Update storage
        self.vectors[new_vector_id] = combined_vector
        self.metadata[new_vector_id]['parent_id'] = vector_id
        
        return new_vector_id
    
    def get_vector(self, vector_id: str) -> np.ndarray:
        """Get a vector by ID."""
        if vector_id not in self.vectors:
            raise ValueError(f"Vector {vector_id} not found")
        return self.vectors[vector_id]
    
    def get_metadata(self, vector_id: str) -> Dict[str, Any]:
        """Get metadata for a vector."""
        if vector_id not in self.metadata:
            raise ValueError(f"Vector {vector_id} not found")
        return self.metadata[vector_id]
    
    def delete_vector(self, vector_id: str):
        """Delete a vector and its metadata."""
        if vector_id in self.vectors:
            del self.vectors[vector_id]
        if vector_id in self.metadata:
            del self.metadata[vector_id]

# Example usage
if __name__ == "__main__":
    # Create unified embedder
    embedder = UnifiedEmbedding()
    
    # Example: Embed text
    text_vector_id = embedder.embed_data(
        "Sample text for embedding",
        "text",
        metadata={'source': 'example'}
    )
    
    # Example: Embed transaction
    transaction_vector_id = embedder.embed_data(
        {
            'amount': 100.0,
            'type': 'purchase'
        },
        "transaction",
        metadata={'source': 'example'}
    )
    
    # Combine vectors
    combined_vector = embedder.combine_vectors(
        [text_vector_id, transaction_vector_id],
        weights=[0.5, 0.5]
    )
    
    print(f"Combined vector shape: {combined_vector.shape}") 