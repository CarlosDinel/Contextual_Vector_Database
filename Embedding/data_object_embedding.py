""" Data Object Embedding Module for Contextual Vector Database

This module implements the data object embedding component that coordinates
various specialized embedders to create comprehensive vector representations
of complex data objects.

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
    from .multimodal_embedding import MultimodalEmbedding
    from .vector_relationship import VectorRelationship
except ImportError:
    # Fallback for direct script execution
    from text_embedding import TextEmbedding
    from transaction_embedding import TransactionEmbedding
    from time_stamp_embedding import TimeStampEmbedding
    from multimodal_embedding import MultimodalEmbedding
    from vector_relationship import VectorRelationship

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataObjectEmbedding:
    """
    Main class for embedding complex data objects, coordinating various specialized embedders.
    """
    def __init__(self, 
                 max_vector_size: int = 1024,
                 normalize: bool = True,
                 dimension_threshold: int = 100,
                 similarity_threshold: float = 0.8,
                 max_children: int = 5):
        """
        Initialize the DataObjectEmbedding.
        
        Args:
            max_vector_size: Maximum size of the combined vector
            normalize: Whether to normalize vectors
            dimension_threshold: Threshold for vector dimensionality that triggers splitting
            similarity_threshold: Minimum similarity required for vectors to be considered related
            max_children: Maximum number of child vectors a mother vector can have
        """
        self.max_vector_size = max_vector_size
        self.normalize = normalize
        
        # Initialize specialized embedders
        self.text_embedder = TextEmbedding()
        self.transaction_embedder = TransactionEmbedding()
        self.timestamp_embedder = TimeStampEmbedding()
        self.multimodal_embedder = MultimodalEmbedding()
        
        # Initialize vector relationship manager
        self.vector_relationship = VectorRelationship(
            dimension_threshold=dimension_threshold,
            similarity_threshold=similarity_threshold,
            max_children=max_children
        )
        
        # Initialize storage
        self.vectors = {}
        self.metadata = {}
        
    def embed_data_object(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Embed a complete data object into a vector representation.
        
        Args:
            data: Data object to embed
            
        Returns:
            np.ndarray: Vector representation of the data object
        """
        try:
            # Initialize vector components
            components = []
            weights = []
            
            # Embed text fields if present
            if 'text' in data:
                text_vec = self.text_embedder.embed(data['text'])
                components.append(text_vec)
                weights.append(0.4)  # Text weight
            
            # Embed transaction data if present
            if 'transaction' in data:
                trans_vec = self.transaction_embedder.embed(data['transaction'])
                components.append(trans_vec)
                weights.append(0.3)  # Transaction weight
            
            # Embed timestamp if present
            if 'timestamp' in data:
                ts_vec = self.timestamp_embedder.embed(data['timestamp'])
                components.append(ts_vec)
                weights.append(0.2)  # Timestamp weight
            
            # Embed metadata if present
            if 'metadata' in data:
                meta_vec = self.text_embedder.embed(str(data['metadata']))
                components.append(meta_vec)
                weights.append(0.1)  # Metadata weight
            
            if not components:
                raise ValueError("No valid components found in data object")
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Combine vectors
            combined_vec = self._combine_vectors(components, weights)
            
            # Ensure we have a numpy array
            if isinstance(combined_vec, str):
                raise ValueError("Received string instead of vector")
                
            # Ensure vector is 1D
            combined_vec = combined_vec.flatten()
            
            # Normalize the final vector
            combined_vec = combined_vec / np.linalg.norm(combined_vec)
            
            return combined_vec
            
        except Exception as e:
            logger.error(f"Failed to embed data object: {e}")
            raise
    
    def _combine_vectors(self, vectors: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """
        Combine multiple vectors with weights.
        
        Args:
            vectors: List of vectors to combine
            weights: List of weights for each vector
            
        Returns:
            np.ndarray: Combined vector
        """
        try:
            if not vectors:
                raise ValueError("No vectors provided for combination")
                
            if len(vectors) != len(weights):
                raise ValueError("Number of vectors must match number of weights")
                
            # Ensure all vectors are 1D
            vectors = [vec.flatten() for vec in vectors]
            
            # Find maximum dimension
            max_dim = max(vec.shape[0] for vec in vectors)
            
            # Pad vectors to same dimension
            padded_vectors = []
            for vector in vectors:
                if vector.shape[0] < max_dim:
                    # Pad with zeros if needed
                    padded = np.zeros(max_dim)
                    padded[:vector.shape[0]] = vector
                    padded_vectors.append(padded)
        else:
                    padded_vectors.append(vector)
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Combine vectors
            combined = np.zeros(max_dim)
            for vec, weight in zip(padded_vectors, weights):
                combined += vec * weight
                
            # Normalize final vector
            return self._normalize_vector(combined)
            
        except Exception as e:
            logger.error(f"Failed to combine vectors: {e}")
            raise
            
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: Input vector
            
        Returns:
            np.ndarray: Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        else:
            return vector
    
    def _generate_vector_id(self) -> str:
        """Generate a unique vector ID."""
        return f"vec_{datetime.now().timestamp()}_{len(self.vectors)}"
    
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
    
    def get_family(self, vector_id: str) -> List[str]:
        """Get all vectors in the same family (mother and children)."""
        return list(self.vector_relationship.get_family(vector_id))
    
    def update_vector(self, 
                     vector_id: str,
                     new_data: Dict[str, Any],
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Update an existing vector with new data.
        
        Args:
            vector_id: ID of the vector to update
            new_data: New data to incorporate
            metadata: Additional metadata
            
        Returns:
            str: ID of the updated vector
        """
        # Get existing vector and metadata
        existing_vector = self.get_vector(vector_id)
        existing_metadata = self.get_metadata(vector_id)
        
        # Create new vector with combined data
        new_vector_id = self.embed_data_object(
            new_data,
            metadata={**existing_metadata, **(metadata or {})}
        )
        
        # If the vector was split, update relationships
        if self.vector_relationship.is_mother_vector(new_vector_id):
            # Update mother-child relationships
            children = self.vector_relationship.get_children(new_vector_id)
            for child_id in children:
                self.metadata[child_id]['parent_id'] = new_vector_id
                
        return new_vector_id
    
    def delete_vector(self, vector_id: str):
        """Delete a vector and its related vectors."""
        # Get all vectors in the family
        family = self.vector_relationship.get_family(vector_id)
        
        # Delete all vectors in the family
        for member_id in family:
            if member_id in self.vectors:
                del self.vectors[member_id]
            if member_id in self.metadata:
                del self.metadata[member_id]
                
        # Update relationships
        if self.vector_relationship.is_mother_vector(vector_id):
            del self.vector_relationship.mother_to_children[vector_id]
            for child_id in self.vector_relationship.get_children(vector_id):
                del self.vector_relationship.child_to_mother[child_id]

# Example usage
if __name__ == "__main__":
    # Create data object embedder
    embedder = DataObjectEmbedding()
    
    # Example: Embed a complex data object
    data_object = {
        'text': "Sample text description",
        'transaction': {
            'amount': 100,
            'type': 'purchase'
        },
        'timestamp': datetime.now().isoformat(),
        'multimodal': {
            'text': "Additional text",
            'image': None  # Replace with actual image if available
        }
    }
    
    # Embed the data object
    vector = embedder.embed_data_object(data_object)
    
    # Get the vector and its family
    vector_id = embedder.get_vector(vector)
    family = embedder.get_family(vector_id)
    
    print(f"Vector shape: {vector.shape}")
    print(f"Vector family: {family}")



