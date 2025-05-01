""" Position Encoder Module

This module implements the position encoding and recalculation logic for the Contexter.
It handles the semantic positioning of vectors in the contextual space, ensuring
that vectors maintain meaningful relationships based on their content and context.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PositionInfo:
    """Container for position-related information."""
    position: np.ndarray
    semantic_distance: float
    temporal_distance: float
    structural_distance: float

class PositionEncoder:
    """Handles position encoding and recalculation for vectors."""
    
    def __init__(
        self,
        position_dim: int = 512,
        max_sequence_length: int = 1000,
        temperature: float = 10000.0,
        semantic_weight: float = 0.6,
        temporal_weight: float = 0.2,
        structural_weight: float = 0.2
    ):
        """Initialize the position encoder.
        
        Args:
            position_dim: Dimension of position encodings
            max_sequence_length: Maximum sequence length to handle
            temperature: Temperature parameter for position encoding
            semantic_weight: Weight for semantic distance
            temporal_weight: Weight for temporal distance
            structural_weight: Weight for structural distance
        """
        self.position_dim = position_dim
        self.max_sequence_length = max_sequence_length
        self.temperature = temperature
        self.semantic_weight = semantic_weight
        self.temporal_weight = temporal_weight
        self.structural_weight = structural_weight
        
        # Initialize position encodings
        self.position_encodings = self._create_position_encodings()
        
    def _create_position_encodings(self) -> np.ndarray:
        """Create sinusoidal position encodings.
        
        Returns:
            Position encodings matrix
        """
        position = np.arange(self.max_sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.position_dim, 2) * 
                         -(np.log(self.temperature) / self.position_dim))
        
        pe = np.zeros((self.max_sequence_length, self.position_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
        
    def encode_position(
        self,
        vector: np.ndarray,
        position: int,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Encode position information into a vector.
        
        Args:
            vector: Original vector
            position: Position in sequence
            metadata: Vector metadata
            
        Returns:
            Position-encoded vector
        """
        if position >= self.max_sequence_length:
            position = position % self.max_sequence_length
            
        # Get position encoding
        pos_encoding = self.position_encodings[position]
        
        # Combine with original vector
        encoded = vector + pos_encoding
        
        # Normalize
        encoded = encoded / np.linalg.norm(encoded)
        
        return encoded
        
    def recalculate_position(
        self,
        vector: np.ndarray,
        context_vectors: List[Tuple[str, np.ndarray, Dict[str, Any]]],
        current_position: int
    ) -> PositionInfo:
        """Recalculate vector position based on semantic relationships.
        
        Args:
            vector: Vector to recalculate position for
            context_vectors: List of (vector_id, vector, metadata) tuples
            current_position: Current position in sequence
            
        Returns:
            PositionInfo containing new position and distances
        """
        if not context_vectors:
            return PositionInfo(
                position=vector,
                semantic_distance=0.0,
                temporal_distance=0.0,
                structural_distance=0.0
            )
            
        # Calculate semantic distances
        semantic_distances = []
        for _, context_vec, _ in context_vectors:
            distance = 1 - np.dot(vector, context_vec) / (
                np.linalg.norm(vector) * np.linalg.norm(context_vec)
            )
            semantic_distances.append(distance)
            
        # Calculate temporal distances
        temporal_distances = []
        for _, _, metadata in context_vectors:
            if 'timestamp' in metadata:
                time_diff = abs(current_position - metadata['timestamp'])
                temporal_distances.append(time_diff)
            else:
                temporal_distances.append(0.0)
                
        # Calculate structural distances
        structural_distances = []
        for _, _, metadata in context_vectors:
            if 'connections' in metadata:
                distance = len(metadata['connections'])
                structural_distances.append(distance)
            else:
                structural_distances.append(0.0)
                
        # Normalize distances
        semantic_distances = np.array(semantic_distances)
        temporal_distances = np.array(temporal_distances)
        structural_distances = np.array(structural_distances)
        
        if len(semantic_distances) > 0:
            semantic_distances = semantic_distances / np.max(semantic_distances)
        if len(temporal_distances) > 0:
            temporal_distances = temporal_distances / np.max(temporal_distances)
        if len(structural_distances) > 0:
            structural_distances = structural_distances / np.max(structural_distances)
            
        # Calculate weighted average position
        new_position = np.zeros_like(vector)
        total_weight = 0.0
        
        for i, (_, context_vec, _) in enumerate(context_vectors):
            # Calculate combined weight
            weight = (
                self.semantic_weight * (1 - semantic_distances[i]) +
                self.temporal_weight * (1 - temporal_distances[i]) +
                self.structural_weight * (1 - structural_distances[i])
            )
            
            new_position += weight * context_vec
            total_weight += weight
            
        if total_weight > 0:
            new_position /= total_weight
            
        # Normalize new position
        new_position = new_position / np.linalg.norm(new_position)
        
        return PositionInfo(
            position=new_position,
            semantic_distance=np.mean(semantic_distances),
            temporal_distance=np.mean(temporal_distances),
            structural_distance=np.mean(structural_distances)
        )
        
    def update_position(
        self,
        vector: np.ndarray,
        position_info: PositionInfo,
        learning_rate: float = 0.1
    ) -> np.ndarray:
        """Update vector position based on position info.
        
        Args:
            vector: Current vector
            position_info: Position information
            learning_rate: Learning rate for update
            
        Returns:
            Updated vector
        """
        # Calculate position update
        position_update = position_info.position - vector
        
        # Apply learning rate
        position_update *= learning_rate
        
        # Update vector
        updated_vector = vector + position_update
        
        # Normalize
        updated_vector = updated_vector / np.linalg.norm(updated_vector)
        
        return updated_vector

print(f"vector_data shape: {vector_data.shape}")
