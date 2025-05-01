""" Influence Calculator Module

This module implements the core influence metrics for the Contexter,
including vector solidness, impact force, impact radius, and impact direction.
These metrics are essential for understanding and quantifying the relationships
between vectors in the contextual space.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from dataclasses import dataclass
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

@dataclass
class VectorMetrics:
    """Container for vector influence metrics."""
    solidness: float
    impact_force: float
    impact_radius: float
    impact_direction: np.ndarray

class InfluenceCalculator:
    """Calculates and manages influence metrics between vectors."""
    
    def __init__(
        self,
        solidness_decay: float = 0.95,
        impact_force_threshold: float = 0.1,
        min_impact_radius: float = 0.01
    ):
        """Initialize the influence calculator.
        
        Args:
            solidness_decay: Rate at which solidness decays over time
            impact_force_threshold: Minimum force required for influence
            min_impact_radius: Minimum radius for vector impact
        """
        self.solidness_decay = solidness_decay
        self.impact_force_threshold = impact_force_threshold
        self.min_impact_radius = min_impact_radius
        self.metrics_cache: Dict[str, VectorMetrics] = {}
        
    def calculate_metrics(
        self,
        vector: np.ndarray,
        metadata: Dict[str, Any],
        context_vectors: List[Tuple[str, np.ndarray]]
    ) -> VectorMetrics:
        """Calculate all influence metrics for a vector.
        
        Args:
            vector: The vector to calculate metrics for
            metadata: Vector metadata
            context_vectors: List of (vector_id, vector) tuples in context
            
        Returns:
            VectorMetrics object containing all calculated metrics
        """
        # Calculate solidness based on vector stability and metadata
        solidness = self._calculate_solidness(vector, metadata)
        
        # Calculate impact force based on vector magnitude and context
        impact_force = self._calculate_impact_force(vector, context_vectors)
        
        # Calculate impact radius based on vector distribution
        impact_radius = self._calculate_impact_radius(vector, context_vectors)
        
        # Calculate impact direction based on vector relationships
        impact_direction = self._calculate_impact_direction(vector, context_vectors)
        
        return VectorMetrics(
            solidness=solidness,
            impact_force=impact_force,
            impact_radius=impact_radius,
            impact_direction=impact_direction
        )
        
    def _calculate_solidness(
        self,
        vector: np.ndarray,
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate vector solidness based on stability and metadata.
        
        Args:
            vector: The vector to calculate solidness for
            metadata: Vector metadata
            
        Returns:
            Solidness value between 0 and 1
        """
        # Base solidness on vector magnitude
        magnitude = np.linalg.norm(vector)
        base_solidness = np.tanh(magnitude)  # Normalize to [0,1]
        
        # Adjust based on metadata
        metadata_factor = 1.0
        if 'stability' in metadata:
            metadata_factor = metadata['stability']
        elif 'age' in metadata:
            # Older vectors are more solid
            age = metadata['age']
            metadata_factor = 1.0 - np.exp(-age)
            
        # Apply decay
        solidness = base_solidness * metadata_factor * self.solidness_decay
        
        return np.clip(solidness, 0, 1)
        
    def _calculate_impact_force(
        self,
        vector: np.ndarray,
        context_vectors: List[Tuple[str, np.ndarray]]
    ) -> float:
        """Calculate impact force based on vector magnitude and context.
        
        Args:
            vector: The vector to calculate impact force for
            context_vectors: List of (vector_id, vector) tuples in context
            
        Returns:
            Impact force value between 0 and 1
        """
        if not context_vectors:
            return 0.0
            
        # Calculate average similarity to context vectors
        similarities = []
        for _, context_vec in context_vectors:
            similarity = 1 - cosine(vector, context_vec)
            similarities.append(similarity)
            
        avg_similarity = np.mean(similarities)
        
        # Impact force is based on both magnitude and context similarity
        magnitude = np.linalg.norm(vector)
        impact_force = magnitude * avg_similarity
        
        # Apply threshold
        if impact_force < self.impact_force_threshold:
            return 0.0
            
        return np.clip(impact_force, 0, 1)
        
    def _calculate_impact_radius(
        self,
        vector: np.ndarray,
        context_vectors: List[Tuple[str, np.ndarray]]
    ) -> float:
        """Calculate impact radius based on vector distribution.
        
        Args:
            vector: The vector to calculate impact radius for
            context_vectors: List of (vector_id, vector) tuples in context
            
        Returns:
            Impact radius value between min_impact_radius and 1
        """
        if not context_vectors:
            return self.min_impact_radius
            
        # Calculate distances to context vectors
        distances = []
        for _, context_vec in context_vectors:
            distance = np.linalg.norm(vector - context_vec)
            distances.append(distance)
            
        # Impact radius is inversely proportional to average distance
        avg_distance = np.mean(distances)
        impact_radius = 1.0 / (1.0 + avg_distance)
        
        return np.clip(impact_radius, self.min_impact_radius, 1.0)
        
    def _calculate_impact_direction(
        self,
        vector: np.ndarray,
        context_vectors: List[Tuple[str, np.ndarray]]
    ) -> np.ndarray:
        """Calculate impact direction based on vector relationships.
        
        Args:
            vector: The vector to calculate impact direction for
            context_vectors: List of (vector_id, vector) tuples in context
            
        Returns:
            Normalized impact direction vector
        """
        if not context_vectors:
            return np.zeros_like(vector)
            
        # Calculate weighted sum of direction vectors
        direction = np.zeros_like(vector)
        total_weight = 0.0
        
        for _, context_vec in context_vectors:
            # Direction is the difference vector
            diff = context_vec - vector
            # Weight by similarity
            weight = 1 - cosine(vector, context_vec)
            direction += weight * diff
            total_weight += weight
            
        if total_weight > 0:
            direction /= total_weight
            
        # Normalize the direction vector
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
            
        return direction
        
    def update_metrics(
        self,
        vector_id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any],
        context_vectors: List[Tuple[str, np.ndarray]]
    ) -> VectorMetrics:
        """Update and cache metrics for a vector.
        
        Args:
            vector_id: ID of the vector
            vector: The vector to update metrics for
            metadata: Vector metadata
            context_vectors: List of (vector_id, vector) tuples in context
            
        Returns:
            Updated VectorMetrics
        """
        metrics = self.calculate_metrics(vector, metadata, context_vectors)
        self.metrics_cache[vector_id] = metrics
        return metrics
        
    def get_metrics(self, vector_id: str) -> VectorMetrics:
        """Get cached metrics for a vector.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            Cached VectorMetrics or None if not found
        """
        return self.metrics_cache.get(vector_id)
        
    def clear_cache(self):
        """Clear the metrics cache."""
        self.metrics_cache.clear()

# example usage 