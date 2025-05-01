"""Edge Dynamics Module for DaPGRaC

This module implements the Shooting Edge Velocity (SEV) mechanism and
segmental/global relationship logic for the DaPGRaC system.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass

from .base_model import Vector
from .relationship_magnetism import RelationshipMagnetism

logger = logging.getLogger(__name__)

@dataclass
class EdgeDynamics:
    """Container for edge dynamics metrics."""
    velocity: float
    direction: np.ndarray
    segmental_strength: float
    global_strength: float
    is_active: bool

class EdgeDynamicsManager:
    """Manages edge dynamics including SEV and relationship logic."""
    
    def __init__(
        self,
        relationship_magnetism: Optional[RelationshipMagnetism] = None,
        velocity_threshold: float = 0.3,
        segmental_weight: float = 0.4,
        global_weight: float = 0.6,
        max_velocity: float = 1.0
    ):
        """Initialize the edge dynamics manager.
        
        Args:
            relationship_magnetism: Relationship magnetism calculator
            velocity_threshold: Threshold for active edges
            segmental_weight: Weight for segmental relationships
            global_weight: Weight for global relationships
            max_velocity: Maximum allowed velocity
        """
        self.relationship_magnetism = relationship_magnetism or RelationshipMagnetism()
        self.velocity_threshold = velocity_threshold
        self.segmental_weight = segmental_weight
        self.global_weight = global_weight
        self.max_velocity = max_velocity
        
    def calculate_edge_dynamics(
        self,
        vector_i: Vector,
        vector_j: Vector,
        segmental_context: List[Vector],
        global_context: List[Vector]
    ) -> EdgeDynamics:
        """Calculate edge dynamics between two vectors.
        
        Args:
            vector_i: First vector
            vector_j: Second vector
            segmental_context: Vectors in segmental context
            global_context: Vectors in global context
            
        Returns:
            EdgeDynamics object containing all metrics
        """
        # Calculate SEV
        velocity, direction = self._calculate_sev(vector_i, vector_j)
        
        # Calculate relationship strengths
        segmental_strength = self._calculate_segmental_strength(
            vector_i,
            vector_j,
            segmental_context
        )
        
        global_strength = self._calculate_global_strength(
            vector_i,
            vector_j,
            global_context
        )
        
        # Determine if edge is active
        is_active = self._is_edge_active(
            velocity,
            segmental_strength,
            global_strength
        )
        
        return EdgeDynamics(
            velocity=velocity,
            direction=direction,
            segmental_strength=segmental_strength,
            global_strength=global_strength,
            is_active=is_active
        )
        
    def _calculate_sev(
        self,
        vector_i: Vector,
        vector_j: Vector
    ) -> Tuple[float, np.ndarray]:
        """Calculate Shooting Edge Velocity between vectors.
        
        Args:
            vector_i: First vector
            vector_j: Second vector
            
        Returns:
            Tuple of (velocity, direction)
        """
        # Calculate direction vector
        direction = vector_j.data - vector_i.data
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 0:
            direction = direction / direction_norm
            
        # Calculate velocity based on relationship magnetism
        magnetism = self.relationship_magnetism.calculate(
            vector_i.data,
            vector_j.data
        )
        
        # Velocity is proportional to magnetism but capped
        velocity = min(magnetism, self.max_velocity)
        
        return velocity, direction
        
    def _calculate_segmental_strength(
        self,
        vector_i: Vector,
        vector_j: Vector,
        segmental_context: List[Vector]
    ) -> float:
        """Calculate segmental relationship strength.
        
        Args:
            vector_i: First vector
            vector_j: Second vector
            segmental_context: Vectors in segmental context
            
        Returns:
            Segmental strength between 0 and 1
        """
        if not segmental_context:
            return 0.0
            
        # Calculate average magnetism to context
        context_magnetism = []
        for context_vec in segmental_context:
            magnetism_i = self.relationship_magnetism.calculate(
                vector_i.data,
                context_vec.data
            )
            magnetism_j = self.relationship_magnetism.calculate(
                vector_j.data,
                context_vec.data
            )
            context_magnetism.append((magnetism_i + magnetism_j) / 2)
            
        # Segmental strength is average context magnetism
        return np.mean(context_magnetism)
        
    def _calculate_global_strength(
        self,
        vector_i: Vector,
        vector_j: Vector,
        global_context: List[Vector]
    ) -> float:
        """Calculate global relationship strength.
        
        Args:
            vector_i: First vector
            vector_j: Second vector
            global_context: Vectors in global context
            
        Returns:
            Global strength between 0 and 1
        """
        if not global_context:
            return 0.0
            
        # Calculate relationship magnetism to global context
        global_magnetism = []
        for context_vec in global_context:
            magnetism = self.relationship_magnetism.calculate(
                vector_i.data,
                context_vec.data
            )
            global_magnetism.append(magnetism)
            
        # Global strength is maximum context magnetism
        return np.max(global_magnetism)
        
    def _is_edge_active(
        self,
        velocity: float,
        segmental_strength: float,
        global_strength: float
    ) -> bool:
        """Determine if an edge should be active.
        
        Args:
            velocity: SEV value
            segmental_strength: Segmental relationship strength
            global_strength: Global relationship strength
            
        Returns:
            True if edge should be active
        """
        # Calculate combined strength
        combined_strength = (
            self.segmental_weight * segmental_strength +
            self.global_weight * global_strength
        )
        
        # Edge is active if both velocity and strength meet thresholds
        return (
            velocity >= self.velocity_threshold and
            combined_strength >= 0.5
        ) 