"""Relationship Executor Module for DaPGRaC

This module implements relationship anchoring, tagging, and metadata management
for the DaPGRaC system.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
from dataclasses import dataclass
from datetime import datetime

from .base_model import Vector
from .edge_dynamics import EdgeDynamicsManager, EdgeDynamics

logger = logging.getLogger(__name__)

@dataclass
class RelationshipMetadata:
    """Container for relationship metadata."""
    created_at: datetime
    last_updated: datetime
    strength: float
    tags: Set[str]
    anchor_points: List[np.ndarray]
    is_anchored: bool

class RelationshipExecutor:
    """Manages relationship creation, anchoring, and metadata."""
    
    def __init__(
        self,
        edge_dynamics_manager: Optional[EdgeDynamicsManager] = None,
        min_strength: float = 0.3,
        max_anchor_points: int = 5
    ):
        """Initialize the relationship executor.
        
        Args:
            edge_dynamics_manager: Edge dynamics manager
            min_strength: Minimum strength for relationships
            max_anchor_points: Maximum number of anchor points
        """
        self.edge_dynamics_manager = edge_dynamics_manager or EdgeDynamicsManager()
        self.min_strength = min_strength
        self.max_anchor_points = max_anchor_points
        self.relationships: Dict[Tuple[str, str], RelationshipMetadata] = {}
        
    def create_relationship(
        self,
        vector_i: Vector,
        vector_j: Vector,
        segmental_context: List[Vector],
        global_context: List[Vector],
        tags: Optional[Set[str]] = None
    ) -> Optional[Tuple[str, str]]:
        """Create a new relationship between vectors.
        
        Args:
            vector_i: First vector
            vector_j: Second vector
            segmental_context: Vectors in segmental context
            global_context: Vectors in global context
            tags: Optional set of relationship tags
            
        Returns:
            Tuple of vector IDs if relationship created, None otherwise
        """
        # Calculate edge dynamics
        dynamics = self.edge_dynamics_manager.calculate_edge_dynamics(
            vector_i,
            vector_j,
            segmental_context,
            global_context
        )
        
        # Check if relationship should be created
        if not dynamics.is_active:
            return None
            
        # Create relationship key
        key = (vector_i.id, vector_j.id)
        
        # Create metadata
        metadata = RelationshipMetadata(
            created_at=datetime.now(),
            last_updated=datetime.now(),
            strength=dynamics.segmental_strength,
            tags=tags or set(),
            anchor_points=[],
            is_anchored=False
        )
        
        # Store relationship
        self.relationships[key] = metadata
        
        return key
        
    def update_relationship(
        self,
        vector_i_id: str,
        vector_j_id: str,
        new_strength: float,
        new_tags: Optional[Set[str]] = None
    ) -> bool:
        """Update an existing relationship.
        
        Args:
            vector_i_id: ID of first vector
            vector_j_id: ID of second vector
            new_strength: New relationship strength
            new_tags: Optional new tags
            
        Returns:
            True if relationship was updated
        """
        key = (vector_i_id, vector_j_id)
        if key not in self.relationships:
            return False
            
        metadata = self.relationships[key]
        
        # Update strength
        metadata.strength = new_strength
        
        # Update tags if provided
        if new_tags is not None:
            metadata.tags.update(new_tags)
            
        # Update timestamp
        metadata.last_updated = datetime.now()
        
        return True
        
    def add_anchor_point(
        self,
        vector_i_id: str,
        vector_j_id: str,
        anchor_point: np.ndarray
    ) -> bool:
        """Add an anchor point to a relationship.
        
        Args:
            vector_i_id: ID of first vector
            vector_j_id: ID of second vector
            anchor_point: New anchor point
            
        Returns:
            True if anchor point was added
        """
        key = (vector_i_id, vector_j_id)
        if key not in self.relationships:
            return False
            
        metadata = self.relationships[key]
        
        # Check if we can add more anchor points
        if len(metadata.anchor_points) >= self.max_anchor_points:
            return False
            
        # Add anchor point
        metadata.anchor_points.append(anchor_point)
        metadata.is_anchored = True
        
        return True
        
    def get_relationship_metadata(
        self,
        vector_i_id: str,
        vector_j_id: str
    ) -> Optional[RelationshipMetadata]:
        """Get metadata for a relationship.
        
        Args:
            vector_i_id: ID of first vector
            vector_j_id: ID of second vector
            
        Returns:
            RelationshipMetadata if relationship exists
        """
        key = (vector_i_id, vector_j_id)
        return self.relationships.get(key)
        
    def get_relationships_by_tag(
        self,
        tag: str
    ) -> List[Tuple[str, str]]:
        """Get all relationships with a specific tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of relationship keys
        """
        return [
            key for key, metadata in self.relationships.items()
            if tag in metadata.tags
        ]
        
    def get_anchored_relationships(
        self
    ) -> List[Tuple[str, str]]:
        """Get all anchored relationships.
        
        Returns:
            List of relationship keys
        """
        return [
            key for key, metadata in self.relationships.items()
            if metadata.is_anchored
        ]
        
    def remove_relationship(
        self,
        vector_i_id: str,
        vector_j_id: str
    ) -> bool:
        """Remove a relationship.
        
        Args:
            vector_i_id: ID of first vector
            vector_j_id: ID of second vector
            
        Returns:
            True if relationship was removed
        """
        key = (vector_i_id, vector_j_id)
        if key in self.relationships:
            del self.relationships[key]
            return True
        return False 