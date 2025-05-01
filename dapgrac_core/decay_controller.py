"""Decay Controller Module for DaPGRaC

This module implements long-term relationship maintenance routines,
including relationship strength tracking and decay management.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from .base_model import Vector
from .relationship_executor import RelationshipExecutor, RelationshipMetadata

logger = logging.getLogger(__name__)

@dataclass
class DecayMetrics:
    """Container for decay metrics."""
    current_strength: float
    decay_rate: float
    time_since_update: timedelta
    is_active: bool

class DecayController:
    """Manages relationship decay and long-term maintenance."""
    
    def __init__(
        self,
        relationship_executor: Optional[RelationshipExecutor] = None,
        base_decay_rate: float = 0.01,
        min_strength: float = 0.1,
        max_time_since_update: timedelta = timedelta(days=30),
        strength_update_interval: timedelta = timedelta(days=1)
    ):
        """Initialize the decay controller.
        
        Args:
            relationship_executor: Relationship executor instance
            base_decay_rate: Base rate of relationship decay
            min_strength: Minimum strength for active relationships
            max_time_since_update: Maximum time since last update
            strength_update_interval: Interval for strength updates
        """
        self.relationship_executor = relationship_executor or RelationshipExecutor()
        self.base_decay_rate = base_decay_rate
        self.min_strength = min_strength
        self.max_time_since_update = max_time_since_update
        self.strength_update_interval = strength_update_interval
        self.last_strength_update: Dict[Tuple[str, str], datetime] = {}
        
    def update_relationship_strengths(self) -> List[Tuple[str, str]]:
        """Update strengths of all relationships.
        
        Returns:
            List of relationship keys that were updated
        """
        updated_relationships = []
        current_time = datetime.now()
        
        for key, metadata in self.relationship_executor.relationships.items():
            # Calculate decay metrics
            metrics = self._calculate_decay_metrics(
                metadata,
                current_time
            )
            
            # Update strength if needed
            if self._should_update_strength(metrics):
                new_strength = self._calculate_new_strength(metrics)
                
                # Update relationship
                self.relationship_executor.update_relationship(
                    key[0],
                    key[1],
                    new_strength
                )
                
                # Update timestamp
                self.last_strength_update[key] = current_time
                
                updated_relationships.append(key)
                
        return updated_relationships
        
    def _calculate_decay_metrics(
        self,
        metadata: RelationshipMetadata,
        current_time: datetime
    ) -> DecayMetrics:
        """Calculate decay metrics for a relationship.
        
        Args:
            metadata: Relationship metadata
            current_time: Current time
            
        Returns:
            DecayMetrics object
        """
        # Calculate time since last update
        time_since_update = current_time - metadata.last_updated
        
        # Calculate decay rate based on various factors
        decay_rate = self._calculate_decay_rate(
            metadata,
            time_since_update
        )
        
        # Determine if relationship is active
        is_active = self._is_relationship_active(
            metadata.strength,
            time_since_update
        )
        
        return DecayMetrics(
            current_strength=metadata.strength,
            decay_rate=decay_rate,
            time_since_update=time_since_update,
            is_active=is_active
        )
        
    def _calculate_decay_rate(
        self,
        metadata: RelationshipMetadata,
        time_since_update: timedelta
    ) -> float:
        """Calculate decay rate for a relationship.
        
        Args:
            metadata: Relationship metadata
            time_since_update: Time since last update
            
        Returns:
            Decay rate between 0 and 1
        """
        # Start with base decay rate
        decay_rate = self.base_decay_rate
        
        # Adjust based on relationship strength
        decay_rate *= (1.0 - metadata.strength)
        
        # Adjust based on time since update
        time_factor = min(
            1.0,
            time_since_update.total_seconds() / self.max_time_since_update.total_seconds()
        )
        decay_rate *= (1.0 + time_factor)
        
        # Adjust based on anchoring
        if metadata.is_anchored:
            decay_rate *= 0.5  # Anchored relationships decay slower
            
        return decay_rate
        
    def _is_relationship_active(
        self,
        strength: float,
        time_since_update: timedelta
    ) -> bool:
        """Determine if a relationship is still active.
        
        Args:
            strength: Current relationship strength
            time_since_update: Time since last update
            
        Returns:
            True if relationship is active
        """
        # Check strength threshold
        if strength < self.min_strength:
            return False
            
        # Check time threshold
        if time_since_update > self.max_time_since_update:
            return False
            
        return True
        
    def _should_update_strength(
        self,
        metrics: DecayMetrics
    ) -> bool:
        """Determine if relationship strength should be updated.
        
        Args:
            metrics: Decay metrics
            
        Returns:
            True if strength should be updated
        """
        # Don't update inactive relationships
        if not metrics.is_active:
            return False
            
        # Update if enough time has passed
        return metrics.time_since_update >= self.strength_update_interval
        
    def _calculate_new_strength(
        self,
        metrics: DecayMetrics
    ) -> float:
        """Calculate new relationship strength.
        
        Args:
            metrics: Decay metrics
            
        Returns:
            New strength value
        """
        # Apply decay
        new_strength = metrics.current_strength * (1.0 - metrics.decay_rate)
        
        # Ensure within bounds
        return max(new_strength, self.min_strength)
        
    def cleanup_inactive_relationships(self) -> List[Tuple[str, str]]:
        """Remove inactive relationships.
        
        Returns:
            List of removed relationship keys
        """
        removed_relationships = []
        current_time = datetime.now()
        
        for key, metadata in list(self.relationship_executor.relationships.items()):
            # Calculate decay metrics
            metrics = self._calculate_decay_metrics(
                metadata,
                current_time
            )
            
            # Remove if inactive
            if not metrics.is_active:
                self.relationship_executor.remove_relationship(key[0], key[1])
                removed_relationships.append(key)
                
        return removed_relationships 