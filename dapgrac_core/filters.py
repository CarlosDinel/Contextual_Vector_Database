"""Candidate Selection Filters for DaPGRaC

This module implements candidate selection logic based on Relationship Magnetism
and topological filters for the DaPGRaC system.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

from .relationship_magnetism import RelationshipMagnetism
from .base_model import Vector

logger = logging.getLogger(__name__)

@dataclass
class FilterMetrics:
    """Container for filter metrics."""
    magnetism_score: float
    topological_score: float
    combined_score: float
    is_selected: bool

class CandidateFilter:
    """Filters candidate vectors based on RM and topological criteria."""
    
    def __init__(
        self,
        relationship_magnetism: Optional[RelationshipMagnetism] = None,
        magnetism_threshold: float = 0.5,
        topological_threshold: float = 0.3,
        min_degree: int = 2,
        max_degree: int = 10
    ):
        """Initialize the candidate filter.
        
        Args:
            relationship_magnetism: Relationship magnetism calculator
            magnetism_threshold: Threshold for RM-based selection
            topological_threshold: Threshold for topological selection
            min_degree: Minimum degree for topological filtering
            max_degree: Maximum degree for topological filtering
        """
        self.relationship_magnetism = relationship_magnetism or RelationshipMagnetism()
        self.magnetism_threshold = magnetism_threshold
        self.topological_threshold = topological_threshold
        self.min_degree = min_degree
        self.max_degree = max_degree
        
    def filter_candidates(
        self,
        vector: Vector,
        candidates: List[Vector],
        existing_relationships: Dict[str, List[str]]
    ) -> List[Tuple[Vector, FilterMetrics]]:
        """Filter candidate vectors based on RM and topological criteria.
        
        Args:
            vector: Source vector
            candidates: List of candidate vectors
            existing_relationships: Dictionary of existing relationships
            
        Returns:
            List of (candidate, metrics) tuples for selected candidates
        """
        results = []
        
        for candidate in candidates:
            # Calculate relationship magnetism
            magnetism_score = self.relationship_magnetism.calculate(
                vector.data,
                candidate.data
            )
            
            # Calculate topological score
            topological_score = self._calculate_topological_score(
                vector.id,
                candidate.id,
                existing_relationships
            )
            
            # Calculate combined score
            combined_score = self._calculate_combined_score(
                magnetism_score,
                topological_score
            )
            
            # Determine if selected
            is_selected = self._is_candidate_selected(
                magnetism_score,
                topological_score,
                combined_score
            )
            
            # Create metrics
            metrics = FilterMetrics(
                magnetism_score=magnetism_score,
                topological_score=topological_score,
                combined_score=combined_score,
                is_selected=is_selected
            )
            
            results.append((candidate, metrics))
            
        return results
        
    def _calculate_topological_score(
        self,
        vector_id: str,
        candidate_id: str,
        existing_relationships: Dict[str, List[str]]
    ) -> float:
        """Calculate topological score for a candidate.
        
        Args:
            vector_id: ID of source vector
            candidate_id: ID of candidate vector
            existing_relationships: Dictionary of existing relationships
            
        Returns:
            Topological score between 0 and 1
        """
        # Get degrees
        vector_degree = len(existing_relationships.get(vector_id, []))
        candidate_degree = len(existing_relationships.get(candidate_id, []))
        
        # Check degree bounds
        if (vector_degree < self.min_degree or vector_degree > self.max_degree or
            candidate_degree < self.min_degree or candidate_degree > self.max_degree):
            return 0.0
            
        # Calculate degree similarity
        degree_similarity = 1.0 - abs(vector_degree - candidate_degree) / self.max_degree
        
        # Check for existing relationship
        has_relationship = (
            candidate_id in existing_relationships.get(vector_id, []) or
            vector_id in existing_relationships.get(candidate_id, [])
        )
        
        # Calculate topological score
        if has_relationship:
            # Existing relationships get a boost
            score = 0.7 + 0.3 * degree_similarity
        else:
            # New relationships based on degree similarity
            score = 0.3 + 0.7 * degree_similarity
            
        return score
        
    def _calculate_combined_score(
        self,
        magnetism_score: float,
        topological_score: float
    ) -> float:
        """Calculate combined selection score.
        
        Args:
            magnetism_score: RM-based score
            topological_score: Topological score
            
        Returns:
            Combined score between 0 and 1
        """
        # Weighted combination
        combined_score = (
            0.6 * magnetism_score +  # RM is primary factor
            0.4 * topological_score  # Topology is secondary factor
        )
        
        return combined_score
        
    def _is_candidate_selected(
        self,
        magnetism_score: float,
        topological_score: float,
        combined_score: float
    ) -> bool:
        """Determine if a candidate should be selected.
        
        Args:
            magnetism_score: RM-based score
            topological_score: Topological score
            combined_score: Combined score
            
        Returns:
            True if candidate should be selected
        """
        # Must meet minimum thresholds
        if (magnetism_score < self.magnetism_threshold or
            topological_score < self.topological_threshold):
            return False
            
        # Combined score must be above average
        return combined_score > 0.5 