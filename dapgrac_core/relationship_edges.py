"""
Relationship Edges implementation for DaPGRaC.

This component implements the Segmental & Global Relationship Edges mechanism,
which differentiates between relationships that apply to entire vectors versus
those that are limited to specific vector segments.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Union, Optional


class RelationshipEdges:
    """
    Manages Segmental & Global Relationship Edges.
    
    This class differentiates between global relationships (applying to entire vectors)
    and segmental relationships (applying to specific vector segments).
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the RelationshipEdges manager.
        
        Args:
            config: Optional configuration dictionary with parameters.
        """
        self.config = config or {}
        self.default_alpha = self.config.get('alpha', 0.5)  # Weight for global vs segmental
    
    def calculate_global(self, 
                        vector_i: np.ndarray, 
                        vector_j: np.ndarray) -> float:
        """
        Calculate global relationship edge.
        
        Args:
            vector_i: The first vector.
            vector_j: The second vector.
            
        Returns:
            The global relationship edge value.
        """
        # For global relationships, we use cosine similarity
        # Normalize vectors
        norm_i = np.linalg.norm(vector_i)
        norm_j = np.linalg.norm(vector_j)
        
        # Prevent division by zero
        if norm_i < 1e-8 or norm_j < 1e-8:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(vector_i, vector_j) / (norm_i * norm_j)
        
        # Scale to [0, 1] range
        global_edge = (similarity + 1) / 2
        
        return global_edge
    
    def calculate_segmental(self, 
                           vector_i: np.ndarray, 
                           vector_j: np.ndarray, 
                           segment: Union[List[int], np.ndarray, slice]) -> float:
        """
        Calculate segmental relationship edge.
        
        Args:
            vector_i: The first vector.
            vector_j: The second vector.
            segment: The segment indices or slice.
            
        Returns:
            The segmental relationship edge value.
        """
        # Extract segment from vectors
        segment_i = vector_i[segment]
        segment_j = vector_j[segment]
        
        # Calculate segmental relationship using cosine similarity
        # Normalize segments
        norm_i = np.linalg.norm(segment_i)
        norm_j = np.linalg.norm(segment_j)
        
        # Prevent division by zero
        if norm_i < 1e-8 or norm_j < 1e-8:
            return 0.0
        
        # Calculate cosine similarity for segment
        similarity = np.dot(segment_i, segment_j) / (norm_i * norm_j)
        
        # Scale to [0, 1] range
        segmental_edge = (similarity + 1) / 2
        
        return segmental_edge
    
    def combine_edges(self, 
                     global_edge: float, 
                     segmental_edges: Dict[str, float], 
                     segment_weights: Optional[Dict[str, float]] = None,
                     alpha: Optional[float] = None) -> float:
        """
        Combine global and segmental edges.
        
        Implements the formula:
        R_edge(V_i,V_j) = α*R_global(V_i,V_j) + (1-α)*Σ(W_s*R_segment(V_i,V_j,S))
        
        Args:
            global_edge: The global relationship edge value.
            segmental_edges: Dictionary mapping segment names to edge values.
            segment_weights: Optional dictionary mapping segment names to weights.
            alpha: Weight for global vs segmental. If None, uses default_alpha.
            
        Returns:
            The combined relationship edge value.
        """
        # Use default alpha if not provided
        if alpha is None:
            alpha = self.default_alpha
        
        # If no segmental edges, return global edge
        if not segmental_edges:
            return global_edge
        
        # Use equal weights if not provided
        if segment_weights is None:
            segment_weights = {segment: 1.0 / len(segmental_edges) for segment in segmental_edges}
        
        # Calculate weighted sum of segmental edges
        weighted_segmental = 0.0
        for segment, edge in segmental_edges.items():
            weight = segment_weights.get(segment, 1.0 / len(segmental_edges))
            weighted_segmental += weight * edge
        
        # Combine global and segmental edges
        combined_edge = alpha * global_edge + (1 - alpha) * weighted_segmental
        
        return combined_edge
    
    def calculate_with_vectors(self, 
                              vector_i: Dict, 
                              vector_j: Dict, 
                              segments: Optional[Dict[str, Union[List[int], slice]]] = None,
                              segment_weights: Optional[Dict[str, float]] = None,
                              alpha: Optional[float] = None) -> Dict:
        """
        Calculate relationship edges using vector dictionaries.
        
        Args:
            vector_i: Dictionary for the first vector.
            vector_j: Dictionary for the second vector.
            segments: Optional dictionary mapping segment names to indices or slices.
            segment_weights: Optional dictionary mapping segment names to weights.
            alpha: Weight for global vs segmental. If None, uses default_alpha.
            
        Returns:
            Dictionary with global, segmental, and combined edge values.
        """
        # Extract vector data
        data_i = np.array(vector_i['data'])
        data_j = np.array(vector_j['data'])
        
        # Calculate global edge
        global_edge = self.calculate_global(data_i, data_j)
        
        # Calculate segmental edges if segments provided
        segmental_edges = {}
        if segments:
            for segment_name, segment_indices in segments.items():
                segmental_edge = self.calculate_segmental(data_i, data_j, segment_indices)
                segmental_edges[segment_name] = segmental_edge
        
        # Combine edges
        combined_edge = self.combine_edges(global_edge, segmental_edges, segment_weights, alpha)
        
        # Return results
        return {
            'global_edge': global_edge,
            'segmental_edges': segmental_edges,
            'combined_edge': combined_edge
        }
    
    def identify_segments(self, 
                         vector: np.ndarray, 
                         num_segments: int = 3) -> Dict[str, slice]:
        """
        Automatically identify segments in a vector.
        
        Args:
            vector: The vector to segment.
            num_segments: Number of segments to create.
            
        Returns:
            Dictionary mapping segment names to slices.
        """
        # Calculate segment size
        vector_length = len(vector)
        segment_size = vector_length // num_segments
        
        # Create segments
        segments = {}
        for i in range(num_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < num_segments - 1 else vector_length
            segment_name = f"segment_{i+1}"
            segments[segment_name] = slice(start, end)
        
        return segments
    
    def is_identity_match(self, vector_i: Dict, vector_j: Dict) -> bool:
        """
        Check if two vectors have matching identity.
        
        Args:
            vector_i: Dictionary for the first vector.
            vector_j: Dictionary for the second vector.
            
        Returns:
            True if identity matches, False otherwise.
        """
        # Check if vectors have identity information
        if 'identity' not in vector_i or 'identity' not in vector_j:
            return False
        
        # Compare identities
        return vector_i['identity'] == vector_j['identity']
