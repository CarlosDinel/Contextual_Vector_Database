"""
Relationship Magnetism implementation for DaPGRaC.

This component implements the Relationship Magnetism mechanism,
which determines the inherent attraction between vectors based on shared contextual properties.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Union, Optional


class RelationshipMagnetism:
    """
    Implements Relationship Magnetism calculations.
    
    Relationship Magnetism determines how strongly two vectors attract each other
    based on their shared characteristics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the RelationshipMagnetism calculator.
        
        Args:
            config: Optional configuration dictionary with parameters.
        """
        self.config = config or {}
        self.epsilon = self.config.get('epsilon', 1e-8)
    
    def calculate(self, 
                 vector_i: np.ndarray, 
                 vector_j: np.ndarray, 
                 weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate relationship magnetism between two vectors.
        
        Implements the formula:
        M_ij = Σ(α_k|V_i,k - V_j,k|)
        
        Args:
            vector_i: The first vector.
            vector_j: The second vector.
            weights: Optional weights for vector dimensions (α_k).
            
        Returns:
            The calculated relationship magnetism.
        """
        # Ensure vectors have the same dimension
        if vector_i.shape != vector_j.shape:
            raise ValueError("Vectors must have the same dimension")
        
        # Use default weights if not provided
        if weights is None:
            weights = np.ones(vector_i.shape)
        elif weights.shape != vector_i.shape:
            raise ValueError("Weights must have the same dimension as vectors")
        
        # Calculate absolute differences
        abs_diff = np.abs(vector_i - vector_j)
        
        # Apply weights
        weighted_diff = weights * abs_diff
        
        # Sum over all dimensions
        magnetism = np.sum(weighted_diff)
        
        # Normalize by vector dimension
        magnetism /= len(vector_i)
        
        # Invert to make similar vectors have higher magnetism
        # Add epsilon to avoid division by zero
        magnetism = 1.0 / (magnetism + self.epsilon)
        
        return magnetism
    
    def calculate_batch(self, 
                       vectors_i: List[np.ndarray], 
                       vectors_j: List[np.ndarray], 
                       weights: Optional[List[np.ndarray]] = None) -> List[float]:
        """
        Calculate relationship magnetism for multiple vector pairs in batch.
        
        Args:
            vectors_i: List of first vectors.
            vectors_j: List of second vectors.
            weights: Optional list of weights for vector dimensions.
            
        Returns:
            List of calculated relationship magnetism values.
        """
        if len(vectors_i) != len(vectors_j):
            raise ValueError("Input lists must have the same length")
        
        if weights is not None and len(weights) != len(vectors_i):
            raise ValueError("Weights list must have the same length as vectors lists")
        
        magnetisms = []
        for i in range(len(vectors_i)):
            w = weights[i] if weights is not None else None
            magnetism = self.calculate(vectors_i[i], vectors_j[i], w)
            magnetisms.append(magnetism)
        
        return magnetisms
    
    def calculate_with_vectors(self, vector_i: Dict, vector_j: Dict) -> float:
        """
        Calculate relationship magnetism using vector dictionaries.
        
        Args:
            vector_i: Dictionary for the first vector.
            vector_j: Dictionary for the second vector.
            
        Returns:
            The calculated relationship magnetism.
        """
        # Extract vector data
        data_i = np.array(vector_i['data'])
        data_j = np.array(vector_j['data'])
        
        # Extract weights if available
        weights = None
        if 'dimension_weights' in vector_i:
            weights = np.array(vector_i['dimension_weights'])
        
        # Calculate magnetism
        magnetism = self.calculate(data_i, data_j, weights)
        
        return magnetism
    
    def is_attraction_significant(self, magnetism: float, threshold: float = 0.5) -> bool:
        """
        Determine if attraction between vectors is significant based on magnetism.
        
        Args:
            magnetism: The calculated magnetism value.
            threshold: The threshold for significant attraction.
            
        Returns:
            True if attraction is significant, False otherwise.
        """
        return magnetism >= threshold
