"""
Shooting Edge Velocity (SEV) implementation for DaPGRaC.

This component implements the Shooting Edge Velocity (SEV) mechanism,
which controls the rate at which relationships are established between vectors.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Union, Optional


class ShootingEdgeVelocity:
    """
    Implements Shooting Edge Velocity (SEV) calculations.
    
    SEV determines the speed at which a relationship is established between
    two vectors in an evolving knowledge space.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ShootingEdgeVelocity calculator.
        
        Args:
            config: Optional configuration dictionary with parameters.
        """
        self.config = config or {}
    
    def calculate(self, 
                 vector_i: np.ndarray, 
                 vector_j: np.ndarray, 
                 magnetism: float, 
                 distance: float, 
                 impact_radius: float) -> float:
        """
        Calculate SEV between two vectors.
        
        Implements the formula:
        SEV_ij = M_ij * e^(-D_ij/R_i)
        
        Args:
            vector_i: The first vector.
            vector_j: The second vector.
            magnetism: The relationship magnetism M_ij.
            distance: The distance D_ij between vectors.
            impact_radius: The impact radius R_i.
            
        Returns:
            The calculated SEV value.
        """
        # Calculate exponential decay term
        decay_term = math.exp(-distance / impact_radius)
        
        # Calculate SEV
        sev = magnetism * decay_term
        
        return sev
    
    def calculate_batch(self, 
                       vectors_i: List[np.ndarray], 
                       vectors_j: List[np.ndarray], 
                       magnetisms: List[float], 
                       distances: List[float], 
                       impact_radii: List[float]) -> List[float]:
        """
        Calculate SEV for multiple vector pairs in batch.
        
        Args:
            vectors_i: List of first vectors.
            vectors_j: List of second vectors.
            magnetisms: List of relationship magnetism values.
            distances: List of distances between vectors.
            impact_radii: List of impact radii.
            
        Returns:
            List of calculated SEV values.
        """
        if not (len(vectors_i) == len(vectors_j) == len(magnetisms) == len(distances) == len(impact_radii)):
            raise ValueError("All input lists must have the same length")
        
        sevs = []
        for i in range(len(vectors_i)):
            sev = self.calculate(
                vectors_i[i],
                vectors_j[i],
                magnetisms[i],
                distances[i],
                impact_radii[i]
            )
            sevs.append(sev)
        
        return sevs
    
    def calculate_with_vectors(self, 
                              vector_i: Dict, 
                              vector_j: Dict, 
                              magnetism_calculator) -> float:
        """
        Calculate SEV using vector dictionaries and a magnetism calculator.
        
        Args:
            vector_i: Dictionary for the first vector.
            vector_j: Dictionary for the second vector.
            magnetism_calculator: RelationshipMagnetism calculator instance.
            
        Returns:
            The calculated SEV value.
        """
        # Extract vector data
        data_i = np.array(vector_i['data'])
        data_j = np.array(vector_j['data'])
        
        # Calculate distance
        distance = np.linalg.norm(data_i - data_j)
        
        # Get impact radius
        impact_radius = vector_i.get('impact_radius', 1.0)
        
        # Calculate magnetism
        magnetism = magnetism_calculator.calculate(data_i, data_j)
        
        # Calculate SEV
        sev = self.calculate(data_i, data_j, magnetism, distance, impact_radius)
        
        return sev
    
    def is_relationship_forming(self, sev: float, threshold: float = 0.1) -> bool:
        """
        Determine if a relationship should form based on SEV.
        
        Args:
            sev: The calculated SEV value.
            threshold: The threshold for relationship formation.
            
        Returns:
            True if relationship should form, False otherwise.
        """
        return sev >= threshold
