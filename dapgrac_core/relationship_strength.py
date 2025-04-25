"""
Relationship Strength implementation for DaPGRaC.

This component implements the Relationship Strength mechanism,
which reinforces meaningful connections that are actively used.
"""

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Union, Optional


class RelationshipStrength:
    """
    Implements Relationship Strength calculations.
    
    Relationship Strength ensures that actively used connections persist
    and grow stronger over time.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the RelationshipStrength calculator.
        
        Args:
            config: Optional configuration dictionary with parameters.
        """
        self.config = config or {}
        self.default_lambda = self.config.get('lambda', 0.5)
    
    def calculate(self, 
                 relationship_strength: float, 
                 sev: float, 
                 lambda_param: Optional[float] = None) -> float:
        """
        Calculate relationship strength.
        
        Implements the formula:
        R_strength(t+1) = λ*R_strength(t) + (1-λ)*SEV_ij
        
        Args:
            relationship_strength: Current relationship strength R_strength(t).
            sev: Shooting Edge Velocity SEV_ij.
            lambda_param: History influence factor λ. If None, uses default_lambda.
            
        Returns:
            The updated relationship strength.
        """
        # Use default lambda if not provided
        if lambda_param is None:
            lambda_param = self.default_lambda
        
        # Apply strength formula
        updated_strength = lambda_param * relationship_strength + (1 - lambda_param) * sev
        
        return updated_strength
    
    def calculate_batch(self, 
                       relationship_strengths: List[float], 
                       sevs: List[float], 
                       lambda_params: Optional[List[float]] = None) -> List[float]:
        """
        Calculate relationship strength for multiple relationships in batch.
        
        Args:
            relationship_strengths: List of current relationship strengths.
            sevs: List of Shooting Edge Velocities.
            lambda_params: Optional list of lambda parameters. If None, uses default_lambda.
            
        Returns:
            List of updated relationship strengths.
        """
        # Set default values if not provided
        if lambda_params is None:
            lambda_params = [self.default_lambda] * len(relationship_strengths)
        
        # Ensure all lists have the same length
        if not (len(relationship_strengths) == len(sevs) == len(lambda_params)):
            raise ValueError("All input lists must have the same length")
        
        # Calculate updated strengths
        updated_strengths = []
        for i in range(len(relationship_strengths)):
            updated_strength = self.calculate(
                relationship_strengths[i],
                sevs[i],
                lambda_params[i]
            )
            updated_strengths.append(updated_strength)
        
        return updated_strengths
    
    def calculate_with_relationship(self, 
                                   relationship: Dict, 
                                   sev: float) -> Dict:
        """
        Calculate relationship strength using a relationship dictionary.
        
        Args:
            relationship: Dictionary containing relationship information.
            sev: Shooting Edge Velocity.
            
        Returns:
            Updated relationship dictionary with new strength.
        """
        # Make a copy to avoid modifying the input
        updated_relationship = relationship.copy()
        
        # Extract relationship attributes
        strength = relationship.get('strength', 0.0)
        lambda_param = relationship.get('lambda', self.default_lambda)
        
        # Calculate updated strength
        updated_strength = self.calculate(strength, sev, lambda_param)
        
        # Update relationship
        updated_relationship['strength'] = updated_strength
        updated_relationship['last_updated'] = time.time()
        
        return updated_relationship
    
    def calculate_with_history(self, 
                              relationship_strengths: List[float], 
                              sev: float, 
                              window_size: int = 5, 
                              lambda_param: Optional[float] = None) -> float:
        """
        Calculate relationship strength considering historical values.
        
        Args:
            relationship_strengths: List of historical relationship strengths.
            sev: Current Shooting Edge Velocity.
            window_size: Number of historical values to consider.
            lambda_param: History influence factor λ. If None, uses default_lambda.
            
        Returns:
            The updated relationship strength.
        """
        # Use default lambda if not provided
        if lambda_param is None:
            lambda_param = self.default_lambda
        
        # Use only the most recent values up to window_size
        recent_strengths = relationship_strengths[-window_size:] if len(relationship_strengths) > 0 else [0.0]
        
        # Calculate average historical strength
        avg_strength = sum(recent_strengths) / len(recent_strengths)
        
        # Apply strength formula with average historical strength
        updated_strength = lambda_param * avg_strength + (1 - lambda_param) * sev
        
        return updated_strength
    
    def is_relationship_strong(self, strength: float, threshold: float = 0.5) -> bool:
        """
        Determine if a relationship is strong based on strength.
        
        Args:
            strength: The relationship strength.
            threshold: The threshold for strong relationships.
            
        Returns:
            True if relationship is strong, False otherwise.
        """
        return strength >= threshold
