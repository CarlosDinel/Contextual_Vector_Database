"""
Relationship Decay implementation for DaPGRaC.

This component implements the Relationship Decay mechanism,
which ensures that obsolete connections gradually weaken over time.
"""

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Union, Optional


class RelationshipDecay:
    """
    Implements Relationship Decay calculations.
    
    Relationship Decay introduces a time-based weakening mechanism,
    ensuring that infrequently used relationships gradually disappear.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the RelationshipDecay calculator.
        
        Args:
            config: Optional configuration dictionary with parameters.
        """
        self.config = config or {}
        self.default_decay_rate = self.config.get('decay_rate', 0.1)
    
    def calculate(self, 
                 relationship_strength: float, 
                 decay_rate: Optional[float] = None, 
                 time_delta: float = 1.0) -> float:
        """
        Calculate relationship decay over time.
        
        Implements the formula:
        R_strength(t) = R_strength(t-1) * e^-β
        
        Args:
            relationship_strength: Current relationship strength R_strength(t-1).
            decay_rate: Decay rate β. If None, uses default_decay_rate.
            time_delta: Time elapsed since last update.
            
        Returns:
            The decayed relationship strength.
        """
        # Use default decay rate if not provided
        if decay_rate is None:
            decay_rate = self.default_decay_rate
        
        # Apply decay formula
        decayed_strength = relationship_strength * math.exp(-decay_rate * time_delta)
        
        return decayed_strength
    
    def calculate_batch(self, 
                       relationship_strengths: List[float], 
                       decay_rates: Optional[List[float]] = None, 
                       time_deltas: Optional[List[float]] = None) -> List[float]:
        """
        Calculate relationship decay for multiple relationships in batch.
        
        Args:
            relationship_strengths: List of current relationship strengths.
            decay_rates: Optional list of decay rates. If None, uses default_decay_rate.
            time_deltas: Optional list of time deltas. If None, uses 1.0.
            
        Returns:
            List of decayed relationship strengths.
        """
        # Set default values if not provided
        if decay_rates is None:
            decay_rates = [self.default_decay_rate] * len(relationship_strengths)
        if time_deltas is None:
            time_deltas = [1.0] * len(relationship_strengths)
        
        # Ensure all lists have the same length
        if not (len(relationship_strengths) == len(decay_rates) == len(time_deltas)):
            raise ValueError("All input lists must have the same length")
        
        # Calculate decayed strengths
        decayed_strengths = []
        for i in range(len(relationship_strengths)):
            decayed_strength = self.calculate(
                relationship_strengths[i],
                decay_rates[i],
                time_deltas[i]
            )
            decayed_strengths.append(decayed_strength)
        
        return decayed_strengths
    
    def calculate_with_relationship(self, relationship: Dict) -> Dict:
        """
        Calculate relationship decay using a relationship dictionary.
        
        Args:
            relationship: Dictionary containing relationship information.
            
        Returns:
            Updated relationship dictionary with decayed strength.
        """
        # Make a copy to avoid modifying the input
        updated_relationship = relationship.copy()
        
        # Extract relationship attributes
        strength = relationship.get('strength', 0.0)
        decay_rate = relationship.get('decay_rate', self.default_decay_rate)
        last_updated = relationship.get('last_updated', time.time())
        
        # Calculate time delta
        current_time = time.time()
        time_delta = (current_time - last_updated) / 3600.0  # Convert to hours
        
        # Calculate decayed strength
        decayed_strength = self.calculate(strength, decay_rate, time_delta)
        
        # Update relationship
        updated_relationship['strength'] = decayed_strength
        updated_relationship['last_updated'] = current_time
        
        return updated_relationship
    
    def is_relationship_significant(self, strength: float, threshold: float = 0.01) -> bool:
        """
        Determine if a relationship is still significant based on strength.
        
        Args:
            strength: The relationship strength.
            threshold: The threshold for significance.
            
        Returns:
            True if relationship is significant, False otherwise.
        """
        return strength >= threshold
    
    def get_half_life(self, decay_rate: float) -> float:
        """
        Calculate the half-life of a relationship based on decay rate.
        
        Args:
            decay_rate: The decay rate β.
            
        Returns:
            The half-life in time units.
        """
        return math.log(2) / decay_rate
