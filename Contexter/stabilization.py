""" Stabilization Module

This module implements the stabilization mechanisms for the Contexter,
including adaptive convergence, energy decay, and negative feedback.
These mechanisms ensure stable and controlled evolution of vector embeddings.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StabilizationMetrics:
    """Container for stabilization metrics."""
    convergence_rate: float
    energy_level: float
    feedback_factor: float

class Stabilizer:
    """Manages stabilization mechanisms for vector evolution."""
    
    def __init__(
        self,
        initial_convergence_rate: float = 0.1,
        energy_decay_rate: float = 0.95,
        feedback_threshold: float = 0.5,
        min_convergence_rate: float = 0.01,
        max_convergence_rate: float = 0.5
    ):
        """Initialize the stabilizer.
        
        Args:
            initial_convergence_rate: Initial rate of convergence
            energy_decay_rate: Rate at which energy decays
            feedback_threshold: Threshold for negative feedback
            min_convergence_rate: Minimum allowed convergence rate
            max_convergence_rate: Maximum allowed convergence rate
        """
        self.initial_convergence_rate = initial_convergence_rate
        self.energy_decay_rate = energy_decay_rate
        self.feedback_threshold = feedback_threshold
        self.min_convergence_rate = min_convergence_rate
        self.max_convergence_rate = max_convergence_rate
        self.energy_history: Dict[str, List[float]] = {}
        self.convergence_history: Dict[str, List[float]] = {}
        
    def calculate_stabilization(
        self,
        vector_id: str,
        current_energy: float,
        iteration: int,
        metadata: Dict[str, Any]
    ) -> StabilizationMetrics:
        """Calculate stabilization metrics for a vector.
        
        Args:
            vector_id: ID of the vector
            current_energy: Current energy level
            iteration: Current iteration number
            metadata: Vector metadata
            
        Returns:
            StabilizationMetrics containing convergence, energy, and feedback factors
        """
        # Update energy history
        if vector_id not in self.energy_history:
            self.energy_history[vector_id] = []
        self.energy_history[vector_id].append(current_energy)
        
        # Calculate adaptive convergence rate
        convergence_rate = self._calculate_convergence_rate(
            vector_id,
            current_energy,
            iteration,
            metadata
        )
        
        # Calculate energy decay
        energy_level = self._calculate_energy_decay(
            vector_id,
            current_energy,
            iteration
        )
        
        # Calculate negative feedback
        feedback_factor = self._calculate_negative_feedback(
            vector_id,
            current_energy,
            metadata
        )
        
        return StabilizationMetrics(
            convergence_rate=convergence_rate,
            energy_level=energy_level,
            feedback_factor=feedback_factor
        )
        
    def _calculate_convergence_rate(
        self,
        vector_id: str,
        current_energy: float,
        iteration: int,
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate adaptive convergence rate.
        
        Args:
            vector_id: ID of the vector
            current_energy: Current energy level
            iteration: Current iteration number
            metadata: Vector metadata
            
        Returns:
            Adaptive convergence rate
        """
        # Base convergence rate
        convergence_rate = self.initial_convergence_rate
        
        # Adjust based on energy history
        if vector_id in self.energy_history:
            energy_history = self.energy_history[vector_id]
            if len(energy_history) > 1:
                # Calculate energy trend
                energy_trend = np.diff(energy_history)
                avg_trend = np.mean(energy_trend)
                
                # Adjust convergence rate based on trend
                if avg_trend < 0:  # Energy decreasing
                    convergence_rate *= 1.1  # Speed up convergence
                else:  # Energy increasing
                    convergence_rate *= 0.9  # Slow down convergence
                    
        # Adjust based on metadata
        if 'stability' in metadata:
            stability = metadata['stability']
            convergence_rate *= stability
            
        # Ensure within bounds
        convergence_rate = np.clip(
            convergence_rate,
            self.min_convergence_rate,
            self.max_convergence_rate
        )
        
        return convergence_rate
        
    def _calculate_energy_decay(
        self,
        vector_id: str,
        current_energy: float,
        iteration: int
    ) -> float:
        """Calculate energy decay.
        
        Args:
            vector_id: ID of the vector
            current_energy: Current energy level
            iteration: Current iteration number
            
        Returns:
            Decayed energy level
        """
        # Apply exponential decay
        decayed_energy = current_energy * (self.energy_decay_rate ** iteration)
        
        # Update energy history
        if vector_id not in self.energy_history:
            self.energy_history[vector_id] = []
        self.energy_history[vector_id].append(decayed_energy)
        
        return decayed_energy
        
    def _calculate_negative_feedback(
        self,
        vector_id: str,
        current_energy: float,
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate negative feedback factor.
        
        Args:
            vector_id: ID of the vector
            current_energy: Current energy level
            metadata: Vector metadata
            
        Returns:
            Negative feedback factor
        """
        # Base feedback factor
        feedback_factor = 1.0
        
        # Check if energy exceeds threshold
        if current_energy > self.feedback_threshold:
            # Apply negative feedback
            feedback_factor = self.feedback_threshold / current_energy
            
        # Adjust based on metadata
        if 'stability' in metadata:
            stability = metadata['stability']
            feedback_factor *= stability
            
        return feedback_factor
        
    def apply_stabilization(
        self,
        vector: np.ndarray,
        update: np.ndarray,
        stabilization_metrics: StabilizationMetrics
    ) -> np.ndarray:
        """Apply stabilization to a vector update.
        
        Args:
            vector: Current vector
            update: Proposed update
            stabilization_metrics: Stabilization metrics
            
        Returns:
            Stabilized update
        """
        # Apply convergence rate
        update *= stabilization_metrics.convergence_rate
        
        # Apply energy decay
        update *= stabilization_metrics.energy_level
        
        # Apply negative feedback
        update *= stabilization_metrics.feedback_factor
        
        # Ensure update doesn't cause vector to leave unit sphere
        new_vector = vector + update
        norm = np.linalg.norm(new_vector)
        if norm > 1.0:
            new_vector /= norm
            
        return new_vector - vector
        
    def clear_history(self):
        """Clear stabilization history."""
        self.energy_history.clear()
        self.convergence_history.clear() 