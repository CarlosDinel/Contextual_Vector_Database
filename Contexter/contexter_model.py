""" Contexter Module for Contextual Vector Database

This module implements the contexter component of the Contextual Vector Database (CVD),
which is responsible for determining context around vectors and preparing them for reembedding.
It works together with the ReembedingOrchestrator, context_aggregator, and influence_calculator.

Author: Carlos D. Almeida
"""

import numpy as np
import copy
import time
from typing import List, Dict, Tuple, Optional, Any, Union, TYPE_CHECKING
from scipy.spatial.distance import euclidean, cosine
from scipy.special import softmax
import logging
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import related components
try:
    from .base_model import Vector
    from .context_aggregator import ContextAggregator
    from .influence_calculator import InfluenceCalculator
    if TYPE_CHECKING:
        from .reembedding_orchestra import ReembeddingOrchestra
except ImportError:
    from Contexter.base_model import Vector
from Contexter.context_aggregator import ContextAggregator
    from Contexter.influence_calculator import InfluenceCalculator
    if TYPE_CHECKING:
        from Contexter.reembedding_orchestra import ReembeddingOrchestra

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Contexter:
    """
    Multi-headed contextual processing engine that evaluates vector positioning
    based on similarity, impact, and influence metrics.
    """
    def __init__(self, 
                 influence_calculator: Optional['InfluenceCalculator'] = None,
                 context_aggregator: Optional[ContextAggregator] = None,
                 learning_rate: float = 0.1,
                 max_iterations: int = 10,
                 convergence_threshold: float = 0.001,
                 num_heads: int = 8):
        """
        Initialize the Contexter with multi-headed attention capabilities.
        
        Args:
            influence_calculator: Calculator for contextual influence
            context_aggregator: Aggregator for context information
            learning_rate: Learning rate for position updates
            max_iterations: Maximum number of iterations for convergence
            convergence_threshold: Threshold for convergence
            num_heads: Number of attention heads for parallel processing
        """
        from .influence_calculator import InfluenceCalculator
        from .context_aggregator import ContextAggregator
        
        self.influence_calculator = InfluenceCalculator() if influence_calculator is None else influence_calculator
        self.context_aggregator = ContextAggregator() if context_aggregator is None else context_aggregator
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.num_heads = num_heads
        self.movement_history = {}
        
    def determine_context(self, vectors: List[Vector]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Determine context around vectors using multi-headed attention.
        
        Args:
            vectors: List of vectors to determine context for
            
        Returns:
            Dictionary mapping vector IDs to lists of (vector_id, impact) tuples
        """
        # Split vectors into attention heads
        head_size = len(vectors) // self.num_heads
        vector_heads = [vectors[i:i + head_size] for i in range(0, len(vectors), head_size)]
        
        # Process each head in parallel
        all_influences = {}
        for head_vectors in vector_heads:
            # Calculate influences for this head
            head_influences = self.influence_calculator.calculate_influences(head_vectors)
            
            # Merge influences
            for vec_id, influences in head_influences.items():
                if vec_id not in all_influences:
                    all_influences[vec_id] = {}
                all_influences[vec_id].update(influences)
        
        # Convert influences to context format
        context = {}
        for vec_id, vec_influences in all_influences.items():
            context[vec_id] = [(other_id, impact) for other_id, impact in vec_influences.items()]
        
        return context
    
    def reembed_vectors(self, vectors: List[Vector]) -> List[Vector]:
        """
        Reembed vectors based on their context using multi-headed attention.
        
        Args:
            vectors: List of vectors
            
        Returns:
            List of reembedded vectors
        """
        current_vectors = vectors.copy()
        
        for iteration in range(self.max_iterations):
            logger.info(f"Reembedding iteration {iteration+1}/{self.max_iterations}")
            
            # Determine context for all vectors using multi-headed attention
            context_map = self.determine_context(current_vectors)
            
            # Update vectors based on context
            updated_vectors = []
            for vector in current_vectors:
                # Get impacts for this vector
                impacts = context_map.get(vector.id, [])
                
                # Calculate movement using multi-headed attention
                movement = self._calculate_movement(vector, impacts, current_vectors)
                
                # Apply movement with learning rate
                new_position = vector.data + movement * self.learning_rate
                
                # Create updated vector
                updated_vector = Vector(
                    id=vector.id,
                    data=new_position,
                    solidness=vector.solidness,
                    impact_radius=vector.impact_radius,
                    metadata=vector.metadata
                )
                updated_vectors.append(updated_vector)
            
            # Check convergence
            if self._check_convergence(current_vectors, updated_vectors):
                logger.info(f"Converged after {iteration+1} iterations")
                break
            
            current_vectors = updated_vectors
        
        return current_vectors
    
    def _calculate_movement(self, 
                          vector: Vector, 
                          impacts: List[Tuple[str, float]],
                          all_vectors: List[Vector]) -> np.ndarray:
        """
        Calculate movement vector based on impacts.
        
        Args:
            vector: The vector
            impacts: List of (vector_id, impact) tuples
            all_vectors: List of all vectors
            
        Returns:
            Movement vector
        """
        movement = np.zeros_like(vector.data)
        total_impact = 0.0
            
        # Create dictionary for easy lookup
        vector_dict = {v.id: v for v in all_vectors}
        
        for other_id, impact in impacts:
            if other_id in vector_dict:
                other_vector = vector_dict[other_id]
                
                # Calculate direction
                direction = other_vector.data - vector.data
                
                # Normalize direction
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                
                # Add to movement
                movement += direction * impact
                total_impact += impact
        
        # Normalize movement
        if total_impact > 0:
            movement = movement / total_impact
        
        return movement
    
    def _check_convergence(self, 
                          old_vectors: List[Vector], 
                          new_vectors: List[Vector]) -> bool:
        """
        Check if vectors have converged.
        
        Args:
            old_vectors: Previous vectors
            new_vectors: Updated vectors
            
        Returns:
            True if converged, False otherwise
        """
        total_movement = 0.0
        
        for old_vec, new_vec in zip(old_vectors, new_vectors):
            movement = np.linalg.norm(new_vec.data - old_vec.data)
            total_movement += movement
        
        avg_movement = total_movement / len(old_vectors)
        return avg_movement < self.convergence_threshold
    
    def calculate_adaptive_learning_rate(self, 
                                      vector_id: str, 
                                      movement_history: List[np.ndarray]) -> float:
        """
        Calculate adaptive learning rate based on movement history.
        
        Args:
            vector_id: ID of the vector
            movement_history: History of vector movements
            
        Returns:
            Adaptive learning rate
        """
        if not movement_history:
            return self.learning_rate
            
        # Calculate average movement magnitude
        magnitudes = [np.linalg.norm(movement) for movement in movement_history]
        avg_magnitude = np.mean(magnitudes)
        
        # Adjust learning rate based on movement history
        if avg_magnitude > 1.0:
            return self.learning_rate / avg_magnitude
        elif avg_magnitude < 0.1:
            return self.learning_rate * 1.5
            
        return self.learning_rate
    
    def calculate_negative_feedback(self, 
                                  vector_id: str, 
                                  metadata: Dict[str, Any]) -> float:
        """
        Calculate negative feedback factor based on vector metadata.
        
        Args:
            vector_id: ID of the vector
            metadata: Vector metadata
            
        Returns:
            Negative feedback factor
        """
        # Default feedback factor
        feedback = 1.0
        
        # Apply metadata-based adjustments
        if 'stability' in metadata:
            stability = metadata.get('stability', 0.5)
            feedback *= (1.0 - 0.5 * stability)  # Reduce movement for stable vectors
            
        return feedback

# Example usage
if __name__ == "__main__":
    # Create some test vectors
    vectors = [
        Vector("v1", np.array([1.0, 0.0, 0.0])),
        Vector("v2", np.array([0.8, 0.2, 0.0])),
        Vector("v3", np.array([0.0, 1.0, 0.0])),
        Vector("v4", np.array([0.0, 0.0, 1.0])),
    ]
    
    # Create a contexter
    contexter = Contexter()
    
    # Determine context
    context_map = contexter.determine_context(vectors)
    print("Context Map:")
    for vector_id, impacts in context_map.items():
        print(f"{vector_id}: {impacts}")
    
    # Reembed vectors
    reembedded_vectors = contexter.reembed_vectors(vectors)
    print("\nReembedded Vectors:")
    for vector in reembedded_vectors:
        print(f"{vector.id}: {vector.data}, solidness={vector.solidness:.2f}")
