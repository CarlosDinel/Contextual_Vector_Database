""" Contexter Module for Contextual Vector Database

This module implements the contexter component of the Contextual Vector Database (CVD),
which is responsible for determining context around vectors and preparing them for reembedding.
It works together with the ReembedingOrchestrator, context_aggregator, and influence_calculator.

Author: Carlos D. Almeida
"""

import numpy as np
import copy
import time
from typing import List, Dict, Tuple, Optional, Any, Union
from scipy.spatial.distance import euclidean, cosine
from scipy.special import softmax
import logging
from Contexter.context_aggregator import ContextAggregator


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Vector:
    """
    Represents a vector in the Contextual Vector Database.
    """
    def __init__(self, 
                 id: str, 
                 data: np.ndarray, 
                 solidness: float = 0.1, 
                 impact_radius: float = 1.0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a vector.
        
        Args:
            id: Unique identifier for the vector
            data: The vector data as a numpy array
            solidness: Initial solidness value (0.0 to 1.0)
            impact_radius: Initial impact radius
            metadata: Additional metadata for the vector
        """
        self.id = id
        self.data = data
        self.solidness = solidness
        self.impact_radius = impact_radius
        self.metadata = metadata or {}
        self.cumulative_impact = 0.0
        self.previous_impact = 0.0
        self.energy = 1.0
        self.last_update_time = time.time()
        

class ContextualInfluenceCalculator:
    """
    Responsible for calculating the contextual influence between vectors.
    """
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize the calculator.
        
        Args:
            epsilon: Small constant to prevent division by zero
        """
        self.epsilon = epsilon
        
    def calculate_similarity(self, vector_i: Vector, vector_j: Vector) -> float:
        """
        Calculate the similarity between two vectors using cosine similarity.
        
        Args:
            vector_i: First vector
            vector_j: Second vector
            
        Returns:
            Similarity score (higher means more similar)
        """
        # Cosine similarity is 1 - cosine distance
        return 1.0 - cosine(vector_i.data, vector_j.data)
    
    def calculate_distance(self, vector_i: Vector, vector_j: Vector) -> float:
        """
        Calculate the Euclidean distance between two vectors.
        
        Args:
            vector_i: First vector
            vector_j: Second vector
            
        Returns:
            Euclidean distance
        """
        return euclidean(vector_i.data, vector_j.data)
    
    def calculate_impact(self, 
                         vector_i: Vector, 
                         vector_j: Vector, 
                         attention_factor: float = 1.0) -> float:
        """
        Calculate the core impact between two vectors.
        
        Impact(Vi, Vj) = Similarity(Vi, Vj) * exp(-Distance(Vi, Vj)^2 / (2 * R_i^2)) * A_ij / (Distance(Vi, Vj) + ε)
        
        Args:
            vector_i: Source vector
            vector_j: Target vector
            attention_factor: Attention factor for hierarchical importance
            
        Returns:
            Impact score
        """
        similarity = self.calculate_similarity(vector_i, vector_j)
        distance = self.calculate_distance(vector_i, vector_j)
        impact_radius = vector_i.impact_radius
        
        # Calculate the exponential decay term
        exp_term = np.exp(-distance**2 / (2 * impact_radius**2))
        
        # Calculate the impact
        impact = similarity * exp_term * attention_factor / (distance + self.epsilon)
        
        return impact


class AdaptiveImpactFunction:
    """
    Manages the dynamic aspects of the impact calculation.
    """
    def __init__(self, decay_factor: float = 0.9):
        """
        Initialize the adaptive impact function.
        
        Args:
            decay_factor: Controls how quickly the impact radius adapts
        """
        self.decay_factor = decay_factor
        
    def calculate_impact_radius(self, 
                               vector: Vector, 
                               cumulative_influence: float) -> float:
        """
        Calculate the dynamic impact radius.
        
        R_i = λ * R_i + (1-λ) * I_cumulative
        
        Args:
            vector: The vector
            cumulative_influence: Cumulative influence received from neighboring vectors
            
        Returns:
            Updated impact radius
        """
        lambda_val = self.decay_factor
        new_radius = lambda_val * vector.impact_radius + (1 - lambda_val) * cumulative_influence
        
        # Ensure the radius stays within reasonable bounds
        new_radius = max(0.1, min(10.0, new_radius))
        
        return new_radius
    
    def calculate_attention_factor(self, 
                                  vector_i: Vector, 
                                  vector_j: Vector) -> float:
        """
        Calculate the attention factor for hierarchical importance.
        
        Args:
            vector_i: Source vector
            vector_j: Target vector
            
        Returns:
            Attention factor
        """
        # This is a simplified implementation
        # In a real system, this would consider hierarchical relationships
        # For now, we use a constant value
        return 1.0
    
    def calculate_weighted_impact(self, 
                                 vector_i: Vector, 
                                 vector_j: Vector, 
                                 base_impact: float) -> float:
        """
        Calculate the weighted impact between two vectors.
        
        Args:
            vector_i: Source vector
            vector_j: Target vector
            base_impact: Base impact value
            
        Returns:
            Weighted impact
        """
        # Apply attention factor
        attention_factor = self.calculate_attention_factor(vector_i, vector_j)
        
        # Apply solidness dampening
        # More solid vectors have less impact
        solidness_dampening = 1.0 - 0.5 * vector_j.solidness
        
        # Calculate weighted impact
        weighted_impact = base_impact * attention_factor * solidness_dampening
        
        return weighted_impact


class SolidnessManager:
    """
    Manages the solidness (stability) of vectors over time.
    """
    def __init__(self, 
                 base_solidness_rate: float = 0.01, 
                 impact_factor: float = 0.2,
                 max_solidness: float = 0.9):
        """
        Initialize the solidness manager.
        
        Args:
            base_solidness_rate: Base rate at which solidness increases
            impact_factor: How much impact affects solidness
            max_solidness: Maximum solidness value
        """
        self.base_solidness_rate = base_solidness_rate
        self.impact_factor = impact_factor
        self.max_solidness = max_solidness
        self.last_update_times = {}
        
    def calculate_solidness(self, 
                           vector: Vector, 
                           current_time: float,
                           cumulative_impact: float) -> float:
        """
        Calculate the updated solidness value.
        
        Args:
            vector: The vector
            current_time: Current time
            cumulative_impact: Cumulative impact received
            
        Returns:
            Updated solidness value
        """
        # Get time since last update
        last_time = self.last_update_times.get(vector.id, 0)
        time_diff = current_time - last_time
        self.last_update_times[vector.id] = current_time
        
        # Calculate time-based solidness increase
        time_factor = self.base_solidness_rate * time_diff
        
        # Calculate impact-based solidness adjustment
        # High impact can either increase or decrease solidness depending on implementation
        impact_adjustment = self.impact_factor * cumulative_impact
        
        # Calculate new solidness
        new_solidness = vector.solidness + time_factor + impact_adjustment
        
        # Ensure solidness stays within bounds
        new_solidness = max(0.0, min(self.max_solidness, new_solidness))
        
        return new_solidness
    
    def update_solidness(self, vector: Vector, new_solidness: float) -> None:
        """
        Update the solidness of a vector.
        
        Args:
            vector: The vector
            new_solidness: New solidness value
        """
        vector.solidness = new_solidness


class HierarchicalInfluenceProcessor:
    """
    Processes influences between vectors considering hierarchical relationships.
    """
    def __init__(self, 
                 learning_rate: float = 0.1, 
                 solidness_factor: float = 0.5,
                 max_movement: float = 1.0):
        """
        Initialize the hierarchical influence processor.
        
        Args:
            learning_rate: Base learning rate for position updates
            solidness_factor: How much solidness affects movement
            max_movement: Maximum allowed movement magnitude
        """
        self.learning_rate = learning_rate
        self.solidness_factor = solidness_factor
        self.max_movement = max_movement
        
    def process_influences(self, 
                          vector: Vector, 
                          all_vectors: List[Vector],
                          impacts: List[Tuple[str, float]],
                          solidness: float) -> Optional[np.ndarray]:
        """
        Process influences and calculate new vector position.
        
        Args:
            vector: The vector
            all_vectors: All vectors
            impacts: List of (vector_id, impact) tuples
            solidness: Current solidness value
            
        Returns:
            New vector position or None if no update needed
        """
        if not impacts:
            return None
            
        # Create a dictionary for easy lookup
        vector_dict = {v.id: v for v in all_vectors}
        
        # Calculate effective learning rate based on solidness
        # More solid vectors move less
        effective_lr = self.learning_rate * (1.0 - self.solidness_factor * solidness)
        
        # Calculate weighted average direction
        movement_vector = np.zeros_like(vector.data)
        total_weight = 0.0
        
        for vector_id, impact in impacts:
            if vector_id in vector_dict:
                other_vector = vector_dict[vector_id]
                
                # Calculate direction and weight
                direction = other_vector.data - vector.data
                weight = impact
                
                # Add to movement vector
                movement_vector += direction * weight
                total_weight += weight
        
        # Normalize and apply learning rate
        if total_weight > 0:
            movement_vector = movement_vector / total_weight
            movement_vector = movement_vector * effective_lr
            
            # Limit maximum movement
            magnitude = np.linalg.norm(movement_vector)
            if magnitude > self.max_movement:
                movement_vector = movement_vector * (self.max_movement / magnitude)
            
            # Calculate new position
            new_position = vector.data + movement_vector
            return new_position
        
        return None


class LocalContextClusterer:
    """
    Clusters vectors to find relevant local context.
    """
    def __init__(self, 
                 max_neighbors: int = 10, 
                 distance_threshold: float = 5.0,
                 impact_threshold: float = 0.01):
        """
        Initialize the local context clusterer.
        
        Args:
            max_neighbors: Maximum number of neighbors to consider
            distance_threshold: Maximum distance for neighbors
            impact_threshold: Minimum impact threshold for relevance
        """
        self.max_neighbors = max_neighbors
        self.distance_threshold = distance_threshold
        self.impact_threshold = impact_threshold
        
    def find_relevant_neighbors(self, 
                               vector: Vector, 
                               all_vectors: List[Vector]) -> List[Vector]:
        """
        Find relevant neighbors for a vector.
        
        Args:
            vector: The vector
            all_vectors: All vectors
            
        Returns:
            List of relevant neighbor vectors
        """
        # Calculate distances to all other vectors
        distances = []
        for other in all_vectors:
            if other.id != vector.id:
                distance = np.linalg.norm(vector.data - other.data)
                distances.append((other, distance))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Take closest neighbors within threshold
        neighbors = []
        for other, distance in distances:
            if distance <= self.distance_threshold:
                neighbors.append(other)
            
            # Limit number of neighbors
            if len(neighbors) >= self.max_neighbors:
                break
        
        return neighbors
    
    def filter_weak_interactions(self, 
                                vector: Vector, 
                                influences: Dict[str, Tuple[Vector, float]]) -> Dict[str, Tuple[Vector, float]]:
        """
        Filter out weak interactions.
        
        Args:
            vector: The vector
            influences: Dictionary mapping vector IDs to (vector, impact) tuples
            
        Returns:
            Filtered influences
        """
        filtered = {}
        
        for vector_id, (other, impact) in influences.items():
            if impact >= self.impact_threshold:
                filtered[vector_id] = (other, impact)
        
        return filtered


class NegativeFeedbackLoop:
    """
    Implements negative feedback to prevent oscillations.
    """
    def __init__(self, 
                 dampening_factor: float = 0.2, 
                 recovery_rate: float = 0.05,
                 max_dampening: float = 0.8):
        """
        Initialize the negative feedback loop.
        
        Args:
            dampening_factor: Factor controlling how quickly dampening increases
            recovery_rate: Rate at which dampening recovers
            max_dampening: Maximum dampening value
        """
        self.dampening_factor = dampening_factor
        self.recovery_rate = recovery_rate
        self.max_dampening = max_dampening
        self.dampening_values = {}
        
    def calculate_dampened_impact(self, 
                                 source: Vector, 
                                 target: Vector, 
                                 impact: float) -> float:
        """
        Calculate dampened impact based on negative feedback.
        
        Args:
            source: Source vector
            target: Target vector
            impact: Original impact value
            
        Returns:
            Dampened impact value
        """
        # Get current dampening for this pair
        pair_key = f"{source.id}_{target.id}"
        dampening = self.dampening_values.get(pair_key, 0.0)
        
        # Apply dampening
        dampened_impact = impact * (1.0 - dampening)
        
        return dampened_impact
    
    def update_feedback(self, vector: Vector, movement_magnitude: float) -> None:
        """
        Update feedback based on movement magnitude.
        
        Args:
            vector: The vector
            movement_magnitude: Magnitude of movement
        """
        # Increase dampening for vectors that move a lot
        for other_id in self.dampening_values.keys():
            if other_id.startswith(f"{vector.id}_") or other_id.endswith(f"_{vector.id}"):
                current_dampening = self.dampening_values[other_id]
                
                # Increase dampening based on movement
                increase = self.dampening_factor * movement_magnitude
                
                # Apply increase with cap
                new_dampening = min(self.max_dampening, current_dampening + increase)
                self.dampening_values[other_id] = new_dampening
    
    def apply_recovery(self) -> None:
        """
        Apply recovery to all dampening values.
        """
        for key in self.dampening_values:
            current = self.dampening_values[key]
            
            # Apply recovery
            new_value = max(0.0, current - self.recovery_rate)
            self.dampening_values[key] = new_value


class EnergyDecay:
    """
    Manages the decay of movement energy over time.
    """
    def __init__(self, decay_rate: float = 0.95):
        """
        Initialize the energy decay mechanism.
        
        Args:
            decay_rate: Rate at which energy decays
        """
        self.decay_rate = decay_rate
        
    def apply_decay(self, vector: Vector) -> None:
        """
        Apply energy decay to a vector.
        
        Args:
            vector: The vector
        """
        vector.energy = vector.energy * self.decay_rate


class AdaptiveConvergence:
    """
    Prevents unnecessary updates when vectors reach stability.
    
    This class implements a mechanism to adaptively determine when vectors
    have reached a stable state and don't need further updates, saving
    computational resources.
    """
    def __init__(self, 
                 min_movement_threshold=0.001, 
                 stability_window=3,
                 adaptive_threshold=True):
        """
        Initialize the adaptive convergence mechanism.
        
        Args:
            min_movement_threshold: Minimum movement threshold to consider a vector stable
            stability_window: Number of consecutive iterations below threshold to consider stable
            adaptive_threshold: Whether to adapt threshold based on vector characteristics
        """
        self.min_movement_threshold = min_movement_threshold
        self.stability_window = stability_window
        self.adaptive_threshold = adaptive_threshold
        self.movement_history = {}  # Maps vector ID to list of recent movements
        
    def should_update(self, vector, proposed_position):
        """
        Determine if a vector should be updated based on convergence criteria.
        
        Args:
            vector: The vector being considered for update
            proposed_position: The proposed new position
            
        Returns:
            Boolean indicating whether the vector should be updated
        """
        # Calculate movement magnitude
        current_position = vector.data
        movement = np.linalg.norm(proposed_position - current_position)
        
        # Get vector-specific threshold
        threshold = self._get_adaptive_threshold(vector) if self.adaptive_threshold else self.min_movement_threshold
        
        # Update movement history
        if vector.id not in self.movement_history:
            self.movement_history[vector.id] = []
        
        history = self.movement_history[vector.id]
        history.append(movement)
        
        # Keep only the most recent movements within the stability window
        if len(history) > self.stability_window:
            history = history[-self.stability_window:]
        self.movement_history[vector.id] = history
        
        # Check if all recent movements are below threshold
        is_stable = all(m < threshold for m in history) and len(history) >= self.stability_window
        
        # If stable, we can skip the update
        return not is_stable
    
    def _get_adaptive_threshold(self, vector):
        """
        Calculate an adaptive threshold based on vector characteristics.
        
        Args:
            vector: The vector to calculate threshold for
            
        Returns:
            Adaptive threshold value
        """
        # Vectors with higher solidness can have lower thresholds
        # This means well-established vectors need to move less to be considered stable
        base_threshold = self.min_movement_threshold
        solidness_factor = 1.0 - 0.5 * vector.solidness  # Reduce threshold by up to 50% for solid vectors
        
        # Scale threshold by vector magnitude to make it relative
        magnitude = np.linalg.norm(vector.data)
        magnitude_factor = 1.0
        if magnitude > 0:
            magnitude_factor = min(1.0, 0.1 / magnitude)  # Smaller vectors get higher relative thresholds
        
        return base_threshold * solidness_factor * magnitude_factor
    
    def reset_history(self, vector_id=None):
        """
        Reset movement history for a specific vector or all vectors.
        
        Args:
            vector_id: ID of vector to reset, or None to reset all
        """
        if vector_id is None:
            self.movement_history = {}
        elif vector_id in self.movement_history:
            self.movement_history[vector_id] = []


class Contexter:
    """
    Main class for the Contexter module.
    """
    def __init__(self, 
                 influence_calculator: Optional[ContextualInfluenceCalculator] = None,
                 impact_function: Optional[AdaptiveImpactFunction] = None,
                 solidness_manager: Optional[SolidnessManager] = None,
                 influence_processor: Optional[HierarchicalInfluenceProcessor] = None,
                 context_clusterer: Optional[LocalContextClusterer] = None,
                 energy_decay: Optional[EnergyDecay] = None,
                 adaptive_convergence: Optional[AdaptiveConvergence] = None,
                 negative_feedback: Optional[NegativeFeedbackLoop] = None):
        """
        Initialize the Contexter.
        
        Args:
            influence_calculator: Calculator for contextual influence
            impact_function: Function for adaptive impact
            solidness_manager: Manager for vector solidness
            influence_processor: Processor for hierarchical influence
            context_clusterer: Clusterer for local context
            energy_decay: Mechanism for energy decay
            adaptive_convergence: Mechanism for adaptive convergence
            negative_feedback: Mechanism for negative feedback
        """
        self.influence_calculator = influence_calculator or ContextualInfluenceCalculator()
        self.impact_function = impact_function or AdaptiveImpactFunction()
        self.solidness_manager = solidness_manager or SolidnessManager()
        self.influence_processor = influence_processor or HierarchicalInfluenceProcessor()
        self.context_clusterer = context_clusterer or LocalContextClusterer()
        self.energy_decay = energy_decay or EnergyDecay() if 'EnergyDecay' in globals() else EnergyDecay()
        self.adaptive_convergence = adaptive_convergence or AdaptiveConvergence()
        self.negative_feedback = negative_feedback or NegativeFeedbackLoop()

    def determine_context(self, vectors: List[Vector]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Determine the context around vectors.
        
        Args:
            vectors: List of vectors
            
        Returns:
            Dictionary mapping vector IDs to lists of (vector_id, impact) tuples
        """
        context_map = {}
        
        for vector_i in vectors:
            # Find relevant neighbors for efficiency
            neighbors = self.context_clusterer.find_relevant_neighbors(vector_i, vectors)
            
            impacts = []
            for vector_j in neighbors:
                if vector_i.id != vector_j.id:
                    # Calculate impact between vectors
                    impact = self.influence_calculator.calculate_impact(vector_i, vector_j)
                    
                    # Apply negative feedback dampening if available
                    if self.negative_feedback:
                        impact = self.negative_feedback.calculate_dampened_impact(
                            vector_j, vector_i, impact)
                    
                    # Filter weak interactions
                    if impact > self.context_clusterer.impact_threshold:
                        impacts.append((vector_j.id, impact))
            
            # Sort by impact and store
            impacts.sort(key=lambda x: x[1], reverse=True)
            context_map[vector_i.id] = impacts
        
        return context_map
    
    def update_vectors(self, vectors: List[Vector], context_map: Dict[str, List[Tuple[str, float]]]) -> List[Vector]:
        """
        Update vectors based on their context.
        
        Args:
            vectors: List of vectors
            context_map: Dictionary mapping vector IDs to lists of (vector_id, impact) tuples
            
        Returns:
            List of updated vectors
        """
        updated_vectors = copy.deepcopy(vectors)
        
        # Create a dictionary for easy lookup
        vector_dict = {v.id: v for v in vectors}
        
        # Apply recovery to negative feedback loop
        if self.negative_feedback:
            self.negative_feedback.apply_recovery()
        
        for i, vector in enumerate(vectors):
            # Get impacts for this vector
            impacts = context_map.get(vector.id, [])
            
            # Calculate cumulative impact
            cumulative_impact = sum(impact for _, impact in impacts)
            
            # Update solidness based on impact
            new_solidness = self.solidness_manager.calculate_solidness(
                vector, 
                time.time(), 
                cumulative_impact
            )
            updated_vectors[i].solidness = new_solidness
            
            # Process hierarchical influence
            updated_position = self.influence_processor.process_influences(
                vector,
                vectors,
                impacts,
                new_solidness
            )
            
            # Apply adaptive convergence check
            should_update = True
            if self.adaptive_convergence and updated_position is not None:
                should_update = self.adaptive_convergence.should_update(vector, updated_position)
            
            # Update vector position if needed
            if updated_position is not None and should_update:
                # Calculate movement magnitude
                movement_magnitude = np.linalg.norm(updated_position - vector.data)
                
                # Update negative feedback
                if self.negative_feedback:
                    self.negative_feedback.update_feedback(vector, movement_magnitude)
                
                # Apply the update
                updated_vectors[i].data = updated_position
                updated_vectors[i].last_update_time = time.time()
        
        return updated_vectors
    
    def reembed_vectors(self, vectors: List[Vector], iterations: int = 5) -> List[Vector]:
        """
        Reembed vectors based on their context.
        
        Args:
            vectors: List of vectors
            iterations: Number of update iterations
            
        Returns:
            List of reembedded vectors
        """
        current_vectors = vectors.copy()
        
        for i in range(iterations):
            logger.info(f"Reembedding iteration {i+1}/{iterations}")
            
            # Determine context for all vectors
            context_map = self.determine_context(current_vectors)
            
            # Update vectors based on context
            current_vectors = self.update_vectors(current_vectors, context_map)
            
            # Apply energy decay to all vectors
            if self.energy_decay:
                for vector in current_vectors:
                    self.energy_decay.apply_decay(vector)
        
        return current_vectors
    
    def calculate_adaptive_learning_rate(self, vector_id: str, movement_history: List[np.ndarray]) -> float:
        """
        Calculate an adaptive learning rate based on movement history.
        
        Args:
            vector_id: ID of the vector
            movement_history: History of vector movements
            
        Returns:
            Adaptive learning rate
        """
        # Base learning rate
        base_lr = 0.1
        
        # If no history, use base rate
        if not movement_history:
            return base_lr
            
        # Calculate average movement magnitude
        magnitudes = [np.linalg.norm(movement) for movement in movement_history]
        avg_magnitude = np.mean(magnitudes) if magnitudes else 0.0
        
        # Adjust learning rate based on movement history
        # If movements are large, reduce learning rate to prevent oscillations
        # If movements are small, increase learning rate to speed up convergence
        if avg_magnitude > 1.0:
            return base_lr / avg_magnitude
        elif avg_magnitude < 0.1:
            return base_lr * 1.5
            
        return base_lr
    
    def calculate_negative_feedback(self, vector_id: str, metadata: Dict[str, Any]) -> float:
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
        
        # Apply metadata-based adjustments if available
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
