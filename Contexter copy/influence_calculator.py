# Add imports at the top of influence_calculator.py
import numpy as np
from typing import Dict, List, Tuple, Any
import sys
import os

# Import your existing components
from context_aggregator import ContextAggregator
from Contexter.contexter_model import Contexter

class InfluenceCalculator:
    """
    Calculates and manages influence between vectors in the Contextual Vector Database.
    Includes the ReembeddingOrchestrator functionality for vector repositioning.
    """
    
    def __init__(self, 
                 batch_size=100, 
                 max_iterations=10,
                 solidness_factor=0.2, 
                 energy_decay_rate=0.95,
                 impact_radius_factor=0.3, 
                 hierarchical_depth=2):
        """
        Initialize the InfluenceCalculator with configuration parameters.
        
        Args:
            batch_size: Number of vectors to process in each batch
            max_iterations: Maximum number of reembedding iterations
            solidness_factor: Factor controlling how quickly vectors become stable
            energy_decay_rate: Rate at which movement energy decays over iterations
            impact_radius_factor: Factor for calculating impact radius
            hierarchical_depth: Depth of hierarchical influence propagation
        """
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.solidness_factor = solidness_factor
        self.energy_decay_rate = energy_decay_rate
        self.impact_radius_factor = impact_radius_factor
        self.hierarchical_depth = hierarchical_depth
        
        # Initialize related components
        self.context_aggregator = ContextAggregator()
        self.contexte_model = Contexter()
        
        # Track vector solidness (stability over time)
        self.vector_solidness = {}
    
    def calculate_impact_radius(self, vector_id: str, vector: np.ndarray, 
                               neighbors: Dict[str, np.ndarray]) -> float:
        """
        Dynamically calculate impact radius based on local density.
        
        Args:
            vector_id: ID of the vector
            vector: The vector's embedding
            neighbors: Dictionary of neighbor IDs to their embeddings
            
        Returns:
            float: The calculated impact radius
        """
        if not neighbors:
            return self.impact_radius_factor
            
        # Calculate average distance to neighbors
        distances = []
        for neighbor_id, neighbor in neighbors.items():
            distance = np.linalg.norm(vector - neighbor)
            distances.append(distance)
            
        avg_distance = np.mean(distances) if distances else self.impact_radius_factor
        # Adjust radius based on local density
        return avg_distance * self.impact_radius_factor
    
    def calculate_initial_impact(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate base impact between two vectors.
        
        Args:
            vector1: First vector embedding
            vector2: Second vector embedding
            
        Returns:
            float: Initial impact value
        """
        # Distance-based impact
        distance = np.linalg.norm(vector1 - vector2)
        distance_impact = 1.0 / (1.0 + distance)
        
        # Semantic similarity impact (cosine similarity)
        dot_product = np.dot(vector1, vector2)
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        similarity = dot_product / (norm_product + 1e-10)  # Avoid division by zero
        
        # Combine distance and semantic impacts
        return 0.5 * distance_impact + 0.5 * max(0, similarity)
    
    def calculate_weighted_impact(self, base_impact: float, 
                                 vector1_id: str, vector2_id: str,
                                 metadata: Dict[str, Dict[str, Any]]) -> float:
        """
        Apply weights to the base impact.
        
        Args:
            base_impact: Initial calculated impact
            vector1_id: ID of first vector
            vector2_id: ID of second vector
            metadata: Dictionary mapping vector IDs to their metadata
            
        Returns:
            float: Weighted impact value
        """
        # Get vector metadata for weighting
        v1_metadata = metadata.get(vector1_id, {})
        v2_metadata = metadata.get(vector2_id, {})
        
        # Example weights based on metadata
        recency_weight = 1.0
        importance_weight = 1.0
        
        if 'timestamp' in v1_metadata and 'timestamp' in v2_metadata:
            # More recent interactions have higher weight
            time_diff = abs(v1_metadata['timestamp'] - v2_metadata['timestamp'])
            recency_weight = 1.0 / (1.0 + 0.1 * time_diff)
            
        if 'importance' in v1_metadata and 'importance' in v2_metadata:
            # Average importance of the two vectors
            importance_weight = 0.5 * (v1_metadata['importance'] + v2_metadata['importance'])
        
        # Apply solidness factor - more solid vectors have less impact
        v1_solidness = self.vector_solidness.get(vector1_id, 0)
        v2_solidness = self.vector_solidness.get(vector2_id, 0)
        solidness_modifier = 1.0 - 0.5 * (v1_solidness + v2_solidness)
        
        return base_impact * recency_weight * importance_weight * solidness_modifier
    
    def propagate_hierarchical_influence(self, vector_id: str, 
                                        neighbors: Dict[str, np.ndarray],
                                        all_vectors: Dict[str, np.ndarray],
                                        depth: int = 1) -> Dict[str, float]:
        """
        Propagate influence through the network hierarchically.
        
        Args:
            vector_id: ID of the source vector
            neighbors: Dictionary of direct neighbor IDs to their embeddings
            all_vectors: Dictionary of all vector IDs to their embeddings
            depth: Current depth of propagation
            
        Returns:
            Dict[str, float]: Dictionary mapping vector IDs to their influence values
        """
        if depth <= 0:
            return {}
            
        # First-level influences
        influences = {neighbor_id: 1.0 for neighbor_id in neighbors}
        
        # Propagate to next level with diminishing effect
        if depth > 1:
            for neighbor_id in neighbors:
                # Get neighbors of this neighbor
                second_neighbors = self.context_aggregator.get_context_neighbors(
                    neighbor_id, all_vectors, exclude=[vector_id])
                
                # Recursive call with reduced depth
                second_influences = self.propagate_hierarchical_influence(
                    neighbor_id, second_neighbors, all_vectors, depth-1)
                
                # Add second-level influences with diminishing factor
                for second_id, influence in second_influences.items():
                    if second_id not in influences and second_id != vector_id:
                        influences[second_id] = influence * 0.5  # Diminishing factor
        
        return influences
    
    def is_significant_impact(self, impact: float, threshold: float = 0.1) -> bool:
        """
        Determine if an impact is significant enough to consider.
        
        Args:
            impact: The calculated impact value
            threshold: Minimum threshold for significance
            
        Returns:
            bool: True if impact is significant, False otherwise
        """
        return impact > threshold
    
    # not needed to update position within this directory 
    def update_vector_positions(self, 
                               batch_vectors: Dict[str, np.ndarray],
                               all_vectors: Dict[str, np.ndarray],
                               metadata: Dict[str, Dict[str, Any]],
                               movement_history: Dict[str, List[np.ndarray]],
                               iteration: int) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Update vector positions based on contextual relationships.
        
        Args:
            batch_vectors: Dictionary of vector IDs to embeddings for this batch
            all_vectors: Dictionary of all vector IDs to embeddings
            metadata: Dictionary of vector IDs to their metadata
            movement_history: Dictionary of vector IDs to their movement history
            iteration: Current iteration number
            
        Returns:
            Tuple[Dict[str, np.ndarray], float]: Updated vectors and total energy
        """
        updates = {}
        total_energy = 0
        
        for vector_id, vector in batch_vectors.items():
            # Get local context (neighbors)
            neighbors = self.context_aggregator.get_context_neighbors(
                vector_id, all_vectors)
            
            # Calculate impact radius
            impact_radius = self.calculate_impact_radius(vector_id, vector, neighbors)
            
            # Get hierarchical influences
            influences = self.propagate_hierarchical_influence(
                vector_id, neighbors, all_vectors, self.hierarchical_depth)
            
            # Calculate position update
            update_vector = np.zeros_like(vector)
            
            for neighbor_id, influence in influences.items():
                neighbor = all_vectors[neighbor_id]
                
                # Calculate impact
                base_impact = self.calculate_initial_impact(vector, neighbor)
                weighted_impact = self.calculate_weighted_impact(
                    base_impact, vector_id, neighbor_id, metadata)
                
                # Apply sparse attention
                if self.is_significant_impact(weighted_impact):
                    # Direction of influence
                    direction = neighbor - vector
                    
                    # Apply influence with diminishing factor
                    update_vector += direction * weighted_impact * influence
            
            # Apply adaptive convergence
            learning_rate = self.contexte_model.calculate_adaptive_learning_rate(
                vector_id, movement_history.get(vector_id, []))
            
            # Apply negative feedback
            feedback = self.contexte_model.calculate_negative_feedback(
                vector_id, metadata.get(vector_id, {}))
            
            # Apply energy decay based on iteration
            energy_factor = self.energy_decay_rate ** iteration
            
            # Final update
            final_update = update_vector * learning_rate * feedback * energy_factor
            
            # Track energy (magnitude of movement)
            update_energy = np.linalg.norm(final_update)
            total_energy += update_energy
            
            # Update vector solidness based on movement
            current_solidness = self.vector_solidness.get(vector_id, 0)
            if update_energy < 0.01:  # Small movement increases solidness
                new_solidness = min(1.0, current_solidness + self.solidness_factor)
            else:  # Large movement decreases solidness
                new_solidness = max(0.0, current_solidness - self.solidness_factor)
            self.vector_solidness[vector_id] = new_solidness
            
            updates[vector_id] = vector + final_update
        
        return updates, total_energy
    

    # look if it is needed to reembed vectors within this function 
    def reembed_vectors(self, 
                       vectors: Dict[str, np.ndarray],
                       metadata: Dict[str, Dict[str, Any]] = None,
                       movement_history: Dict[str, List[np.ndarray]] = None,
                       iterations: int = None) -> Dict[str, Any]:
        """
        Main method to reembed vectors.
        
        Args:
            vectors: Dictionary of vector IDs to their embeddings
            metadata: Dictionary of vector IDs to their metadata
            movement_history: Dictionary of vector IDs to their movement history
            iterations: Number of iterations to run
            
        Returns:
            Dict[str, Any]: Metrics and updated vectors
        """
        if metadata is None:
            metadata = {}
        if movement_history is None:
            movement_history = {}
        if iterations is None:
            iterations = self.max_iterations
            
        metrics = {
            'iterations': iterations,
            'energy_per_iteration': [],
            'vectors_updated': 0
        }
        
        updated_vectors = vectors.copy()
        all_vector_ids = list(vectors.keys())
        
        for iteration in range(iterations):
            # Process vectors in batches
            total_iteration_energy = 0
            
            for i in range(0, len(all_vector_ids), self.batch_size):
                batch_ids = all_vector_ids[i:i+self.batch_size]
                batch_vectors = {vid: updated_vectors[vid] for vid in batch_ids}
                
                # Update positions
                updates, batch_energy = self.update_vector_positions(
                    batch_vectors, updated_vectors, metadata, movement_history, iteration)
                total_iteration_energy += batch_energy
                
                # Apply updates
                for vector_id, new_position in updates.items():
                    updated_vectors[vector_id] = new_position
                    
                    # Update movement history
                    if vector_id not in movement_history:
                        movement_history[vector_id] = []
                    movement_history[vector_id].append(new_position)
                    
                    metrics['vectors_updated'] += 1
            
            # Record energy for this iteration
            metrics['energy_per_iteration'].append(total_iteration_energy)
            
            # Early stopping if energy is very low
            if total_iteration_energy < 0.001 * len(all_vector_ids):
                print(f"Converged after {iteration+1} iterations")
                break
        
        metrics['updated_vectors'] = updated_vectors
        metrics['final_solidness'] = self.vector_solidness
        return metrics



# example usage 