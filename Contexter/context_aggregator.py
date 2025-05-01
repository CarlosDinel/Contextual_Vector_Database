""" Context Aggregator Module

This module implements the context aggregation and iteration management
for the Contexter. It handles the coordination of vector updates and
ensures proper convergence of the re-embedding process.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .influence_calculator import InfluenceCalculator, VectorMetrics
from .position_encoder import PositionEncoder, PositionInfo
from .stabilization import Stabilizer, StabilizationMetrics
from .base_model import Vector

logger = logging.getLogger(__name__)

@dataclass
class IterationMetrics:
    """Container for iteration metrics."""
    iteration: int
    total_energy: float
    convergence_rate: float
    vectors_updated: int

class ContextAggregator:
    """Manages context aggregation and iteration control."""
    
    def __init__(
        self,
        influence_calculator: Optional[InfluenceCalculator] = None,
        position_encoder: Optional[PositionEncoder] = None,
        stabilizer: Optional[Stabilizer] = None,
        batch_size: int = 32,
        max_workers: int = 4,
        max_iterations: int = 100,
        convergence_threshold: float = 0.001
    ):
        """Initialize the context aggregator.
        
        Args:
            influence_calculator: Influence calculator instance
            position_encoder: Position encoder instance
            stabilizer: Stabilizer instance
            batch_size: Size of processing batches
            max_workers: Maximum number of worker threads
            max_iterations: Maximum number of iterations
            convergence_threshold: Threshold for convergence
        """
        self.influence_calculator = influence_calculator or InfluenceCalculator()
        self.position_encoder = position_encoder or PositionEncoder()
        self.stabilizer = stabilizer or Stabilizer()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.iteration_metrics: List[IterationMetrics] = []
        
    def aggregate_context(
        self,
        vectors: List[Vector],
        metadata: Dict[str, Dict[str, Any]]
    ) -> List[Vector]:
        """Aggregate context and update vectors.
        
        Args:
            vectors: List of vectors to update
            metadata: Dictionary of vector metadata
            
        Returns:
            Updated vectors
        """
        current_vectors = vectors.copy()
        iteration = 0
        
        while iteration < self.max_iterations:
            logger.info(f"Starting iteration {iteration + 1}/{self.max_iterations}")
            
            # Process vectors in batches
            batch_results = []
            for i in range(0, len(current_vectors), self.batch_size):
                batch = current_vectors[i:i + self.batch_size]
                batch_results.extend(
                    self._process_batch(batch, current_vectors, metadata, iteration)
                )
            
            # Calculate iteration metrics
            iteration_metrics = self._calculate_iteration_metrics(batch_results)
            self.iteration_metrics.append(iteration_metrics)
            
            # Update vectors
            current_vectors = [result.vector for result in batch_results]
            
            # Check for convergence
            if self._check_convergence(iteration_metrics):
                logger.info(f"Convergence reached after {iteration + 1} iterations")
                break
                
            iteration += 1
            
        return current_vectors
        
    def _process_batch(
        self,
        batch: List[Vector],
        all_vectors: List[Vector],
        metadata: Dict[str, Dict[str, Any]],
        iteration: int
    ) -> List[Tuple[Vector, float]]:
        """Process a batch of vectors.
        
        Args:
            batch: Batch of vectors to process
            all_vectors: All vectors in the system
            metadata: Vector metadata
            iteration: Current iteration number
            
        Returns:
            List of (updated vector, energy) tuples
        """
        results = []
        
        for vector in batch:
            # Get context vectors
            context_vectors = self._get_context_vectors(vector, all_vectors)
            
            # Calculate influence metrics
            metrics = self.influence_calculator.calculate_metrics(
                vector.data,
                metadata.get(vector.id, {}),
                [(v.id, v.data) for v in context_vectors]
            )
            
            # Recalculate position
            position_info = self.position_encoder.recalculate_position(
                vector.data,
                [(v.id, v.data, metadata.get(v.id, {})) for v in context_vectors],
                metadata.get(vector.id, {}).get('position', 0)
            )
            
            # Calculate energy
            energy = np.linalg.norm(position_info.position - vector.data)
            
            # Calculate stabilization
            stabilization = self.stabilizer.calculate_stabilization(
                vector.id,
                energy,
                iteration,
                metadata.get(vector.id, {})
            )
            
            # Apply stabilization to update
            update = position_info.position - vector.data
            stabilized_update = self.stabilizer.apply_stabilization(
                vector.data,
                update,
                stabilization
            )
            
            # Create updated vector
            updated_vector = Vector(
                id=vector.id,
                data=vector.data + stabilized_update,
                metadata=vector.metadata.copy()
            )
            
            results.append((updated_vector, energy))
            
        return results
        
    def _get_context_vectors(
        self,
        vector: Vector,
        all_vectors: List[Vector]
    ) -> List[Vector]:
        """Get context vectors for a given vector.
        
        Args:
            vector: Vector to get context for
            all_vectors: All vectors in the system
            
        Returns:
            List of context vectors
        """
        # Calculate distances to all other vectors
        distances = []
        for other in all_vectors:
            if other.id != vector.id:
                distance = np.linalg.norm(vector.data - other.data)
                distances.append((distance, other))
                
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        
        # Return closest vectors as context
        return [v for _, v in distances[:self.batch_size]]
        
    def _calculate_iteration_metrics(
        self,
        batch_results: List[Tuple[Vector, float]]
    ) -> IterationMetrics:
        """Calculate metrics for an iteration.
        
        Args:
            batch_results: List of (vector, energy) tuples
            
        Returns:
            Iteration metrics
        """
        total_energy = sum(energy for _, energy in batch_results)
        convergence_rate = self._calculate_convergence_rate()
        vectors_updated = len(batch_results)
        
        return IterationMetrics(
            iteration=len(self.iteration_metrics),
            total_energy=total_energy,
            convergence_rate=convergence_rate,
            vectors_updated=vectors_updated
        )
        
    def _calculate_convergence_rate(self) -> float:
        """Calculate current convergence rate.
        
        Returns:
            Convergence rate
        """
        if len(self.iteration_metrics) < 2:
            return 1.0
            
        # Calculate energy trend
        energies = [metrics.total_energy for metrics in self.iteration_metrics]
        energy_trend = np.diff(energies)
        
        # Calculate convergence rate
        if len(energy_trend) > 0:
            avg_trend = np.mean(energy_trend)
            return 1.0 / (1.0 + abs(avg_trend))
            
        return 1.0
        
    def _check_convergence(self, metrics: IterationMetrics) -> bool:
        """Check if the process has converged.
        
        Args:
            metrics: Current iteration metrics
            
        Returns:
            True if converged, False otherwise
        """
        # Check energy threshold
        if metrics.total_energy < self.convergence_threshold:
            return True
            
        # Check convergence rate
        if metrics.convergence_rate < 0.1:
            return True
            
        return False
        
    def get_iteration_metrics(self) -> List[IterationMetrics]:
        """Get all iteration metrics.
        
        Returns:
            List of iteration metrics
        """
        return self.iteration_metrics
        
    def clear_metrics(self):
        """Clear iteration metrics."""
        self.iteration_metrics.clear()
        
    def close(self):
        """Clean up resources."""
        self.executor.shutdown()
