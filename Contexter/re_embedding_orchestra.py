""" Re-embedding Orchestra Module

This module implements the orchestration pipeline for re-embedding vectors
in the Contexter. It coordinates the process of updating vector embeddings
based on their context and relationships.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .influence_calculator import InfluenceCalculator, VectorMetrics
from .position_encoder import PositionEncoder, PositionInfo
from .base_model import Vector

logger = logging.getLogger(__name__)

@dataclass
class ReembeddingResult:
    """Container for re-embedding results."""
    vector: Vector
    metrics: VectorMetrics
    position_info: PositionInfo
    energy: float

class ReembeddingOrchestra:
    """Orchestrates the re-embedding process for vectors."""
    
    def __init__(
        self,
        influence_calculator: Optional[InfluenceCalculator] = None,
        position_encoder: Optional[PositionEncoder] = None,
        batch_size: int = 32,
        max_workers: int = 4,
        convergence_threshold: float = 0.001
    ):
        """Initialize the re-embedding orchestra.
        
        Args:
            influence_calculator: Influence calculator instance
            position_encoder: Position encoder instance
            batch_size: Size of processing batches
            max_workers: Maximum number of worker threads
            convergence_threshold: Threshold for convergence
        """
        self.influence_calculator = influence_calculator or InfluenceCalculator()
        self.position_encoder = position_encoder or PositionEncoder()
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.convergence_threshold = convergence_threshold
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def reembed_vectors(
        self,
        vectors: List[Vector],
        max_iterations: int = 10,
        learning_rate: float = 0.1
    ) -> List[ReembeddingResult]:
        """Re-embed a list of vectors based on their context.
        
        Args:
            vectors: List of vectors to re-embed
            max_iterations: Maximum number of iterations
            learning_rate: Learning rate for updates
            
        Returns:
            List of re-embedding results
        """
        results = []
        current_vectors = vectors.copy()
        
        for iteration in range(max_iterations):
            logger.info(f"Starting re-embedding iteration {iteration + 1}/{max_iterations}")
            
            # Process vectors in batches
            batch_results = []
            for i in range(0, len(current_vectors), self.batch_size):
                batch = current_vectors[i:i + self.batch_size]
                batch_results.extend(
                    self._process_batch(batch, current_vectors, learning_rate)
                )
            
            # Update vectors with new embeddings
            current_vectors = [result.vector for result in batch_results]
            results.extend(batch_results)
            
            # Check for convergence
            if self._check_convergence(batch_results):
                logger.info(f"Convergence reached after {iteration + 1} iterations")
                break
                
        return results
        
    def _process_batch(
        self,
        batch: List[Vector],
        all_vectors: List[Vector],
        learning_rate: float
    ) -> List[ReembeddingResult]:
        """Process a batch of vectors.
        
        Args:
            batch: Batch of vectors to process
            all_vectors: All vectors in the system
            learning_rate: Learning rate for updates
            
        Returns:
            List of re-embedding results for the batch
        """
        results = []
        
        for vector in batch:
            # Get context vectors
            context_vectors = self._get_context_vectors(vector, all_vectors)
            
            # Calculate influence metrics
            metrics = self.influence_calculator.calculate_metrics(
                vector.data,
                vector.metadata,
                [(v.id, v.data) for v in context_vectors]
            )
            
            # Recalculate position
            position_info = self.position_encoder.recalculate_position(
                vector.data,
                [(v.id, v.data, v.metadata) for v in context_vectors],
                vector.metadata.get('position', 0)
            )
            
            # Update vector position
            new_data = self.position_encoder.update_position(
                vector.data,
                position_info,
                learning_rate
            )
            
            # Calculate energy (magnitude of change)
            energy = np.linalg.norm(new_data - vector.data)
            
            # Create updated vector
            updated_vector = Vector(
                id=vector.id,
                data=new_data,
                metadata=vector.metadata.copy()
            )
            
            # Store result
            results.append(ReembeddingResult(
                vector=updated_vector,
                metrics=metrics,
                position_info=position_info,
                energy=energy
            ))
            
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
        
    def _check_convergence(self, results: List[ReembeddingResult]) -> bool:
        """Check if the re-embedding process has converged.
        
        Args:
            results: List of re-embedding results
            
        Returns:
            True if converged, False otherwise
        """
        if not results:
            return True
            
        # Calculate average energy
        avg_energy = np.mean([result.energy for result in results])
        
        # Check if below threshold
        return avg_energy < self.convergence_threshold
        
    def close(self):
        """Clean up resources."""
        self.executor.shutdown() 