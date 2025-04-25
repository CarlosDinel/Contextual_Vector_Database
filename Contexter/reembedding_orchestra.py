""" Reembedding Orchestrator Module for Contextual Vector Database

This module implements the reembedding orchestrator component of the Contextual Vector Database (CVD),
which coordinates the reembedding process between different components.

Author: Carlos D. Almeida
"""

import os
import sys
import time
import logging
from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime   

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import related components
try:
    from .base_model import Vector
    from .contexter_model import Contexter
    from .context_aggregator import ContextAggregator
    from .influence_calculator import InfluenceCalculator
    from .position_encoder import PositionEncoder
except ImportError:
    from Contexter.base_model import Vector
    from Contexter.contexter_model import Contexter
    from Contexter.context_aggregator import ContextAggregator
    from Contexter.influence_calculator import InfluenceCalculator
    from Contexter.position_encoder import PositionEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReembedingOrchestrator:
    """
    Main class for orchestrating the reembedding process.
    """
    def __init__(self, 
                 reembedding_interval: float = 3600.0,
                 batch_size: int = 100,
                 max_iterations: int = 10):
        """
        Initialize the ReembeddingOrchestrator.
        
        Args:
            reembedding_interval: Time between reembedding operations in seconds
            batch_size: Number of vectors to process in each batch
            max_iterations: Maximum number of iterations for convergence
        """
        self.reembedding_interval = reembedding_interval
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.last_reembedding_time = None
        
        # Initialize components
        self.influence_calculator = InfluenceCalculator()
        self.context_aggregator = ContextAggregator()
        self.contexter = Contexter(
            influence_calculator=self.influence_calculator,
            context_aggregator=self.context_aggregator
        )
        self.position_encoder = PositionEncoder()
        
        # Set up circular references
        self.influence_calculator.contexter = self.contexter
        
    def should_reembed(self) -> bool:
        """
        Check if reembedding should be performed based on the time interval.
        
        Returns:
            True if reembedding should be performed, False otherwise
        """
        current_time = datetime.now()
        
        if self.last_reembedding_time is None:
            return True
        
        time_diff = (current_time - self.last_reembedding_time).total_seconds()
        return time_diff >= self.reembedding_interval
    
    def get_vectors_for_reembedding(self, all_vectors: List[Vector]) -> List[Vector]:
        """
        Get vectors that should be reembedded.
        
        Args:
            all_vectors: List of all vectors
            
        Returns:
            List of vectors for reembedding
        """
        # This is a simplified implementation
        # In a real system, this would select vectors based on various criteria
        # For now, we just take a batch of vectors
        return all_vectors[:min(self.batch_size, len(all_vectors))]
    
    def prepare_context(self, vectors: List[Vector]) -> Dict[str, Dict[str, Any]]:
        """
        Prepare context information for vectors.
        
        Args:
            vectors: List of vectors
            
        Returns:
            Dictionary mapping vector IDs to context information
        """
        context_info = {}
        
        for vector in vectors:
            # Prepare different types of context
            # This is a simplified implementation
            temporal_context = {'timestamp': datetime.now().isoformat()}
            semantic_context = {'keywords': vector.metadata.get('keywords', [])}
            structural_context = {'connections': vector.metadata.get('connections', [])}
            
            # Aggregate context
            aggregated_context = self.context_aggregator.aggregate_context(
                vector, temporal_context, semantic_context, structural_context)
            
            context_info[vector.id] = aggregated_context
        
        return context_info
    
    def calculate_influences(self, 
                            vectors: List[Vector], 
                            context_info: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate influences between vectors.
        
        Args:
            vectors: List of vectors
            context_info: Dictionary mapping vector IDs to context information
            
        Returns:
            Dictionary mapping vector IDs to dictionaries mapping other vector IDs to influence scores
        """
        influences = {}
        
        for vector_i in vectors:
            vector_influences = {}
            
            for vector_j in vectors:
                if vector_i.id != vector_j.id:
                    # Get context for vector_i
                    context = context_info.get(vector_i.id, {})
                    
                    # Calculate influence
                    influence = self.influence_calculator.calculate_influence(
                        vector_i, vector_j, context)
                    
                    # Store influence
                    vector_influences[vector_j.id] = influence
            
            influences[vector_i.id] = vector_influences
        
        return influences
    
    def reembed_vectors(self, 
                       vectors: List[Vector], 
                       context_info: Dict[str, Dict[str, Any]],
                       influences: Dict[str, Dict[str, float]]) -> List[Vector]:
        """
        Reembed vectors using the contexter.
        
        Args:
            vectors: List of vectors
            context_info: Dictionary mapping vector IDs to context information
            influences: Dictionary mapping vector IDs to dictionaries mapping other vector IDs to influence scores
            
        Returns:
            List of reembedded vectors
        """
        # Use the contexter to reembed vectors
        reembedded_vectors = self.contexter.reembed_vectors(vectors)
        
        # Update last reembedding time
        self.last_reembedding_time = datetime.now()
        
        return reembedded_vectors
    
    def orchestrate_reembedding(self, all_vectors: List[Vector]) -> List[Vector]:
        """
        Orchestrate the reembedding process following the flow:
        1. InfluenceCalculator determines vector influences
        2. ContextAggregator creates context
        3. ContexterModel determines repositioning
        4. PositionEncoder applies repositioning
        
        Args:
            all_vectors: List of all vectors
            
        Returns:
            List of all vectors after reembedding
        """
        if not self.should_reembed():
            logger.info("Skipping reembedding, not enough time has passed since last reembedding")
            return all_vectors
        
        logger.info("Starting reembedding process")
        
        # Get vectors for reembedding
        vectors_to_reembed = self.get_vectors_for_reembedding(all_vectors)
        logger.info(f"Selected {len(vectors_to_reembed)} vectors for reembedding")
        
        # Step 1: Calculate influences between vectors
        influences = self.influence_calculator.calculate_influences(vectors_to_reembed)
        logger.info("Calculated influences between vectors")
        
        # Step 2: Create context for vectors
        context_info = self.context_aggregator.prepare_context(all_vectors, vectors_to_reembed)
        logger.info("Prepared context information")
        
        # Step 3: Determine repositioning using contexter
        repositioning_info = self.contexter.determine_context(vectors_to_reembed)
        logger.info("Determined vector repositioning")
        
        # Step 4: Apply repositioning using position encoder
        reembedded_vectors = []
        for vector in vectors_to_reembed:
            # Get repositioning information for this vector
            vector_repositioning = repositioning_info.get(vector.id, [])
            
            # Encode current position
            current_position = self.position_encoder.encode_position(vector.data)
            
            # Apply repositioning
            new_position = self.position_encoder.decode_position(
                self._apply_repositioning(current_position, vector_repositioning)
            )
            
            # Create updated vector
            updated_vector = Vector(
                id=vector.id,
                data=new_position,
                solidness=vector.solidness,
                impact_radius=vector.impact_radius,
                metadata=vector.metadata
            )
            reembedded_vectors.append(updated_vector)
        
        logger.info("Applied vector repositioning")
        
        # Update vectors in the original list
        reembedded_dict = {v.id: v for v in reembedded_vectors}
        updated_all_vectors = []
        
        for vector in all_vectors:
            if vector.id in reembedded_dict:
                updated_all_vectors.append(reembedded_dict[vector.id])
            else:
                updated_all_vectors.append(vector)
        
        # Update last reembedding time
        self.last_reembedding_time = datetime.now()
        
        return updated_all_vectors
    
    def _apply_repositioning(self, current_position, repositioning_info):
        """
        Apply repositioning information to current position.
        
        Args:
            current_position: Current encoded position
            repositioning_info: List of (vector_id, impact) tuples for repositioning
            
        Returns:
            New encoded position
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated repositioning logic
        new_position = current_position.copy()
        
        for _, impact in repositioning_info:
            # Apply impact to position
            new_position += impact
            
        return new_position
    
    def run_continuous_reembedding(self, all_vectors: List[Vector], max_iterations: int = 10) -> List[Vector]:
        """
        Run continuous reembedding for a specified number of iterations.
        
        Args:
            all_vectors: List of all vectors
            max_iterations: Maximum number of iterations
            
        Returns:
            List of all vectors after reembedding
        """
        current_vectors = all_vectors.copy()
        
        for i in range(max_iterations):
            logger.info(f"Reembedding iteration {i+1}/{max_iterations}")
            
            # Orchestrate reembedding
            current_vectors = self.orchestrate_reembedding(current_vectors)
            
            # Wait for the next interval
            if i < max_iterations - 1:
                logger.info(f"Waiting for next reembedding interval ({self.reembedding_interval} seconds)")
                # In a real system, this would be handled by a scheduler
                # For testing, we just set a small interval
                time.sleep(1)  # Use a small value for testing
        
        return current_vectors


# Example usage
if __name__ == "__main__":
    # Create some test vectors
    vectors = [
        Vector("v1", np.array([1.0, 0.0, 0.0]), metadata={'keywords': ['red', 'apple'], 'connections': ['v2']}),
        Vector("v2", np.array([0.8, 0.2, 0.0]), metadata={'keywords': ['orange', 'fruit'], 'connections': ['v1', 'v3']}),
        Vector("v3", np.array([0.0, 1.0, 0.0]), metadata={'keywords': ['green', 'leaf'], 'connections': ['v2']}),
        Vector("v4", np.array([0.0, 0.0, 1.0]), metadata={'keywords': ['blue', 'sky'], 'connections': []}),
    ]
    
    # Create orchestrator
    orchestrator = ReembedingOrchestrator(
        reembedding_interval=1,  # 1 second for testing
        batch_size=100
    )
    
    # Run a single reembedding operation
    reembedded_vectors = orchestrator.orchestrate_reembedding(vectors)
    print("\nReembedded Vectors:")
    for vector in reembedded_vectors:
        print(f"{vector.id}: {vector.data}, solidness={vector.solidness:.2f}")
    
    # Run continuous reembedding (with a small number of iterations for testing)
    final_vectors = orchestrator.run_continuous_reembedding(vectors, max_iterations=2)
    print("\nFinal Vectors after Continuous Reembedding:")
    for vector in final_vectors:
        print(f"{vector.id}: {vector.data}, solidness={vector.solidness:.2f}")
