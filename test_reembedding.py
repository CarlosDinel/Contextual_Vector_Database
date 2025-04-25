#!/usr/bin/env python3

import numpy as np
from Contexter.reembedding_orchestra import ReembedingOrchestrator
from Contexter.base_model import Vector

def main():
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

if __name__ == "__main__":
    main() 