"""Base model for the Contextual Vector Database."""

import numpy as np
from typing import Dict, Any, Optional
import time

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