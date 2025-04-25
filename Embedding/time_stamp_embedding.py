""" Time Stamp Embedding Module for Contextual Vector Database

This module implements the timestamp embedding component that converts
temporal data into vector representations.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import Union
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeStampEmbedding:
    """
    Handles embedding of timestamp data into vector representations.
    """
    def __init__(self, vector_size: int = 32):
        """
        Initialize the timestamp embedder.
        
        Args:
            vector_size: Size of the output vector
        """
        self.vector_size = vector_size
        
    def embed(self, timestamp: Union[str, datetime]) -> np.ndarray:
        """
        Embed a timestamp into a vector representation that preserves temporal proximity.
        
        Args:
            timestamp: ISO format string or datetime object
            
        Returns:
            np.ndarray: Vector representation of the timestamp
        """
        try:
            # Convert string to datetime if needed
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
                
            # Extract temporal features with better normalization
            # Use cyclical encoding for periodic features
            year = timestamp.year / 10000.0  # Normalize year
            
            # Cyclical encoding for periodic features
            month_sin = np.sin(2 * np.pi * timestamp.month / 12.0)
            month_cos = np.cos(2 * np.pi * timestamp.month / 12.0)
            
            day_sin = np.sin(2 * np.pi * timestamp.day / 31.0)
            day_cos = np.cos(2 * np.pi * timestamp.day / 31.0)
            
            hour_sin = np.sin(2 * np.pi * timestamp.hour / 24.0)
            hour_cos = np.cos(2 * np.pi * timestamp.hour / 24.0)
            
            minute_sin = np.sin(2 * np.pi * timestamp.minute / 60.0)
            minute_cos = np.cos(2 * np.pi * timestamp.minute / 60.0)
            
            second_sin = np.sin(2 * np.pi * timestamp.second / 60.0)
            second_cos = np.cos(2 * np.pi * timestamp.second / 60.0)
            
            # Cyclical encoding for weekday
            weekday_sin = np.sin(2 * np.pi * timestamp.weekday() / 7.0)
            weekday_cos = np.cos(2 * np.pi * timestamp.weekday() / 7.0)
            
            # Create base vector with cyclical features
            base_vector = np.array([
                year,
                month_sin, month_cos,
                day_sin, day_cos,
                hour_sin, hour_cos,
                minute_sin, minute_cos,
                second_sin, second_cos,
                weekday_sin, weekday_cos
            ])
            
            # Pad or truncate to desired size
            if len(base_vector) < self.vector_size:
                # Pad with zeros
                vector = np.pad(base_vector, (0, self.vector_size - len(base_vector)))
            else:
                # Truncate
                vector = base_vector[:self.vector_size]
                
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
                
            return vector
            
        except Exception as e:
            logger.error(f"Failed to embed timestamp: {e}")
            raise

    def decode(self, vector):
        """
        Convert a vector back into a time stamp.

        Args:
            vector (numpy.ndarray): The vector to be converted back into a time stamp.

        Returns:
            datetime: The resulting time stamp.
        """
        year = int(vector[0] * 3000)
        month = int(vector[1] * 12)
        day = int(vector[2] * 31)
        hour = int(vector[3] * 24)
        minute = int(vector[4] * 60)
        second = int(vector[5] * 60)
        
        return datetime(year, month, day, hour, minute, second)

    def extract_features(self, time_stamp):
        """
        Extract features from the time stamp.

        Args:
            time_stamp (str or datetime): The time stamp to extract features from.

        Returns:
            dict: A dictionary of extracted features from the time stamp.
        """
        if isinstance(time_stamp, str):
            time_stamp = datetime.fromisoformat(time_stamp)
        
        features = {
            'year': time_stamp.year,
            'month': time_stamp.month,
            'day': time_stamp.day,
            'hour': time_stamp.hour,
            'minute': time_stamp.minute,
            'second': time_stamp.second,
        }
        
        return features

# Example usage
if __name__ == "__main__":
    # Create timestamp embedder
    embedder = TimeStampEmbedding()
    
    # Example timestamp
    timestamp = datetime.now().isoformat()
    
    # Embed the timestamp
    vector = embedder.embed(timestamp)
    
    print(f"Timestamp vector shape: {vector.shape}")
    print(f"Timestamp vector: {vector[:5]}")  # Print first 5 components

