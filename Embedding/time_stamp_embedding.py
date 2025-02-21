import numpy as np
from datetime import datetime

class TimeStampEmbedding:
    """
    Time stamp embedding class for converting time stamps into vectors.
    These vectors are crucial for transactional vectors and contain
    information about transactions made by a dataobject.
    """
    def __init__(self, vector_size=10):
        """
        Initialize the TimeStampEmbedding class.

        Args:
            vector_size (int): Size of the resulting vector. Default is 10.
        """
        self.vector_size = vector_size

    def encode(self, time_stamp):
        """
        Convert a time stamp into a vector.

        Args:
            time_stamp (str or datetime): The time stamp to be converted into a vector.

        Returns:
            numpy.ndarray: The resulting vector representation of the time stamp.
        """
        if isinstance(time_stamp, str):
            time_stamp = datetime.fromisoformat(time_stamp)
        
        # Extract various time components
        year = time_stamp.year
        month = time_stamp.month
        day = time_stamp.day
        hour = time_stamp.hour
        minute = time_stamp.minute
        second = time_stamp.second
        
        # Create a vector representation
        time_stamp_vector = np.array([
            year / 3000,  # Normalize year
            month / 12,   # Normalize month
            day / 31,     # Normalize day
            hour / 24,    # Normalize hour
            minute / 60,  # Normalize minute
            second / 60,  # Normalize second
            np.sin(2 * np.pi * month / 12),  # Cyclical representation of month
            np.cos(2 * np.pi * month / 12),
            np.sin(2 * np.pi * day / 31),    # Cyclical representation of day
            np.cos(2 * np.pi * day / 31)
        ])
        
        return time_stamp_vector

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
# ts_embedder = TimeStampEmbedding()
# timestamp = "2023-06-15 14:30:00"
# encoded = ts_embedder.encode(timestamp)
# decoded = ts_embedder.decode(encoded)

# print(f"Original: {timestamp}")
# print(f"Encoded: {encoded}")
# print(f"Decoded: {decoded}")

