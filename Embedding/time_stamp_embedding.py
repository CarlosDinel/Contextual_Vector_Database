import numpy as np
from sentence_transformers import SentenceTransformer

"""Creates an embedding for time stamps, these vector are crucial for transactional vector. 
These vectors contain information about transactions made by a dataobject."""

class TimeStampEmbedding:
    """
    Time stamp embedding class for converting time stamps into vectors.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the TimeStampEmbedding class with a specific model.

        Args:
            model_name (str): Name of the model used for time stamp embedding.
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, time_stamp):
        """
        Convert a time stamp into a vector.

        Args:
            time_stamp (str): The time stamp to be converted into a vector.

        Returns:
            numpy.ndarray: The resulting vector representation of the time stamp.
        """
        return self.model.encode(time_stamp)    

print(TimeStampEmbedding().encode("2022-01-01"))    