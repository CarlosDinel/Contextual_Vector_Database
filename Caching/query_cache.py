import numpy as np  
from typing import Dict, List, Optional
import sys
import os   
# Voeg het pad naar de modules toe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Embedding')))
from Embedding.text_embedding import TextEmbedding


class VectorRetriever: 

    def __init__(self):
        """
        Initializes the VectorRetriever with a dictionary to store vectors.
        """
        self.data = {}  # Dictionary to store vectors

    def create_matching_vector(self, data: Dict[str, str]) -> List[float]:
        """
        Creates a vector representation for the given data.

        Args:
            data (Dict[str, str]): A dictionary containing the data to be vectorized.

        Returns:
            List[float]: The vector representation of the given data.
        """
        text_embedder = TextEmbedding()
        search_vector = text_embedder.generate_vector(data)
        return search_vector    
    

    def retrieve_closest_vector(self, search_vector: List[float]) -> Optional[str]:
        """
        Retrieves the closest vector to the search vector.

        Args:
            search_vector (List[float]): The vector to search for.

        Returns:
            Optional[str]: The key of the closest vector, or None if no vectors are cached.
        """
        if not self.data:
            return None
        
        best_match = None
        best_score = -1  # Cosine similarity varieert tussen -1 en 1
        
        for key, vector in self.data.items():
            similarity = self.cosine_similarity(search_vector, vector)
            if similarity > best_score:
                best_score = similarity
                best_match = key
        
        return best_match
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculates the cosine similarity between two vectors.

        Args:
            vec1 (List[float]): The first vector.
            vec2 (List[float]): The second vector.

        Returns:
            float: The cosine similarity between the two vectors.
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def cache_vectors(self, data: Dict[str, List[float]]):
        """
        Caches the vectors in the retriever.

        Args:
            data (Dict[str, List[float]]): A dictionary of vectors to cache.
        """
        self.data = data    

    

 # Example usage
if __name__ == "__main__":
    retriever = VectorRetriever()

    # Cache some vectors
    vectors = {
        "vector1": [1, 2, 3],
        "vector2": [4, 5, 6],
        "vector3": [7, 8, 9]
    }
    retriever.cache_vectors(vectors)

    # Retrieve the closest vector
    search_vector = [1, 2, 3]
    closest_vector_key = retriever.retrieve_closest_vector(search_vector)
    print(f"The closest vector to {search_vector} is {closest_vector_key}")    