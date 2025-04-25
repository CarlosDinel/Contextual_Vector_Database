# """ Text Embedding Module for Contextual Vector Database

# This module implements the text embedding component that converts
# text data into vector representations using sentence transformers.

# Author: Carlos D. Almeida
# """

# import numpy as np
# from typing import Union, List
# import logging
# from sentence_transformers import SentenceTransformer

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# customer_data = {
#     "name": "Pieter de Vries",
#     "age": 35,
#     "location": "Amsterdam",
#     "aankoop": "Laptop",
#     "prijs": 999.99,
#     "datum": "2023-06-15"
# }

# class TextEmbedding:
#     """
#     Handles embedding of text data into vector representations.
#     """
#     def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
#         """
#         Initialize the text embedder.
        
#         Args:
#             model_name: Name of the sentence transformer model to use
#         """
#         try:
#             self.model = SentenceTransformer(model_name)
#         except Exception as e:
#             logger.error(f"Failed to initialize sentence transformer: {e}")
#             raise
            
#     def embed(self, text: Union[str, List[str]]) -> np.ndarray:
#         """
#         Embed text into a vector representation.
        
#         Args:
#             text: Text or list of texts to embed
            
#         Returns:
#             np.ndarray: Vector representation of the text
#         """
#         try:
#             # Convert single text to list
#             if isinstance(text, str):
#                 text = [text]
                
#             # Get embeddings
#             embeddings = self.model.encode(text)
            
#             # If multiple texts, average the embeddings
#             if len(embeddings) > 1:
#                 embeddings = np.mean(embeddings, axis=0)
                
#             return embeddings
            
#         except Exception as e:
#             logger.error(f"Failed to embed text: {e}")
#             raise

#     def encode_text(self, text):
#         if not text: 
#             raise ValueError("Text is empty")
#         try:
#             return self.model.encode(text)
#         except Exception as e:
#             print(f"Error: {e}")
#             return None

#     def encode_customer(self, customer_data):   
#         text_representation = f"Name: {customer_data['name']}, Age: {customer_data['age']}, Location: {customer_data['location']}, Aankoop: {customer_data['aankoop']}, Prijs: {customer_data['prijs']}, Datum: {customer_data['datum']}"
#         customer_vector = self.encode_text(text_representation)
#         return customer_vector  
    
#     def decode_vector(self, vector):
#         # This method would require a reverse mapping from vectors to text, which is not straightforward.
#         # For simplicity, we'll return a placeholder text.
#         return "Decoded text from vector"

#     def batch_encode(self, texts):
#         if not texts:
#             raise ValueError("Input texts cannot be empty")
#         try:
#             return self.model.encode(texts)
#         except Exception as e:  
#             print(f"Error: {e}")
#             return None

#     def generate_vector(self, features):
#         """
#         Generate a vector from the extracted features.

#         Args:
#             features (dict): A dictionary of extracted features.

#         Returns:
#             numpy.ndarray: The generated vector representation of the features.
#         """
#         text_representation = ", ".join([f"{key}: {value}" for key, value in features.items()])
#         return self.encode_text(text_representation)

#     def update_model(self, model_name):
#         try:
#             self.model = SentenceTransformer(model_name)
#             print(f"Model updated to {model_name}")
#         except Exception as e:
#             print(f"Error: {e}")
#             return None

#     def save_model(self, path):
#         try:
#             self.model.save(path)
#             print(f"Model saved to {path}")
#         except Exception as e:
#             print(f"Error: {e}")
#             return None

#     def load_model(self, path): 
#         try:
#             self.model = SentenceTransformer(path)
#             print(f"Model loaded from {path}")
#         except Exception as e:
#             print(f"Error: {e}")
#             return None

# # Example usage
# if __name__ == "__main__":
#     # Create text embedder
#     embedder = TextEmbedding()
    
#     # Example text
#     text = "This is a sample text for embedding"
    
#     # Embed the text
#     vector = embedder.embed(text)
    
#     print(f"Text vector shape: {vector.shape}")
#     print(f"Text vector: {vector[:5]}")  # Print first 5 components

import numpy as np
import logging
from typing import Union, List
from sentence_transformers import SentenceTransformer

# Logger setup
logger = logging.getLogger(__name__)

class TextEmbedding:
    """
    Handles embedding of text data into vector representations.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the text embedder.
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize sentence transformer: {e}")
            raise
            
    def embed(self, text: Union[str, List[str]], aggregate: bool = True) -> np.ndarray:
        """
        Embed text into a vector representation.
        
        Args:
            text: Text or list of texts to embed
            aggregate: If True, returns the mean vector for multiple inputs
            
        Returns:
            np.ndarray: Vector representation of the text
        """
        try:
            # Ensure text is a list
            if isinstance(text, str):
                text = [text]
            if not isinstance(text, list) or not all(isinstance(t, str) for t in text):
                raise ValueError("Input must be a string or list of strings.")
            
            embeddings = self.model.encode(text)

            # Aggregate embeddings if multiple texts provided
            if aggregate and len(embeddings) > 1:
                return np.mean(embeddings, axis=0)

            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text input.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text input must be a non-empty string.")
        return self.embed(text, aggregate=False)

    def encode_customer(self, customer_data: dict) -> np.ndarray:
        """
        Generate an embedding for customer data.
        """
        try:
            text_representation = ", ".join([f"{key}: {value}" for key, value in customer_data.items()])
            return self.encode_text(text_representation)
        except Exception as e:
            logger.error(f"Failed to encode customer data: {e}")
            return None

    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple text inputs.
        """
        if not texts or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a non-empty list of strings.")
        return self.embed(texts, aggregate=False)

    def update_model(self, model_name: str):
        """
        Update the embedding model.
        """
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Model updated to {model_name}")
        except Exception as e:
            logger.error(f"Error updating model: {e}")

    def save_model(self, path: str):
        """
        Save the model.
        """
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self, path: str):
        """
        Load a saved model.
        """
        try:
            self.model = SentenceTransformer(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

# Example usage
if __name__ == "__main__":
    embedder = TextEmbedding()
    text = "This is a sample text for embedding"
    vector = embedder.embed(text)

    print(f"Text vector shape: {vector.shape}")
    print(f"Text vector: {vector[:5]}")  # Print first 5 components
