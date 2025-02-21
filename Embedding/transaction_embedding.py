
"""Creates an embedding for time stamps, these vector are crucial for transactional vector.
Using timestamp vectors, these transactions will be added to the data object vector.
In this way, the data object vector will contain information about the transactions made by the data object."""

import time_stamp_embedding
import text_embedding
import numpy as np  

class TransactionEmbedding:   
    def __init__(self, vector_size=10):
        self.vector_size = vector_size 
        self.vector = None  



    def encode_transaction(self, transaction_data):
        """Encode the transaction data into a vector.

            Args:
                transaction_data (dict): A dictionary containing transaction data, including a 'timestamp' key.

            Returns:
                numpy.ndarray: The encoded vector representation of the transaction timestamp."""
        if not transaction_data:
            raise ValueError("Transaction data is empty")
        if 'timestamp' not in transaction_data:
            raise ValueError("Transaction data must include a 'timestamp' key")
        try:
            ts_embedder = time_stamp_embedding.TimeStampEmbedding() 
            return ts_embedder.encode(transaction_data['timestamp'])
        except Exception as e:
            print(f"Error1: {e}")
            return None
        
    def vector_search (self, data_objec_identification):
            """searches the vector where the transaction is destinated to go to """
            return None 
        
    def extract_features(self, transaction_data):
        """Extract features from the transaction data.

        Args:
            transaction_data (dict): A dictionary containing transaction data, including a 'timestamp' key.

        Returns:
            dict: A dictionary of extracted features from the transaction timestamp."""
        if not transaction_data:
            raise ValueError("Transaction data is empty")
        if 'timestamp' not in transaction_data:
            raise ValueError("Transaction data must include a 'timestamp' key")
        try:
            ts_embedder = time_stamp_embedding.TimeStampEmbedding()
            features = ts_embedder.extract_features(transaction_data['timestamp'])
            features['timestamp'] = transaction_data['timestamp']
            return features
        except Exception as e:
            print(f"Error2: {e}")
            return None
                
                
    def generate_transaction_vector(self, features):
        """
        Generate a transaction vector from the extracted features.

        Args:
            features (dict): A dictionary of extracted features.

        Returns:
            numpy.ndarray: The generated vector representation of the transaction features.
        """
        if not features:
            raise ValueError("Features are empty")
        try:
            text_embedder = text_embedding.TextEmbedding()  
            return text_embedder.generate_vector(features)
        except Exception as e:
            print(f"Error3: {e}")
            return None 
    
    def combine_to_transaction_vector(self, transaction_vector, product_vector, time_stamp_vector):
        """
        Combine transaction, product, and timestamp vectors into a single transaction vector.

        Args:
            transaction_vector (numpy.ndarray): The vector representation of the transaction.
            product_vector (numpy.ndarray): The vector representation of the product.
            time_stamp_vector (numpy.ndarray): The vector representation of the timestamp.

        Returns:
            numpy.ndarray: The combined vector representation of the transaction, product, and timestamp.
        """
        if not transaction_vector.any() or not product_vector.any() or not time_stamp_vector.any():
            raise ValueError("Vectors are empty")
        try:
            return np.concatenate([transaction_vector, product_vector, time_stamp_vector])
        except Exception as e:
            print(f"Error4: {e}")
            return None
        
