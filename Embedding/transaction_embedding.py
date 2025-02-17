"""Creates an embedding for time stamps, these vector are crucial for transactional vector.
Using timestamp vectors, these transactions will be added to the data object vector.
In this way, the data object vector will contain information about the transactions made by the data object."""

class TimeStampEmbedding:   
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
        try:
            return self.encode(transaction_data['timestamp'])
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    def extract_features(self, transaction_data):
            """Extract features from the transaction data.

                Args:
                    transaction_data (dict): A dictionary containing transaction data, including a 'timestamp' key.

                Returns:
                    dict: A dictionary of extracted features from the transaction timestamp. """
            
            if not transaction_data:
                raise ValueError("Transaction data is empty")
            try:
                return self.extract_features(transaction_data['timestamp'])
            except Exception as e:
                print(f"Error: {e}")
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
            return self.generate_vector(features)
        except Exception as e:
            print(f"Error: {e}")
            return None 
    
    def combineToTransactionVector(self, transaction_vector, product_vector, time_stamp_vector):
        """
        Combine transaction, product, and timestamp vectors into a single transaction vector.

        Args:
            transaction_vector (numpy.ndarray): The vector representation of the transaction.
            product_vector (numpy.ndarray): The vector representation of the product.
            time_stamp_vector (numpy.ndarray): The vector representation of the timestamp.

        Returns:
            numpy.ndarray: The combined vector representation of the transaction, product, and timestamp.
        """
        if not transaction_vector or not product_vector or not time_stamp_vector:
            raise ValueError("Vectors are empty")
        try:
            return self.combine_vectors(transaction_vector, product_vector, time_stamp_vector)
        except Exception as e:
            print(f"Error: {e}")
            return None
        
