import sys
import os

# Voeg het pad naar de modules toe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Embedding')))

import text_embedding 
import time_stamp_embedding 
import numpy as np
import data_object_embedding 
import transaction_embedding    
from datetime import datetime


if __name__ == "__main__":
    # Create instances of the embedding classes
    customer_embedder = text_embedding.TextEmbedding()
    time_stamp_embedder = time_stamp_embedding.TimeStampEmbedding()

    # Encode customer data and time stamp
    customer_data = "voorbeeld customer data"  # Define customer_data
    timestamp = "2024-01-01"  # Define timestamp
    customer_vector = customer_embedder.encode_customer(customer_data)
    time_stamp_vector = time_stamp_embedder.encode(timestamp)

    # Combine vectors
    base_vector = np.array([0.1] * 200)  # Voorbeeld base vector
    data_object_embedding = data_object_embedding.DataObjectEmbedding(base_vector=base_vector, max_vector_size=300)
    combined_vector = data_object_embedding.combine_vectors(customer_vector, time_stamp_vector)
    print(combined_vector)

    # Calculate solidness
    data_object_embedding.calculate_solidness()
    solidness = data_object_embedding.solidness
    print(f"solidness: {solidness}")