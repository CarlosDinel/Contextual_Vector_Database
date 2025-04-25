""" Transaction Embedding Module for Contextual Vector Database

This module implements the transaction embedding component that converts
transaction data into vector representations.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import Dict, Any
import logging
from datetime import datetime

# Import from local package
from .time_stamp_embedding import TimeStampEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransactionEmbedding:
    """
    Handles embedding of transaction data into vector representations.
    """
    def __init__(self):
        """Initialize the transaction embedder."""
        self.timestamp_embedder = TimeStampEmbedding()
        
    def embed(self, transaction_data: Dict[str, Any]) -> np.ndarray:
        """
        Embed transaction data into a vector representation.
        
        Args:
            transaction_data: Dictionary containing transaction information
            
        Returns:
            np.ndarray: Vector representation of the transaction
        """
        # Extract transaction features
        amount = float(transaction_data.get('amount', 0))
        transaction_type = transaction_data.get('type', 'unknown')
        product_id = transaction_data.get('product_id', '')
        category = transaction_data.get('category', '')
        
        # Create base vector
        base_vector = np.zeros(64)  # Adjust size as needed
        
        # Encode amount (normalized)
        base_vector[0] = amount / 1000.0  # Assuming max amount is 1000
        
        # Encode transaction type
        type_encoding = {
            'purchase': 1,
            'refund': -1,
            'transfer': 0.5,
            'unknown': 0
        }
        base_vector[1] = type_encoding.get(transaction_type, 0)
        
        # Encode product ID (simple hash)
        base_vector[2] = hash(product_id) % 1000 / 1000.0
        
        # Encode category (simple hash)
        base_vector[3] = hash(category) % 1000 / 1000.0
        
        # Add timestamp if available
        if 'timestamp' in transaction_data:
            timestamp_vector = self.timestamp_embedder.embed(transaction_data['timestamp'])
            base_vector[4:4+len(timestamp_vector)] = timestamp_vector
            
        # Normalize the vector
        norm = np.linalg.norm(base_vector)
        if norm > 0:
            base_vector = base_vector / norm
            
        return base_vector

# Example usage
if __name__ == "__main__":
    # Create transaction embedder
    embedder = TransactionEmbedding()
    
    # Example transaction data
    transaction = {
        'amount': 150.50,
        'type': 'purchase',
        'product_id': 'PROD123',
        'category': 'electronics',
        'timestamp': datetime.now().isoformat()
    }
    
    # Embed the transaction
    vector = embedder.embed(transaction)
    
    print(f"Transaction vector shape: {vector.shape}")
    print(f"Transaction vector: {vector[:5]}")  # Print first 5 components
