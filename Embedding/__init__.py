""" Contextual Vector Database Embedding Package

This package provides a comprehensive embedding system for the Contextual Vector Database,
including specialized embedders for different data types and a unified interface for
coordinating them.

Author: Carlos D. Almeida
"""

from .text_embedding import TextEmbedding
from .transaction_embedding import TransactionEmbedding
from .time_stamp_embedding import TimeStampEmbedding
from .data_object_embedding import DataObjectEmbedding
from .multimodal_embedding import MultimodalEmbedding
from .vector_relationship import VectorRelationship
from .unified_embedding import UnifiedEmbedding

__version__ = "1.0.0"

__all__ = [
    'TextEmbedding',
    'TransactionEmbedding',
    'TimeStampEmbedding',
    'DataObjectEmbedding',
    'MultimodalEmbedding',
    'VectorRelationship',
    'UnifiedEmbedding'
]

def create_embedder(max_vector_size: int = 1024, normalize: bool = True) -> UnifiedEmbedding:
    """
    Create a UnifiedEmbedding instance with default settings.
    
    Args:
        max_vector_size: Maximum size of the combined vector
        normalize: Whether to normalize vectors
        
    Returns:
        UnifiedEmbedding: Configured embedder instance
    """
    return UnifiedEmbedding(
        max_vector_size=max_vector_size,
        normalize=normalize
    )

# Example usage
if __name__ == "__main__":
    # Create embedder
    embedder = create_embedder()
    
    # Example: Embed text
    text_vector_id = embedder.embed_data(
        "Sample text for embedding",
        "text",
        metadata={'source': 'example'}
    )
    
    print(f"Created text vector with ID: {text_vector_id}")

