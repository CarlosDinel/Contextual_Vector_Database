""" Multimodal Embedding Module for Contextual Vector Database

This module implements the multimodal embedding component that handles
various types of data (text, images, etc.) and converts them into
unified vector representations.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# Import from local package
from .text_embedding import TextEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalEmbedding:
    """
    Handles embedding of multimodal data into vector representations.
    """
    def __init__(self):
        """Initialize the multimodal embedder."""
        # Initialize text embedder
        self.text_embedder = TextEmbedding()
        
        # Initialize image embedder
        try:
            self.image_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.image_model.eval()  # Set to evaluation mode
            self.image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        except Exception as e:
            logger.error(f"Failed to initialize image model: {e}")
            self.image_model = None
            self.image_transform = None
            
    def embed(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Embed multimodal data into a unified vector representation.
        
        Args:
            data: Dictionary containing text and image data
            
        Returns:
            np.ndarray: Combined vector representation
        """
        try:
            vectors = []
            weights = []
            
            # Embed text if available
            if 'text' in data and data['text']:
                text_vector = self.text_embedder.embed(data['text'])
                vectors.append(text_vector)
                weights.append(0.6)  # Higher weight for text
                
            # Embed image if available
            if 'image' in data and data['image']:
                image_vector = self._embed_image(data['image'])
                if image_vector is not None:
                    vectors.append(image_vector)
                    weights.append(0.4)  # Lower weight for image
                
            if not vectors:
                raise ValueError("No valid data to embed")
                
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Ensure all vectors have the same dimension
            max_dim = max(v.shape[0] for v in vectors)
            normalized_vectors = []
            
            for vector in vectors:
                # Ensure vector is 1D
                vector = vector.flatten()
                
                if vector.shape[0] < max_dim:
                    # Pad with zeros if needed
                    padded = np.zeros(max_dim)
                    padded[:vector.shape[0]] = vector
                    normalized_vectors.append(padded)
                else:
                    normalized_vectors.append(vector)
            
            # Combine vectors
            combined = np.zeros(max_dim)
            for vector, weight in zip(normalized_vectors, weights):
                combined += vector * weight
                
            # Normalize final vector
            return self._normalize_vector(combined)
            
        except Exception as e:
            logger.error(f"Failed to embed multimodal data: {e}")
            raise
            
    def _embed_image(self, image: Any) -> Optional[np.ndarray]:
        """
        Embed an image into a vector representation.
        
        Args:
            image: Image data (PIL Image, numpy array, or file path)
            
        Returns:
            Optional[np.ndarray]: Vector representation of the image
        """
        try:
            if self.image_model is None:
                logger.warning("Image model not initialized")
                return None
                
            # Convert image to PIL Image if needed
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
                
            # Transform image
            image_tensor = self.image_transform(image)
            image_tensor = image_tensor.unsqueeze(0)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.image_model(image_tensor)
                
            # Convert to numpy array and ensure 1D
            vector = embedding.squeeze().numpy()
            vector = vector.flatten()  # Ensure 1D array
            
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
                
            return vector
            
        except Exception as e:
            logger.error(f"Failed to embed image: {e}")
            return None

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: Input vector
            
        Returns:
            np.ndarray: Normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        else:
            return vector

# Example usage
if __name__ == "__main__":
    # Create multimodal embedder
    embedder = MultimodalEmbedding()
    
    # Example multimodal data
    data = {
        'text': "Sample text description",
        'image': None  # Replace with actual image if available
    }
    
    # Embed the data
    vector = embedder.embed(data)
    
    print(f"Multimodal vector shape: {vector.shape}")
    print(f"Multimodal vector: {vector[:5]}")  # Print first 5 components
