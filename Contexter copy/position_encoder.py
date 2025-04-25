import numpy as np
import logging

logger = logging.getLogger(__name__)

class PositionEncoder:
    """
    Encodes and decodes vector positions in the contextual space.
    """
    def __init__(self, dimensions=3, encoding_method='direct'):
        """
        Initialize the position encoder.
        
        Args:
            dimensions: Number of dimensions in the vector space
            encoding_method: Method to encode positions ('direct', 'normalized', 'spherical')
        """
        self.dimensions = dimensions
        self.encoding_method = encoding_method
        
    def encode_position(self, vector_data):
        """
        Encode vector data into a position representation.
        
        Args:
            vector_data: Raw vector data
            
        Returns:
            Encoded position
        """
        if self.encoding_method == 'direct':
            return self._direct_encoding(vector_data)
        elif self.encoding_method == 'normalized':
            return self._normalized_encoding(vector_data)
        elif self.encoding_method == 'spherical':
            return self._spherical_encoding(vector_data)
        else:
            logger.warning(f"Unknown encoding method: {self.encoding_method}, using direct encoding")
            return self._direct_encoding(vector_data)
    
    def decode_position(self, encoded_position):
        """
        Decode position representation back to vector data.
        
        Args:
            encoded_position: Encoded position
            
        Returns:
            Decoded vector data
        """
        if self.encoding_method == 'direct':
            return self._direct_decoding(encoded_position)
        elif self.encoding_method == 'normalized':
            return self._normalized_decoding(encoded_position)
        elif self.encoding_method == 'spherical':
            return self._spherical_decoding(encoded_position)
        else:
            logger.warning(f"Unknown encoding method: {self.encoding_method}, using direct decoding")
            return self._direct_decoding(encoded_position)
    
    def _direct_encoding(self, vector_data):
        """Direct encoding - no transformation"""
        return np.array(vector_data)
    
    def _direct_decoding(self, encoded_position):
        """Direct decoding - no transformation"""
        return np.array(encoded_position)
    
    def _normalized_encoding(self, vector_data):
        """Normalize vector to unit length"""
        vector = np.array(vector_data)
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def _normalized_decoding(self, encoded_position):
        """No special decoding needed for normalized vectors"""
        return np.array(encoded_position)
    
    def _spherical_encoding(self, vector_data):
        """Convert to spherical coordinates"""
        vector = np.array(vector_data)
        
        # Only implemented for 2D and 3D vectors
        if len(vector) == 2:
            r = np.linalg.norm(vector)
            theta = np.arctan2(vector[1], vector[0])
            return np.array([r, theta])
        elif len(vector) == 3:
            r = np.linalg.norm(vector)
            if r == 0:
                return np.array([0, 0, 0])
            theta = np.arccos(vector[2] / r)
            phi = np.arctan2(vector[1], vector[0])
            return np.array([r, theta, phi])
        else:
            # Fall back to direct encoding for higher dimensions
            logger.warning(f"Spherical encoding not implemented for {len(vector)} dimensions, using direct encoding")
            return vector
    
    def _spherical_decoding(self, encoded_position):
        """Convert from spherical coordinates back to Cartesian"""
        encoded = np.array(encoded_position)
        
        # Only implemented for 2D and 3D vectors
        if len(encoded) == 2:
            r, theta = encoded
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return np.array([x, y])
        elif len(encoded) == 3:
            r, theta, phi = encoded
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            return np.array([x, y, z])
        else:
            # Fall back to direct decoding for higher dimensions
            logger.warning(f"Spherical decoding not implemented for {len(encoded)} dimensions, using direct decoding")
            return encoded
