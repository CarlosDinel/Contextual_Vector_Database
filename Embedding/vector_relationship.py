""" Vector Relationship Module for Contextual Vector Database

This module implements the mother-child vector relationship functionality for the
Contextual Vector Database (CVD), which allows vectors to split when they become
too large and maintain relationships between the resulting vectors.

Author: Carlos D. Almeida 
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union, Set
import logging
from datetime import datetime
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorRelationship:
    """
    Manages relationships between vectors, particularly mother-child relationships.
    """
    def __init__(self, 
                 dimension_threshold: int = 100,
                 similarity_threshold: float = 0.8,
                 max_children: int = 5):
        """
        Initialize the vector relationship manager.
        
        Args:
            dimension_threshold: Threshold for vector dimensionality that triggers splitting
            similarity_threshold: Minimum similarity required for vectors to be considered related
            max_children: Maximum number of child vectors a mother vector can have
        """
        self.dimension_threshold = dimension_threshold
        self.similarity_threshold = similarity_threshold
        self.max_children = max_children
        
        # Track relationships between vectors
        self.mother_to_children: Dict[str, List[str]] = {}  # mother_id -> [child_ids]
        self.child_to_mother: Dict[str, str] = {}     # child_id -> mother_id
        self.child_capacity: Dict[str, int] = {}      # child_id -> remaining capacity
        
        logger.info(f"Initialized VectorRelationship with max_children={max_children}")
        
    def should_split_vector(self, vector_data: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """
        Determine if a vector should be split based on its dimensionality and complexity.
        """
        try:
            # 1. Basic dimensionality check
            n_features = len(vector_data)
            if n_features > self.dimension_threshold:
                logger.info(f"Vector exceeds dimension threshold ({n_features} > {self.dimension_threshold})")
                return True
            
            # 2. Enhanced complexity metrics
            # Sparsity (ratio of non-zero elements)
            sparsity = np.count_nonzero(vector_data) / n_features
            
            # Entropy (measure of information density)
            hist, _ = np.histogram(vector_data, bins=50)
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Variance (measure of data spread)
            variance = np.var(vector_data)
            
            # 3. Semantic complexity (if text data)
            semantic_complexity = 0.0
            if metadata and 'text' in metadata:
                text = metadata['text']
                # Count unique words, sentence complexity, etc.
                words = text.split()
                unique_words = len(set(words))
                semantic_complexity = unique_words / len(words)
            
            # 4. Combined complexity score with weights
            complexity_score = (
                0.3 * (1 - sparsity) +  # Higher complexity for dense vectors
                0.2 * (entropy / np.log2(50)) +  # Normalized entropy
                0.2 * (min(variance, 1.0)) +  # Normalized variance
                0.3 * semantic_complexity  # Semantic complexity
            )
            
            # 5. Dynamic threshold based on metadata
            threshold = 0.7  # Default threshold
            if metadata:
                if metadata.get('is_critical', False):
                    threshold = 0.5  # Lower threshold for critical vectors
                if metadata.get('update_frequency', 0) > 100:
                    threshold = 0.6  # Lower threshold for frequently updated vectors
            
            # 6. Final decision
            if complexity_score > threshold:
                logger.info(f"Vector has high complexity score: {complexity_score:.3f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in should_split_vector: {e}")
            return False  # Default to not splitting on error
        
    def split_vector(self, vector_id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, List[np.ndarray], List[str], Dict[str, Any]]:
        """
        Split a vector into multiple child vectors while maintaining relationships.
        
        Args:
            vector_id: ID of the vector to split
            vector: The vector to split
            metadata: Optional metadata for the vector
            
        Returns:
            Tuple containing:
                - Mother vector
                - List of child vectors
                - List of child IDs
                - Critical metadata
        """
        try:
            # Ensure vector is 1D
            vector = vector.flatten()
            n_features = vector.shape[0]
            
            # Calculate optimal number of splits based on dimensionality
            num_splits = min(self.max_children, max(2, n_features // self.dimension_threshold))
            split_size = n_features // num_splits
            
            # Create mother vector (first split)
            mother_vector = vector[:split_size]
            
            # Create child vectors
            child_vectors = []
            child_ids = []
            
            # Initialize metadata if None
            if metadata is None:
                metadata = {}
            
            # Preserve critical metadata
            critical_metadata = {
                'entity_type': metadata.get('entity_type', 'unknown'),
                'class': metadata.get('class', 'unknown'),
                'identity': metadata.get('identity', vector_id),
                'is_mother': True,
                'child_references': []
            }
            
            # Create child vectors
            for i in range(1, num_splits):
                start_idx = i * split_size
                end_idx = start_idx + split_size if i < num_splits - 1 else n_features
                child_vector = vector[start_idx:end_idx]
                
                # Create child ID and metadata
                child_id = f"{vector_id}_child_{i}"
                child_metadata = {
                    'entity_type': critical_metadata['entity_type'],
                    'class': critical_metadata['class'],
                    'identity': critical_metadata['identity'],
                    'is_child': True,
                    'mother_reference': vector_id,
                    'split_index': i,
                    'total_splits': num_splits
                }
                
                child_vectors.append(child_vector)
                child_ids.append(child_id)
                critical_metadata['child_references'].append(child_id)
                
                # Update relationships
                self.mother_to_children[vector_id] = child_ids
                self.child_to_mother[child_id] = vector_id
            
            return mother_vector, child_vectors, child_ids, critical_metadata
            
        except Exception as e:
            logger.error(f"Failed to split vector {vector_id}: {e}")
            raise
        
    def get_family_tree(self, vector_id: str) -> Dict[str, Any]:
        """
        Get the complete family tree for a vector, including all metadata.
        """
        vector_entry = self.get_vector(vector_id)
        if not vector_entry:
            return None
        
        tree = {
            'id': vector_id,
            'metadata': vector_entry['metadata'],
            'children': []
        }
        
        # If this is a mother vector, get all children
        if self.is_mother_vector(vector_id):
            children = self.get_children(vector_id)
            for child_id in children:
                child_tree = self.get_family_tree(child_id)
                if child_tree:
                    tree['children'].append(child_tree)
        
        return tree
        
    def is_mother_vector(self, vector_id: str) -> bool:
        """Check if a vector is a mother vector."""
        return vector_id in self.mother_to_children
        
    def is_child_vector(self, vector_id: str) -> bool:
        """Check if a vector is a child vector."""
        return vector_id in self.child_to_mother
        
    def get_children(self, mother_id: str) -> List[str]:
        """Get the child vector IDs for a mother vector."""
        return self.mother_to_children.get(mother_id, [])
        
    def get_mother(self, child_id: str) -> Optional[str]:
        """Get the mother vector ID for a child vector."""
        return self.child_to_mother.get(child_id)
        
    def get_family(self, vector_id: str) -> Set[str]:
        """Get all vectors in the same family (mother and all children)."""
        family = {vector_id}
        
        # If this is a mother vector, add all children
        if self.is_mother_vector(vector_id):
            family.update(self.get_children(vector_id))
            
        # If this is a child vector, add mother and all siblings
        elif self.is_child_vector(vector_id):
            mother_id = self.get_mother(vector_id)
            family.add(mother_id)
            family.update(self.get_children(mother_id))
            
        return family
        
    def propagate_influence(self, 
                           source_id: str, 
                           target_id: str, 
                           influence: float,
                           influence_map: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Propagate influence between vectors, considering mother-child relationships.
        """
        # Initialize if needed
        if source_id not in influence_map:
            influence_map[source_id] = {}
            
        # Direct influence
        influence_map[source_id][target_id] = influence
        
        # If source is a mother vector, propagate to children
        if self.is_mother_vector(source_id):
            children = self.get_children(source_id)
            for child_id in children:
                if child_id not in influence_map:
                    influence_map[child_id] = {}
                influence_map[child_id][target_id] = influence * 0.8
                
        # If source is a child vector, propagate to mother and siblings
        elif self.is_child_vector(source_id):
            mother_id = self.get_mother(source_id)
            siblings = self.get_children(mother_id)
            
            # Propagate to mother
            if mother_id not in influence_map:
                influence_map[mother_id] = {}
            influence_map[mother_id][target_id] = influence * 0.9
            
            # Propagate to siblings
            for sibling_id in siblings:
                if sibling_id != source_id:
                    if sibling_id not in influence_map:
                        influence_map[sibling_id] = {}
                    influence_map[sibling_id][target_id] = influence * 0.5
                    
        # If target is a mother vector, propagate to its children
        if self.is_mother_vector(target_id):
            children = self.get_children(target_id)
            for child_id in children:
                influence_map[source_id][child_id] = influence * 0.8
                
        # If target is a child vector, propagate to its mother and siblings
        elif self.is_child_vector(target_id):
            mother_id = self.get_mother(target_id)
            siblings = self.get_children(mother_id)
            
            # Propagate to mother
            influence_map[source_id][mother_id] = influence * 0.9
            
            # Propagate to siblings
            for sibling_id in siblings:
                if sibling_id != target_id:
                    influence_map[source_id][sibling_id] = influence * 0.5
                    
        return influence_map
        
    def move_as_unit(self, 
                    vector_id: str, 
                    movement_vector: np.ndarray,
                    positions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Move a vector and its related vectors (mother/children) as a single unit.
        """
        # Get all vectors in the family
        family = self.get_family(vector_id)
        
        # Apply movement to all family members
        for member_id in family:
            if member_id in positions:
                # Mother moves fully, children move with diminishing factor
                factor = 1.0
                if self.is_child_vector(member_id):
                    factor = 0.8
                    
                # Apply movement
                positions[member_id] = positions[member_id] + movement_vector * factor
                
        return positions 

    def verify_semantic_coherence(self, mother_id: str) -> bool:
        """
        Verify that child vectors maintain semantic coherence with their mother.
        """
        if not self.is_mother_vector(mother_id):
            return True
        
        mother_vec = self.get_vector(mother_id)
        children = self.get_children(mother_id)
        
        # Check metadata consistency
        mother_metadata = mother_vec['metadata']
        for child_id in children:
            child_vec = self.get_vector(child_id)
            child_metadata = child_vec['metadata']
            
            # Verify critical information is preserved
            if (mother_metadata.get('entity_type') != child_metadata.get('entity_type') or
                mother_metadata.get('class') != child_metadata.get('class') or
                mother_metadata.get('identity') != child_metadata.get('identity')):
                return False
            
            # Verify vector similarity
            mother_data = mother_vec['vector']
            child_data = child_vec['vector']
            similarity = np.dot(mother_data, child_data) / (np.linalg.norm(mother_data) * np.linalg.norm(child_data))
            if similarity < self.similarity_threshold:
                return False
            
        return True



