""" Test Embedding Verification Module

This module provides utilities for verifying that the embedding system correctly
vectorizes and embeds different types of data while maintaining semantic meaning.

Author: Carlos D. Almeida
"""

import unittest
import numpy as np
from typing import Dict, Any, List
import logging
from datetime import datetime

from Embedding.test_data_generator import TestDataGenerator
from Embedding.unified_embedding import UnifiedEmbedding
from Embedding.text_embedding import TextEmbedding
from Embedding.transaction_embedding import TransactionEmbedding
from Embedding.time_stamp_embedding import TimeStampEmbedding
from Embedding.data_object_embedding import DataObjectEmbedding
from Embedding.multimodal_embedding import MultimodalEmbedding
from Embedding.vector_relationship import VectorRelationship

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestEmbeddingVerification(unittest.TestCase):
    """Test cases for verifying embedding functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = TestDataGenerator()
        self.unified_embedder = UnifiedEmbedding()
        self.text_embedder = TextEmbedding()
        self.transaction_embedder = TransactionEmbedding()
        self.timestamp_embedder = TimeStampEmbedding()
        self.data_object_embedder = DataObjectEmbedding()
        self.multimodal_embedder = MultimodalEmbedding()
        
    def test_text_embedding(self):
        """Test text embedding and semantic similarity."""
        # Generate similar texts
        text1 = self.generator.generate_text(num_sentences=2)
        text2 = text1.replace("purchased", "bought")  # Similar meaning
        
        # Generate different text
        text3 = self.generator.generate_text(num_sentences=2)
        
        # Embed texts
        vec1 = self.text_embedder.embed(text1).flatten()  # Flatten to 1D array
        vec2 = self.text_embedder.embed(text2).flatten()
        vec3 = self.text_embedder.embed(text3).flatten()
        
        # Calculate similarities
        sim_similar = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        sim_different = np.dot(vec1, vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
        
        # Verify that similar texts have higher similarity
        self.assertGreater(sim_similar, sim_different)
        logger.info(f"Text similarity test passed. Similar: {sim_similar:.4f}, Different: {sim_different:.4f}")
        
    def test_transaction_embedding(self):
        """Test transaction embedding and amount similarity."""
        # Generate similar transactions
        trans1 = self.generator.generate_transaction()
        trans2 = trans1.copy()
        trans2['amount'] = trans1['amount'] * 1.1  # Similar amount
        
        # Generate different transaction
        trans3 = self.generator.generate_transaction()
        trans3['amount'] = trans1['amount'] * 5  # Different amount
        
        # Embed transactions
        vec1 = self.transaction_embedder.embed(trans1)
        vec2 = self.transaction_embedder.embed(trans2)
        vec3 = self.transaction_embedder.embed(trans3)
        
        # Calculate similarities
        sim_similar = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        sim_different = np.dot(vec1, vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
        
        # Verify that similar transactions have higher similarity
        self.assertGreater(sim_similar, sim_different)
        logger.info(f"Transaction similarity test passed. Similar: {sim_similar:.4f}, Different: {sim_different:.4f}")
        
    def test_timestamp_embedding(self):
        """Test timestamp embedding and temporal proximity."""
        # Generate close timestamps
        ts1 = self.generator.generate_timestamp(days_range=1)
        ts2 = self.generator.generate_timestamp(days_range=1)
        
        # Generate distant timestamp
        ts3 = self.generator.generate_timestamp(days_range=30)
        
        # Embed timestamps
        vec1 = self.timestamp_embedder.embed(ts1).flatten()
        vec2 = self.timestamp_embedder.embed(ts2).flatten()
        vec3 = self.timestamp_embedder.embed(ts3).flatten()
        
        # Calculate similarities
        sim_close = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        sim_distant = np.dot(vec1, vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
        
        # Verify that close timestamps have higher similarity
        self.assertGreater(sim_close, sim_distant)
        logger.info(f"Timestamp similarity test passed. Close: {sim_close:.4f}, Distant: {sim_distant:.4f}")
        
    def test_multimodal_embedding(self):
        """Test multimodal embedding and cross-modal similarity."""
        # Generate multimodal data
        data1 = self.generator.generate_multimodal_data(include_image=True)
        data2 = self.generator.generate_multimodal_data(include_image=True)
        
        # Embed multimodal data
        vec1 = self.multimodal_embedder.embed(data1)
        vec2 = self.multimodal_embedder.embed(data2)
        
        # Verify vector dimensions
        self.assertEqual(len(vec1), len(vec2))
        
        # Calculate similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        logger.info(f"Multimodal embedding test passed. Vector dimensions: {len(vec1)}, Similarity: {similarity:.4f}")
        
    def test_data_object_embedding(self):
        """Test complete data object embedding."""
        # Generate data object
        data = self.generator.generate_data_object()
        
        # Embed data object
        vec = self.data_object_embedder.embed_data_object(data)
        
        # Verify vector dimensions and normalization
        self.assertIsInstance(vec, np.ndarray)
        self.assertGreater(len(vec), 0)
        self.assertAlmostEqual(np.linalg.norm(vec), 1.0, places=6)
        logger.info(f"Data object embedding test passed. Vector dimensions: {len(vec)}")
        
    def test_unified_embedding(self):
        """Test unified embedding system."""
        # Generate various data types
        text = self.generator.generate_text()
        transaction = self.generator.generate_transaction()
        timestamp = self.generator.generate_timestamp()
        multimodal = self.generator.generate_multimodal_data()
        
        # Embed using unified embedder
        text_vec = self.unified_embedder.embed_data(text, "text")
        trans_vec = self.unified_embedder.embed_data(transaction, "transaction")
        ts_vec = self.unified_embedder.embed_data(timestamp, "timestamp")
        multi_vec = self.unified_embedder.embed_data(multimodal, "multimodal")
        
        # Verify all vectors are created
        self.assertIsNotNone(text_vec)
        self.assertIsNotNone(trans_vec)
        self.assertIsNotNone(ts_vec)
        self.assertIsNotNone(multi_vec)
        
        logger.info("Unified embedding test passed")
        
    def test_vector_normalization(self):
        """Test vector normalization."""
        # Generate and embed data
        data = self.generator.generate_data_object()
        vec = self.data_object_embedder.embed_data_object(data)
        
        # Check normalization
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=6)
        logger.info(f"Vector normalization test passed. Norm: {norm:.6f}")
        
    def test_vector_combination(self):
        """Test vector combination with weights."""
        # Generate two vectors
        vec1 = self.text_embedder.embed(self.generator.generate_text())
        vec2 = self.text_embedder.embed(self.generator.generate_text())
        
        # Ensure vectors are 1D
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        # Store vectors in unified embedder
        vec1_id = self.unified_embedder._generate_vector_id()
        vec2_id = self.unified_embedder._generate_vector_id()
        self.unified_embedder.vectors[vec1_id] = vec1
        self.unified_embedder.vectors[vec2_id] = vec2
        
        # Combine vectors with different weights
        combined = self.unified_embedder.combine_vectors(
            [vec1_id, vec2_id],
            weights=[0.7, 0.3]
        )
        
        # Ensure combined vector is 1D
        combined = combined.flatten()
        
        # Verify combined vector
        self.assertEqual(len(combined), len(vec1))
        self.assertAlmostEqual(np.linalg.norm(combined), 1.0, places=6)
        
        # Calculate similarity with original vectors
        sim1 = np.dot(combined, vec1) / (np.linalg.norm(combined) * np.linalg.norm(vec1))
        sim2 = np.dot(combined, vec2) / (np.linalg.norm(combined) * np.linalg.norm(vec2))
        logger.info(f"Vector combination test passed. Similarities: {sim1:.4f}, {sim2:.4f}")

    def test_vector_splitting(self):
        """Test vector splitting behavior based on dimensionality."""
        # Initialize vector relationship manager
        vector_rel = VectorRelationship(dimension_threshold=100)
        
        # Test 1: Vector below threshold (should not split)
        small_vector = np.random.rand(50)  # 50 features
        mother_vec, child_vecs, child_ids = vector_rel.split_vector("small_vec", small_vector)
        self.assertEqual(len(child_vecs), 0)
        self.assertEqual(len(child_ids), 0)
        self.assertEqual(len(mother_vec), 50)
        logger.info("Small vector test passed - no splitting occurred")
        
        # Test 2: Vector above threshold (should split)
        large_vector = np.random.rand(150)  # 150 features
        mother_vec, child_vecs, child_ids = vector_rel.split_vector("large_vec", large_vector)
        self.assertGreater(len(child_vecs), 0)
        self.assertGreater(len(child_ids), 0)
        self.assertLess(len(mother_vec), 150)  # Should be reduced
        logger.info(f"Large vector test passed - split into {len(child_vecs)} children")
        
        # Test 3: Vector with insufficient features (should not split)
        tiny_vector = np.random.rand(1)  # 1 feature
        mother_vec, child_vecs, child_ids = vector_rel.split_vector("tiny_vec", tiny_vector)
        self.assertEqual(len(child_vecs), 0)
        self.assertEqual(len(child_ids), 0)
        self.assertEqual(len(mother_vec), 1)
        logger.info("Tiny vector test passed - no splitting occurred")
        
        # Test 4: Vector relationships
        self.assertTrue(vector_rel.is_mother_vector("large_vec"))
        self.assertTrue(vector_rel.is_child_vector("large_vec_child_1"))
        self.assertEqual(vector_rel.get_mother("large_vec_child_1"), "large_vec")
        self.assertIn("large_vec_child_1", vector_rel.get_children("large_vec"))
        logger.info("Vector relationships test passed")
        
        # Test 5: Family relationships
        family = vector_rel.get_family("large_vec")
        self.assertIn("large_vec", family)
        self.assertTrue(all(child_id in family for child_id in child_ids))
        logger.info("Family relationships test passed")

def run_verification_tests():
    """Run all verification tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmbeddingVerification)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == "__main__":
    run_verification_tests() 