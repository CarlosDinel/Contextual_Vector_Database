import sys
import os
import unittest
import numpy as np
from scipy.stats import entropy

# Voeg het pad naar de modules toe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/carlosalmeida/Desktop/Contectual_Vector_Database/Embedding/')))

from Embedding.text_embedding import TextEmbedding
from Embedding.time_stamp_embedding import TimeStampEmbedding
from Embedding.transaction_embedding import TransactionEmbedding
from Embedding.data_object_embedding import DataObjectEmbedding, VectorSplitting, VectorSolidness

class TestDataObjectEmbedding(unittest.TestCase):
    def setUp(self):
        # Create a small base vector for testing
        self.base_vector = np.array([0.1, 0.2, 0.3, 0.4])
        self.max_vector_size = 10
        self.data_object = DataObjectEmbedding(
            base_vector=self.base_vector,
            max_vector_size=self.max_vector_size
        )

    def test_initialization(self):
        """Test proper initialization of DataObjectEmbedding"""
        self.assertTrue(np.array_equal(self.data_object.base_vector, self.base_vector))
        self.assertEqual(self.data_object.max_vector_size, self.max_vector_size)
        self.assertEqual(len(self.data_object.transaction_vectors), 0)
        self.assertEqual(len(self.data_object.child_vectors), 0)

    def test_update_vector_with_empty_base(self):
        """Test updating vector when base_vector is empty"""
        data_object = DataObjectEmbedding(max_vector_size=10)
        mock_transaction = {'amount': 100, 'type': 'purchase', 'timestamp': '2023-06-15 14:30:00'}
        data_object.update_vector(mock_transaction)
        self.assertIsNotNone(data_object.vector)
        self.assertTrue(len(data_object.vector) > 0)

    def test_vector_splitting(self):
        """Test vector splitting when max size is exceeded"""
        # Create a vector that will exceed max_size
        long_vector = np.ones(15)
        splitter = VectorSplitting(long_vector, max_vector_size=10)
        main_vector, child_vectors = splitter.split_vector()
        
        self.assertEqual(len(main_vector), 10)
        self.assertEqual(len(child_vectors), 1)
        self.assertEqual(len(child_vectors[0]), 5)

    def test_solidness_calculation(self):
        """Test solidness calculation with known values"""
        vector = np.array([0.1, 0.2, 0.3, 0.4])
        solidness = VectorSolidness(vector)
        result = solidness.calculate_solidness()
        
        self.assertIsInstance(result, float)
        self.assertTrue(0 <= result <= 1)

    def test_move_in_space(self):
        """Test vector movement in space"""
        displacement = np.array([0.1, 0.1, 0.1, 0.1])
        original_vector = self.data_object.vector.copy()
        self.data_object.move_in_space(displacement)
        
        expected_vector = original_vector + displacement
        self.assertTrue(np.array_equal(self.data_object.vector, expected_vector))

    def test_combine_vectors(self):
        """Test vector combination"""
        customer_vector = np.array([0.1, 0.2])
        time_stamp_vector = np.array([0.3, 0.4])
        
        result = self.data_object.combine_vectors(customer_vector, time_stamp_vector)
        expected = np.array([0.1, 0.2, 0.3, 0.4])
        
        self.assertTrue(np.array_equal(result, expected))

    def test_vector_solidness_components(self):
        """Test individual components of vector solidness"""
        vector = np.array([0.1, 0.2, 0.3, 0.4])
        solidness = VectorSolidness(vector)
        
        # Test magnitude
        magnitude = solidness.calculate_solidness_magnitude()
        self.assertTrue(0 <= magnitude <= 1)
        
        # Test entropy
        entropy_val = solidness.calculate_solidness_entropy()
        self.assertTrue(0 <= entropy_val <= 1)
        
        # Test sparsity
        sparsity = solidness.calculate_solidness_sparsity()
        self.assertTrue(0 <= sparsity <= 1)
        
        # Test age
        age = solidness.calculate_solidness_age()
        self.assertTrue(0 <= age <= 1)

    def test_update_vector_with_transactions(self):
        """Test updating vector with multiple transactions"""
        mock_transaction1 = {'amount': 100, 'type': 'purchase', 'timestamp': '2023-06-15 14:30:00'}
        mock_transaction2 = {'amount': 200, 'type': 'refund', 'timestamp': '2023-06-16 10:30:00'}
        
        original_length = len(self.data_object.vector)
        self.data_object.update_vector(mock_transaction1)
        self.data_object.update_vector(mock_transaction2)
        
        self.assertTrue(len(self.data_object.transaction_vectors) > 0)
        self.assertNotEqual(len(self.data_object.vector), original_length)

    def test_get_full_vector(self):
        """Test getting full vector including child vectors"""
        # Add some child vectors
        self.data_object.child_vectors = [np.array([0.5, 0.6]), np.array([0.7, 0.8])]
        full_vector = self.data_object.get_full_vector()
        
        expected_length = len(self.data_object.vector) + \
                         sum(len(cv) for cv in self.data_object.child_vectors)
        self.assertEqual(len(full_vector), expected_length)

    def test_edge_cases(self):
        """Test edge cases"""
        # Test with zero vector
        zero_vector = np.zeros(4)
        data_object = DataObjectEmbedding(base_vector=zero_vector)
        solidness = data_object.calculate_solidness()
        self.assertIsInstance(solidness, float)
        
        # Test with very large values
        large_vector = np.array([1e6, 1e6, 1e6, 1e6])
        data_object = DataObjectEmbedding(base_vector=large_vector)
        solidness = data_object.calculate_solidness()
        self.assertIsInstance(solidness, float)
        
        # Test with very small values
        small_vector = np.array([1e-6, 1e-6, 1e-6, 1e-6])
        data_object = DataObjectEmbedding(base_vector=small_vector)
        solidness = data_object.calculate_solidness()
        self.assertIsInstance(solidness, float)

if __name__ == '__main__':
    unittest.main()