import sys
import os

# Voeg het pad naar de modules toe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Embedding')))

import unittest
import numpy as np
import transaction_embedding    

class TestTransactionEmbedding(unittest.TestCase):  
    def setUp(self):
        self.embedding = transaction_embedding.TransactionEmbedding(vector_size=10)
        self.sample_transaction = {
            "timestamp": "2023-06-15 14:30:00",
            "product": "Laptop",
            "price": 1200.50,
            "quantity": 1,
        }

    def test_encode_transaction(self):
        encoded = self.embedding.encode_transaction(self.sample_transaction)
        self.assertIsNotNone(encoded)
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(len(encoded), self.embedding.vector_size)

    def test_extract_features(self):
        features = self.embedding.extract_features(self.sample_transaction)
        self.assertIsNotNone(features)
        self.assertIsInstance(features, dict)
        self.assertIn('timestamp', features)
        self.assertIn('year', features)
        self.assertIn('month', features)
        self.assertIn('day', features)
        self.assertIn('hour', features)
        self.assertIn('minute', features)
        self.assertIn('second', features)
        
    def test_generate_transaction_vector(self):
        features = self.embedding.extract_features(self.sample_transaction)
        vector = self.embedding.generate_transaction_vector(features)
        self.assertIsNotNone(vector)
        self.assertIsInstance(vector, np.ndarray)

    def test_combine_to_transaction_vector(self):
        transaction_vector = np.array([1, 2, 3])
        product_vector = np.array([4, 5, 6])
        time_stamp_vector = np.array([7, 8, 9])
        combined = self.embedding.combine_to_transaction_vector(
            transaction_vector, product_vector, time_stamp_vector
        )
        self.assertIsNotNone(combined)
        self.assertIsInstance(combined, np.ndarray)
        self.assertEqual(len(combined), 9)  # 3 + 3 + 3

    def test_empty_transaction_data(self):
        with self.assertRaises(ValueError):
            self.embedding.encode_transaction({})

    def test_missing_timestamp(self):
        invalid_transaction = self.sample_transaction.copy()
        del invalid_transaction['timestamp']
        with self.assertRaises(ValueError):
            self.embedding.encode_transaction(invalid_transaction)

    def test_vector_search(self):
        # This is a placeholder test and should be updated when vector_search is implemented
        result = self.embedding.vector_search("test_id")
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()