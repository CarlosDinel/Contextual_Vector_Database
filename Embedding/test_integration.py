""" Integration Test Module for Embedding and Contexter

This module provides tests for the interaction between the embedding system
and the contexter system, demonstrating the complete lifecycle of data objects
including embedding, context determination, transactions, and vector splitting.

Author: Carlos D. Almeida
"""

import unittest
import numpy as np
from typing import Dict, Any, List, Union, Optional
import logging
from datetime import datetime
import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import required modules
from Embedding.unified_embedding import UnifiedEmbedding
from Embedding.text_embedding import TextEmbedding
from Embedding.time_stamp_embedding import TimeStampEmbedding
from Embedding.vector_relationship import VectorRelationship
from Contexter.contexter_model import Contexter
from Contexter.context_aggregator import ContextAggregator
from Contexter.influence_calculator import InfluenceCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestEmbeddingContexterIntegration(unittest.TestCase):
    """Test cases for embedding and contexter integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.unified_embedder = UnifiedEmbedding()
        self.text_embedder = TextEmbedding()
        self.timestamp_embedder = TimeStampEmbedding()
        self.vector_relationship = VectorRelationship()
        self.contexter = Contexter()
        self.context_aggregator = ContextAggregator()
        self.influence_calculator = InfluenceCalculator()
        
    def test_data_object_lifecycle(self):
        """Test complete lifecycle of a data object including embedding, context, and transactions."""
        # 1. Create initial data object
        data = {
            'text': "Sample text for testing",
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'source': 'test',
                'category': 'example',
                'priority': 'high'
            }
        }
        
        # 2. Create initial embedding
        unified_vec = self.unified_embedder.embed_data(data, "text")
        
        # 3. Check if vector needs splitting
        if self.vector_relationship.should_split_vector(unified_vec, data['metadata']):
            mother_vec, child_vecs, child_ids = self.vector_relationship.split_vector(
                "test_obj", unified_vec, data['metadata']
            )
            vectors = [mother_vec] + child_vecs
            vector_ids = ["test_obj"] + child_ids
        else:
            vectors = [unified_vec]
            vector_ids = ["test_obj"]
        
        # 4. Create Vector objects with initial context
        from Contexter.base_model import Vector
        vector_objects = [
            Vector(id=vid, data=vdata, metadata=data['metadata'])
            for vid, vdata in zip(vector_ids, vectors)
        ]
        
        # 5. Determine initial context
        context_map = self.contexter.determine_context(vector_objects)
        
        # 6. Simulate transaction (new data affecting context)
        transaction_data = {
            'text': "Updated text with new context",
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'source': 'transaction',
                'category': 'update',
                'priority': 'high'
            }
        }
        
        # 7. Update context with transaction
        transaction_vector = self.unified_embedder.embed_data(transaction_data, "text")
        transaction_vec = Vector(
            id="transaction",
            data=transaction_vector,
            metadata=transaction_data['metadata']
        )
        
        updated_context = self.context_aggregator.prepare_context(
            vector_objects,
            [transaction_vec]
        )
        
        # 8. Reembed vectors with updated context
        reembedded_vectors = self.contexter.reembed_vectors(vector_objects)
        
        # 9. Verify vector properties and relationships
        for i, (orig_vec, reemb_vec) in enumerate(zip(vectors, reembedded_vectors)):
            # Check dimensions preserved
            self.assertEqual(len(reemb_vec.data), len(orig_vec))
            
            # Check normalization
            self.assertAlmostEqual(np.linalg.norm(reemb_vec.data), 1.0, places=6)
            
            # Check similarity maintained
            similarity = np.dot(orig_vec, reemb_vec.data)
            self.assertGreater(similarity, 0.7)
            
            # Check context relationships
            if i > 0:  # For child vectors
                mother_id = self.vector_relationship.get_mother(reemb_vec.id)
                self.assertIsNotNone(mother_id)
                self.assertEqual(mother_id, vector_ids[0])
        
        logger.info(f"Data object lifecycle test passed. Vector count: {len(reembedded_vectors)}")
        
    def test_context_preservation_after_split(self):
        """Test that context relationships are preserved after vector splitting."""
        # Create large data object that will need splitting
        # Use a realistic long text instead of repeated phrases
        large_text = """
        This is a comprehensive test document that will be used to test vector splitting functionality.
        The document contains multiple paragraphs with meaningful content to ensure proper embedding.
        We want to test how the system handles large vectors and maintains context relationships.
        The text includes various topics and concepts to create a rich semantic representation.
        This will help verify that the splitting mechanism works correctly while preserving semantic meaning.
        The document should be long enough to trigger the splitting threshold but still maintain coherence.
        We'll use this to test both the embedding quality and the context preservation after splitting.
        The content should be diverse enough to create meaningful vector representations.
        This will help ensure that the mother-child relationships maintain semantic relevance.
        The test will verify that context is properly preserved across the split vectors.
        """
        
        data = {
            'text': large_text,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'source': 'test',
                'category': 'example',
                'priority': 'high'
            }
        }
        
        # Create initial embedding
        unified_vec = self.unified_embedder.embed_data(data, "text")
        
        # Create initial Vector object
        from Contexter.base_model import Vector
        initial_vector = Vector(id="large_obj", data=unified_vec, metadata=data['metadata'])
        
        # Determine initial context
        initial_context = self.contexter.determine_context([initial_vector])
        
        # Split vector
        mother_vec, child_vecs, child_ids = self.vector_relationship.split_vector(
            "large_obj", unified_vec, data['metadata']
        )
        
        # Create Vector objects for split vectors
        split_vectors = [
            Vector(id="mother", data=mother_vec, metadata=data['metadata'])
        ] + [
            Vector(id=child_id, data=child_vec, metadata=data['metadata'])
            for child_id, child_vec in zip(child_ids, child_vecs)
        ]
        
        # Determine context for split vectors
        split_context = self.contexter.determine_context(split_vectors)
        
        # Verify context relationships
        for child_id in child_ids:
            mother_id = self.vector_relationship.get_mother(child_id)
            self.assertEqual(mother_id, "mother")
            
            # Check context relationships in split_context
            self.assertIn(mother_id, split_context)
            self.assertIn(child_id, split_context)
            
            # Verify mother-child relationship in context
            mother_context = split_context[mother_id]
            child_context = split_context[child_id]
            
            # Mother should influence child and vice versa
            mother_to_child = next((impact for vid, impact in mother_context if vid == child_id), 0)
            child_to_mother = next((impact for vid, impact in child_context if vid == mother_id), 0)
            
            self.assertGreater(mother_to_child, 0)
            self.assertGreater(child_to_mother, 0)
            
            # Verify semantic similarity between mother and child
            similarity = np.dot(mother_vec, child_vecs[child_ids.index(child_id)])
            self.assertGreater(similarity, 0.5)  # Should maintain some semantic similarity
        
        logger.info(f"Context preservation test passed. Split vectors: {len(split_vectors)}")
        
    def test_transaction_based_context_update(self):
        """Test how transactions affect context and vector relationships."""
        # Create initial data objects
        texts = [
            "First sample text",
            "Second sample text",
            "Third sample text"
        ]
        
        # Create embeddings and Vector objects
        from Contexter.base_model import Vector
        vector_objects = []
        for i, text in enumerate(texts):
            vec = self.text_embedder.embed(text)
            metadata = {
                'source': f'test{i}',
                'timestamp': datetime.now().isoformat(),
                'category': 'initial'
            }
            vector_objects.append(Vector(id=f"vec_{i}", data=vec, metadata=metadata))
        
        # Determine initial context
        initial_context = self.contexter.determine_context(vector_objects)
        
        # Simulate transaction affecting all vectors
        transaction = {
            'text': "Transaction affecting all vectors",
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'source': 'transaction',
                'category': 'update',
                'priority': 'high',
                'affects_all': True
            }
        }
        
        # Create transaction vector
        transaction_vec = self.text_embedder.embed(transaction['text'])
        transaction_vector = Vector(id="transaction", data=transaction_vec, metadata=transaction['metadata'])
        
        # Update context with transaction
        updated_context = self.context_aggregator.prepare_context(vector_objects, [transaction_vector])
        
        # Reembed vectors with updated context
        reembedded_vectors = self.contexter.reembed_vectors(vector_objects)
        
        # Verify transaction impact
        for i, (orig_vec, reemb_vec) in enumerate(zip(vector_objects, reembedded_vectors)):
            # Check that vectors have moved
            movement = np.linalg.norm(reemb_vec.data - orig_vec.data)
            self.assertGreater(movement, 0)
            
            # Check context relationships
            vector_context = updated_context.get(reemb_vec.id, [])
            transaction_impact = next((impact for vid, impact in vector_context if vid == "transaction"), 0)
            self.assertGreater(transaction_impact, 0)
        
        logger.info(f"Transaction-based context update test passed. Affected vectors: {len(reembedded_vectors)}")

    def test_embedding_module_core_functions(self):
        """Test core functionality of the embedding module."""
        # 1. Test text embedding quality
        text1 = "Machine learning is transforming industries through automation."
        text2 = "Deep learning models are revolutionizing pattern recognition."
        text3 = "The weather forecast predicts sunny conditions."
        
        vec1 = self.text_embedder.embed(text1)
        vec2 = self.text_embedder.embed(text2)
        vec3 = self.text_embedder.embed(text3)
        
        # Similar texts should have higher similarity
        sim_similar = np.dot(vec1, vec2)
        sim_different = np.dot(vec1, vec3)
        self.assertGreater(sim_similar, sim_different)
        
        # 2. Test timestamp embedding
        time1 = datetime.now().isoformat()
        time2 = (datetime.now().replace(hour=1)).isoformat()
        
        vec_time1 = self.timestamp_embedder.embed(time1)
        vec_time2 = self.timestamp_embedder.embed(time2)
        
        # Closer times should have higher similarity
        time_sim = np.dot(vec_time1, vec_time2)
        self.assertGreater(time_sim, 0.5)
        
        # 3. Test vector splitting
        large_text = "This is a comprehensive test document that will be used to test vector splitting functionality. " * 10
        large_vec = self.text_embedder.embed(large_text)
        
        if self.vector_relationship.should_split_vector(large_vec):
            mother_vec, child_vecs, child_ids = self.vector_relationship.split_vector(
                "test_split", large_vec, {'text': large_text}
            )
            self.assertGreater(len(child_vecs), 0)
            
            # Verify mother-child relationship
            for child_id in child_ids:
                self.assertEqual(self.vector_relationship.get_mother(child_id), "test_split")
        
        # 4. Test multimodal embedding
        data = {
            'text': "Sample multimodal data",
            'timestamp': datetime.now().isoformat(),
            'metadata': {'source': 'test'}
        }
        
        unified_vec = self.unified_embedder.embed_data(data, "text")
        self.assertIsInstance(unified_vec, np.ndarray)
        self.assertGreater(len(unified_vec), 0)
        
        logger.info("Embedding module core functions test passed")
        
    def test_contexter_module_core_functions(self):
        """Test core functionality of the contexter module using real embedded data."""
        # 1. Create real data objects with different types of content
        data_objects = [
            {
                'text': "Machine learning is transforming industries through automation and prediction.",
                'timestamp': datetime.now().isoformat(),
                'metadata': {'source': 'tech', 'category': 'AI', 'priority': 'high'}
            },
            {
                'text': "Deep learning models are revolutionizing pattern recognition and decision making.",
                'timestamp': datetime.now().isoformat(),
                'metadata': {'source': 'tech', 'category': 'AI', 'priority': 'high'}
            },
            {
                'text': "The weather forecast predicts sunny conditions for the weekend.",
                'timestamp': datetime.now().isoformat(),
                'metadata': {'source': 'weather', 'category': 'forecast', 'priority': 'medium'}
            }
        ]
        
        # 2. Create embeddings for each data object
        from Contexter.base_model import Vector
        vector_objects = []
        for i, data in enumerate(data_objects):
            # Create unified embedding combining text and timestamp
            vec = self.unified_embedder.embed_data(data, "text")
            vector_objects.append(Vector(
                id=f"vec_{i}",
                data=vec,
                metadata=data['metadata']
            ))
        
        # 3. Test context determination with real data
        context_map = self.contexter.determine_context(vector_objects)
        
        # Verify context map
        self.assertEqual(len(context_map), len(vector_objects))
        for vec_id in context_map:
            self.assertGreater(len(context_map[vec_id]), 0)
            
            # Check that similar content has stronger context relationships
            if vec_id == "vec_0":  # First AI-related text
                ai_context = context_map[vec_id]
                weather_context = context_map["vec_2"]  # Weather text
                
                # AI texts should have stronger relationship than AI-weather
                ai_impact = next((impact for vid, impact in ai_context if vid == "vec_1"), 0)
                weather_impact = next((impact for vid, impact in ai_context if vid == "vec_2"), 0)
                self.assertGreater(ai_impact, weather_impact)
        
        # 4. Test influence calculation with real data
        influences = self.influence_calculator.calculate_influences(vector_objects)
        
        # Verify influence calculations
        self.assertEqual(len(influences), len(vector_objects))
        for vec_id, vec_influences in influences.items():
            self.assertEqual(len(vec_influences), len(vector_objects) - 1)
            for other_id, influence in vec_influences.items():
                self.assertGreaterEqual(influence, 0)
                self.assertLessEqual(influence, 1)
                
                # Check that similar content has higher influence
                if vec_id == "vec_0" and other_id == "vec_1":  # AI texts
                    self.assertGreater(influence, 0.5)  # High influence between similar content
        
        # 5. Test context aggregation with real data
        aggregated_context = self.context_aggregator.prepare_context(vector_objects, vector_objects)
        
        # Verify aggregation
        self.assertEqual(len(aggregated_context), len(vector_objects))
        for vec_id, context in aggregated_context.items():
            self.assertGreater(len(context), 0)
            for other_id, impact in context:
                self.assertGreaterEqual(impact, 0)
                self.assertLessEqual(impact, 1)
                
                # Check that aggregated context preserves semantic relationships
                if vec_id == "vec_0" and other_id == "vec_1":  # AI texts
                    self.assertGreater(impact, 0.5)  # High impact between similar content
        
        # 6. Test reembedding process with real data
        reembedded_vectors = self.contexter.reembed_vectors(vector_objects)
        
        # Verify reembedding
        self.assertEqual(len(reembedded_vectors), len(vector_objects))
        for i, (orig_vec, reemb_vec) in enumerate(zip(vector_objects, reembedded_vectors)):
            # Check dimensions preserved
            self.assertEqual(len(reemb_vec.data), len(orig_vec.data))
            
            # Check normalization
            self.assertAlmostEqual(np.linalg.norm(reemb_vec.data), 1.0, places=6)
            
            # Check some movement occurred
            movement = np.linalg.norm(reemb_vec.data - orig_vec.data)
            self.assertGreater(movement, 0)
            
            # Check similarity maintained
            similarity = np.dot(orig_vec.data, reemb_vec.data)
            self.assertGreater(similarity, 0.7)
            
            # Check that semantic relationships are preserved
            if i == 0:  # First AI text
                # Should maintain high similarity with other AI text
                ai_similarity = np.dot(reemb_vec.data, reembedded_vectors[1].data)
                weather_similarity = np.dot(reemb_vec.data, reembedded_vectors[2].data)
                self.assertGreater(ai_similarity, weather_similarity)
        
        logger.info("Contexter module core functions test passed with real embedded data")

def run_integration_tests():
    """Run all integration tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmbeddingContexterIntegration)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == "__main__":
    run_integration_tests() 