""" Test Transformer Integration Module

This module provides tests for the transformer-inspired architecture,
including multi-headed attention in the Contexter and feed-forward
network capabilities in DaPGRaC.

Author: Carlos D. Almeida
"""

import unittest
import numpy as np
from typing import Dict, Any, List
import logging
from datetime import datetime

from Embedding.unified_embedding import UnifiedEmbedding
from Contexter.contexter_model import Contexter
from Contexter.influence_calculator import InfluenceCalculator
from Contexter.context_aggregator import ContextAggregator
from dapgrac_core.dapgrac import DaPGRaC
from Contexter.base_model import Vector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestTransformerIntegration(unittest.TestCase):
    """Test cases for transformer-inspired architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.unified_embedder = UnifiedEmbedding()
        self.contexter = Contexter(num_heads=8)
        self.dapgrac = DaPGRaC(config={
            'hidden_layers': [64, 32],
            'sev_threshold': 0.1,
            'relationship_threshold': 0.01
        })
        
    def test_multi_headed_attention(self):
        """Test multi-headed attention in the Contexter."""
        # Create test vectors
        vectors = []
        for i in range(16):  # Multiple vectors to test attention heads
            data = {
                'text': f"Test text {i} for multi-headed attention testing",
                'timestamp': datetime.now().isoformat(),
                'metadata': {'source': 'test', 'category': f'category_{i}'}
            }
            vec = self.unified_embedder.embed_data(data, "text")
            vectors.append(Vector(
                id=f"vec_{i}",
                data=vec,
                metadata=data['metadata']
            ))
        
        # Determine context with multi-headed attention
        context_map = self.contexter.determine_context(vectors)
        
        # Verify attention heads processing
        for vec in vectors:
            # Check that each vector has context relationships
            self.assertIn(vec.id, context_map)
            impacts = context_map[vec.id]
            
            # Verify attention distribution
            total_impact = sum(impact for _, impact in impacts)
            self.assertGreater(total_impact, 0)
            
            # Verify attention head processing
            head_size = len(vectors) // self.contexter.num_heads
            self.assertEqual(len(impacts), head_size)
            
        logger.info("Multi-headed attention test passed")
        
    def test_feed_forward_processing(self):
        """Test feed-forward network processing in DaPGRaC."""
        # Create test vectors
        vectors = []
        for i in range(8):
            data = {
                'text': f"Test text {i} for feed-forward processing",
                'timestamp': datetime.now().isoformat(),
                'metadata': {'source': 'test', 'category': f'category_{i}'}
            }
            vec = self.unified_embedder.embed_data(data, "text")
            vectors.append({
                'id': f"vec_{i}",
                'data': vec.tolist(),
                'metadata': data['metadata']
            })
        
        # Process through feed-forward network
        processed_vectors = self.dapgrac.process_vectors(vectors)
        
        # Verify feed-forward transformations
        for orig_vec, proc_vec in zip(vectors, processed_vectors):
            # Check vector dimensions through hidden layers
            self.assertEqual(len(proc_vec['data']), self.dapgrac.hidden_layers[-1])
            
            # Verify ReLU activation
            self.assertTrue(all(x >= 0 for x in proc_vec['data']))
            
            # Check relationship formation
            self.assertIn('relationships', proc_vec)
            
        logger.info("Feed-forward processing test passed")
        
    def test_transformer_loop(self):
        """Test the complete transformer-inspired loop between Contexter and DaPGRaC."""
        # Create initial vectors
        vectors = []
        for i in range(8):
            data = {
                'text': f"Test text {i} for transformer loop testing",
                'timestamp': datetime.now().isoformat(),
                'metadata': {'source': 'test', 'category': f'category_{i}'}
            }
            vec = self.unified_embedder.embed_data(data, "text")
            vectors.append(Vector(
                id=f"vec_{i}",
                data=vec,
                metadata=data['metadata']
            ))
        
        # First pass: Contexter attention
        context_map = self.contexter.determine_context(vectors)
        
        # Convert to DaPGRaC format
        dapgrac_vectors = [{
            'id': vec.id,
            'data': vec.data.tolist(),
            'metadata': vec.metadata
        } for vec in vectors]
        
        # Second pass: DaPGRaC feed-forward
        processed_vectors = self.dapgrac.process_vectors(dapgrac_vectors, context_map)
        
        # Verify transformer loop
        for i, (orig_vec, proc_vec) in enumerate(zip(vectors, processed_vectors)):
            # Check context preservation
            self.assertIn(orig_vec.id, context_map)
            
            # Check feed-forward transformation
            self.assertEqual(len(proc_vec['data']), self.dapgrac.hidden_layers[-1])
            
            # Check relationship formation
            self.assertIn('relationships', proc_vec)
            relationships = proc_vec['relationships']
            self.assertGreater(len(relationships), 0)
            
            # Verify relationship strength
            for target_id, relationship in relationships.items():
                self.assertGreater(relationship['strength'], 0)
                self.assertLessEqual(relationship['strength'], 1.0)
                
        logger.info("Transformer loop test passed")
        
    def test_attention_head_distribution(self):
        """Test the distribution of attention across heads."""
        # Create test vectors
        vectors = []
        for i in range(32):  # Large number to test head distribution
            data = {
                'text': f"Test text {i} for attention head distribution",
                'timestamp': datetime.now().isoformat(),
                'metadata': {'source': 'test', 'category': f'category_{i}'}
            }
            vec = self.unified_embedder.embed_data(data, "text")
            vectors.append(Vector(
                id=f"vec_{i}",
                data=vec,
                metadata=data['metadata']
            ))
        
        # Process with multi-headed attention
        context_map = self.contexter.determine_context(vectors)
        
        # Calculate attention distribution
        head_impacts = {}
        for vec_id, impacts in context_map.items():
            for target_id, impact in impacts:
                head = int(target_id.split('_')[1]) % self.contexter.num_heads
                if head not in head_impacts:
                    head_impacts[head] = []
                head_impacts[head].append(impact)
        
        # Verify balanced distribution
        for head in range(self.contexter.num_heads):
            self.assertIn(head, head_impacts)
            avg_impact = np.mean(head_impacts[head])
            self.assertGreater(avg_impact, 0)
            
        logger.info("Attention head distribution test passed")
        
    def test_feed_forward_layer_progression(self):
        """Test the progression of vectors through feed-forward layers."""
        # Create test vector
        data = {
            'text': "Test text for feed-forward layer progression",
            'timestamp': datetime.now().isoformat(),
            'metadata': {'source': 'test', 'category': 'test'}
        }
        vec = self.unified_embedder.embed_data(data, "text")
        vector = {
            'id': 'test_vec',
            'data': vec.tolist(),
            'metadata': data['metadata']
        }
        
        # Process through each layer
        layer_outputs = []
        current_data = np.array(vector['data'])
        
        for layer_size in self.dapgrac.hidden_layers:
            # Apply feed-forward transformation
            transformed = self.dapgrac._apply_feed_forward(current_data, layer_size)
            layer_outputs.append(transformed)
            current_data = transformed
        
        # Verify layer progression
        for i, (prev_layer, next_layer) in enumerate(zip(layer_outputs[:-1], layer_outputs[1:])):
            # Check dimension reduction
            self.assertLessEqual(len(next_layer), len(prev_layer))
            
            # Check non-linearity
            self.assertTrue(np.any(next_layer != prev_layer[:len(next_layer)]))
            
        logger.info("Feed-forward layer progression test passed")

if __name__ == '__main__':
    unittest.main() 