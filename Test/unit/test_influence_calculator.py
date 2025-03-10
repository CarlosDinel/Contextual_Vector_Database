import unittest
import numpy as np
import os
import sys
# Voeg het pad naar de modules toe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/carlosalmeida/Desktop/Contectual_Vector_Database/Contexter')))

from Contexter.influence_calculator import InfluenceCalculator

class TestInfluenceCalculator(unittest.TestCase):
    
    def setUp(self):
        """Setup method to create an instance of InfluenceCalculator before each test."""
        self.calculator = InfluenceCalculator()  # Use default parameters
        self.vector1 = np.array([1, 0, 0])
        self.vector2 = np.array([0, 1, 0])
        self.data_object1 = {"category": "A"}
        self.data_object2 = {"category": "B"}

    def test_calculate_distance(self):
        """Test the calculate_distance method."""
        distance = self.calculator.calculate_distance(self.vector1, self.vector2)
        self.assertAlmostEqual(distance, np.sqrt(2))

    def test_calculate_similarity(self):
        """Test the calculate_similarity method."""
        similarity = self.calculator.calculate_similarity(self.vector1, self.vector2)
        self.assertAlmostEqual(similarity, 0.0)  # Vectors are orthogonal
        
        # Test with identical vectors
        similarity_identical = self.calculator.calculate_similarity(self.vector1, self.vector1)
        self.assertAlmostEqual(similarity_identical, 1.0)

    def test_calculate_attention_factor(self):
        """Test the calculate_attention_factor method."""
        # This requires mocking or defining the get_hierarchical_weight and get_contextual_weight
        # For now, just test that it returns a value between 0 and 1 when those return 1.0
        attention_factor = self.calculator.calculate_attention_factor(self.data_object1, self.data_object2)
        self.assertTrue(0 <= attention_factor <= 1)
    
    def test_calculate_impact(self):
        """Test the calculate_impact method."""
        # Basic test: Check impact is a non-negative value
        impact = self.calculator.calculate_impact(self.vector1, self.vector2, self.data_object1, self.data_object2)
        self.assertGreaterEqual(impact, 0)

    def test_update_impact_radius(self):
        """Test the update_impact_radius method."""
        vectors = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]
        self.calculator.update_impact_radius(vectors)
        self.assertGreater(self.calculator.impact_radius, 0)  # Radius should be updated

    def test_apply_energy_decay(self):
        """Test the apply_energy_decay method."""
        initial_energy = self.calculator.energy
        self.calculator.apply_energy_decay()
        self.assertLess(self.calculator.energy, initial_energy)  # Energy should decrease

    def test_apply_adaptive_convergence(self):
        """Test the apply_adaptive_convergence method."""
        new_vector = self.vector1 + self.calculator.tau_threshold / 2  # Slightly changed vector
        converged_vector = self.calculator.apply_adaptive_convergence(self.vector1, new_vector)
        np.testing.assert_array_equal(converged_vector, self.vector1)  # Should return original
        
        #Test with vector change above threshold
        new_vector = self.vector1 + self.calculator.tau_threshold * 2
        converged_vector = self.calculator.apply_adaptive_convergence(self.vector1, new_vector)
        np.testing.assert_array_equal(converged_vector, new_vector) #Should return new vector

    def test_apply_negative_feedback(self):
        """Test the apply_negative_feedback method."""
        influence = 1.0
        recent_change = 0.5
        feedback = self.calculator.apply_negative_feedback(influence, recent_change)
        self.assertLess(feedback, 1.0)  # Feedback should reduce influence

    def test_determine_impact_direction(self):
        """Test the determine_impact_direction method."""
        direction = self.calculator.determine_impact_direction(self.vector1, self.vector2)
        expected_direction = self.vector2 - self.vector1
        np.testing.assert_array_equal(direction, expected_direction)

    def test_re_embed_vector(self):
        """Test the re_embed_vector method."""
        vectors = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]
        data_objects = [{}, {}, {}]  # Empty data objects for now
        new_vector = self.calculator.re_embed_vector(self.vector1, vectors, data_objects)
        self.assertIsInstance(new_vector, np.ndarray)
        self.assertEqual(new_vector.shape, self.vector1.shape)
