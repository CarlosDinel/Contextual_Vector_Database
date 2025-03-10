import numpy as np
import multiprocessing

class InfluenceCalculator:
    def __init__(self, base_impact_radius=1.0, attention_factor_multiplier=1.0, lambda_decay=0.95, 
                 tau_threshold=0.01, solidity_factor=1.0, initial_energy=1.0):
        """
        Optimized Influence Calculator with dynamic impact radius, adaptive solidness, 
        weighted impact force, and parallel processing.

        Args:
            base_impact_radius (float): The base impact radius for influence calculation.
            attention_factor_multiplier (float): Multiplier for the attention factor.
            lambda_decay (float): Energy decay factor.
            tau_threshold (float): Adaptive convergence threshold.
            solidity_factor (float): Factor for vector solidness.
            initial_energy (float): Initial energy for energy decay.
        """
        self.base_impact_radius = base_impact_radius
        self.attention_factor_multiplier = attention_factor_multiplier
        self.lambda_decay = lambda_decay  # Energy Decay factor
        self.tau_threshold = tau_threshold  # Adaptive Convergence threshold
        self.epsilon = 1e-9  # Small value to avoid division by zero
        self.solidity_factor = solidity_factor  # The VectorSolidness!
        self.energy = initial_energy  # Initial energy for energy decay
        self.vector_solidness = {}  # Dictionary to store solidness per vector
        self.vector_impact_radius = {}  # Dictionary for dynamic impact radius

    def initialize_vectors(self, vectors):
        """
        Initializes solidness and impact radius for each vector.

        Args:
            vectors (List[np.ndarray]): List of vectors to initialize.
        """
        for idx, vector in enumerate(vectors):
            self.vector_solidness[idx] = self.solidity_factor
            self.vector_impact_radius[idx] = self.base_impact_radius

    def calculate_impact(self, vector_i, vector_j, index_i, index_j):
        """
        Calculates the context-aware impact of vector_j on vector_i.

        Args:
            vector_i (np.ndarray): The first vector.
            vector_j (np.ndarray): The second vector.
            index_i (int): Index of the first vector.
            index_j (int): Index of the second vector.

        Returns:
            float: The calculated impact value.
        """
        similarity = self.calculate_similarity(vector_i, vector_j)
        distance = self.calculate_distance(vector_i, vector_j)
        impact_radius = self.vector_impact_radius.get(index_i, self.base_impact_radius)
        solidness_i = self.vector_solidness.get(index_i, self.solidity_factor)
        solidness_j = self.vector_solidness.get(index_j, self.solidity_factor)
        
        weight_factor = (solidness_j) / (solidness_i + self.epsilon)  # Weighted influence
        exponent = np.exp(-(distance**2) / (2 * impact_radius**2))
        
        impact = weight_factor * (similarity / (distance + self.epsilon)) * exponent * self.energy
        return impact

    def calculate_distance(self, vector_i, vector_j):
        """
        Calculates the Euclidean distance between two vectors.

        Args:
            vector_i (np.ndarray): The first vector.
            vector_j (np.ndarray): The second vector.

        Returns:
            float: The Euclidean distance between the two vectors.
        """
        return np.linalg.norm(vector_i - vector_j)

    def calculate_similarity(self, vector_i, vector_j):
        """
        Optimized cosine similarity calculation.

        Args:
            vector_i (np.ndarray): The first vector.
            vector_j (np.ndarray): The second vector.

        Returns:
            float: The cosine similarity between the two vectors.
        """
        return np.dot(vector_i, vector_j) / (np.linalg.norm(vector_i) * np.linalg.norm(vector_j) + self.epsilon)

    def update_impact_radius(self, vector_idx, impact_sum):
        """
        Updates the impact radius dynamically per vector based on total received impact.

        Args:
            vector_idx (int): Index of the vector to update.
            impact_sum (float): Total received impact.
        """
        self.vector_impact_radius[vector_idx] = self.base_impact_radius + 0.5 * impact_sum

    def update_solidness(self, vector_idx, impact_sum):
        """
        Adaptively updates the solidness per vector based on total received impact.

        Args:
            vector_idx (int): Index of the vector to update.
            impact_sum (float): Total received impact.
        """
        self.vector_solidness[vector_idx] = self.lambda_decay * self.vector_solidness[vector_idx] + (1 - self.lambda_decay) * impact_sum

    def re_embed_vectors_parallel(self, vectors):
        """
        Re-embeds vectors in parallel using multiprocessing.

        Args:
            vectors (List[np.ndarray]): List of vectors to re-embed.

        Returns:
            np.ndarray: Array of updated vectors.
        """
        num_processes = min(multiprocessing.cpu_count(), len(vectors))
        pool = multiprocessing.Pool(processes=num_processes)
        
        # Prepare input data for parallel processing
        inputs = [(vectors[i], vectors, i) for i in range(len(vectors))]

        # Execute in parallel
        updated_vectors = pool.starmap(self.re_embed_vector, inputs)

        # Close pool
        pool.close()
        pool.join()

        return np.array(updated_vectors)

    def re_embed_vector(self, vector_i, all_vectors, index_i):
        """
        Re-embeds a single vector based on the influence of all other vectors.

        Args:
            vector_i (np.ndarray): The vector to re-embed.
            all_vectors (List[np.ndarray]): List of all vectors.
            index_i (int): Index of the vector to re-embed.

        Returns:
            np.ndarray: The updated vector.
        """
        total_impact = np.zeros_like(vector_i)
        impact_sum = 0

        for j, vector_j in enumerate(all_vectors):
            if index_i == j:
                continue  # Skip self-influence

            impact = self.calculate_impact(vector_i, vector_j, index_i, j)
            impact_direction = (vector_j - vector_i) / (np.linalg.norm(vector_j - vector_i) + self.epsilon)
            total_impact += impact * impact_direction
            impact_sum += impact

        # Update solidness and impact radius dynamically
        self.update_impact_radius(index_i, impact_sum)
        self.update_solidness(index_i, impact_sum)

        # Apply solidity factor and return updated vector
        return vector_i + (self.vector_solidness[index_i] * total_impact)


