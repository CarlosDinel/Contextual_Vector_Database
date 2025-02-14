import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize
import text_embedding   
import time_stamp_embedding
from scipy.stats import entropy

class DataObjectEmbedding: 

    def __init___(self, name="", age=0, location=""): 
        self.attributes = self.initialise_attributes()
        self.vector = None
        self.solidness = 0.0
        self.is_locked = False


    def initialise_attributes(self, name="", age=0, location=""):    
        """Initialise the attributes of the data object."""
        return {"name": "", "age": 0, "location": ""}
    
    def combine_vectors(self, customer_vector, time_stamp_vector=None): 
        """Combines vectors into a single representation."""
        data_object_vector =  np.concatenate([customer_vector, time_stamp_vector])  
        return data_object_vector     

    
    def calculate_solidness(self): 
        """Calculates the solidness of the data object.
        formula = S(V_i) = 1 / (1 + e^(-k(I_i - I_0)))
        where:The concept of Vector Solidness you've described is an innovative approach to 
        quantifying the stability of vectors in dynamic vector spaces. This method is particularly 
        relevant for Contextual Vector Databases (CVDs) and could be applied in various fields, 
        including cardiovascular disease (CVD) risk prediction and cumulative 
        impact assessments in marine environments."""
        if self.vector_solidness(self):
            self.solidness = self.vector_solidness.calculate_solidness()    

        return None
    
    def update_vector(self, new_data): 
        """Updates the vector representation only when necessary.
        this creates the dynamic natur of the data object vector in the vector space."""
        if not self.is_locked:
            # implement update logic here

            pass
        return None
    
    def get_embedding(self): 
        """Get the embedding of the data object."""
        return None 
    
    def lock_vector(self):
        """Locks the vector, preventing modifications."""
        return None
    
    def unlock_vector(self):
        """Unlocks the vector, allowing modifications."""
        return None


class VectorSolidness: 
    """Calculates the solidness of a vector in a dynamic vector space."""
    def __init__(self, combined_vector, k=1.0, I0=1.0, H0=1.0, S0=0.5, A0=1.0): 
        self.vector = combined_vector
        self.k = k
        self.I0 = I0
        self.H0 = H0
        self.S0 = S0
        self.A0 = A0
        self.age = 0
        self.update_count = 0   

    def calculate_solidness_magnitude(self):
        magnitude = np.linalg.norm(self.vector)
        return 1 / (1 + np.exp(-self.k * (magnitude - self.I0)))

    def calculate_solidness_entropy(self):
        # Normalize vector to calculate entropy
        normalized_vector = self.vector / np.linalg.norm(self.vector)
        H = entropy(np.abs(normalized_vector))
        return 1 / (1 + np.exp(-self.k * (H - self.H0)))

    def calculate_solidness_sparsity(self):
        sparsity = np.sum(self.vector == 0) / len(self.vector)
        return 1 / (1 + np.exp(-self.k * (sparsity - self.S0)))

    def calculate_solidness_age(self):
        return 1 / (1 + np.exp(-self.k * (self.age - self.A0)))
    
    def calculate_solidness(self):
        magnitude = self.calculate_solidness_magnitude()
        entropy = self.calculate_solidness_entropy()
        sparsity = self.calculate_solidness_sparsity()
        age = self.calculate_solidness_age()
        initial_solidness = (magnitude + entropy + sparsity + age) / 4
        return initial_solidness
    

# Create instances of the embedding classes
customer_embedder = text_embedding.TextEmbedding()
time_stamp_embedder = time_stamp_embedding.TimeStampEmbedding()

# Encode customer data and time stamp
customer_vector = customer_embedder.encode_customer(text_embedding.customer_data)
time_stamp_vector = time_stamp_embedder.encode(time_stamp_embedding.timestamp)

# Combine vectors
data_object_embedding = DataObjectEmbedding()
combined_vector = data_object_embedding.combine_vectors(customer_vector, time_stamp_vector)
print(combined_vector)

# Calculate solidness
vector_solidness = VectorSolidness(combined_vector) 
solidness = vector_solidness.calculate_solidness()  
print(f"solidness{solidness}")