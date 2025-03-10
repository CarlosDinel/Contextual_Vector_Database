import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize
import text_embedding  
import transaction_embedding     
import time_stamp_embedding
from scipy.stats import entropy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataObjectEmbedding:
    def __init__(self, base_vector=None, max_vector_size=1024):
        self.base_vector = base_vector if base_vector is not None else np.array([])
        self.transaction_vectors = []
        self.child_vectors = []
        self.max_vector_size = max_vector_size
        self.vector = self.base_vector.copy() if base_vector is not None else np.array([])
        self.transaction_embedder = transaction_embedding.TransactionEmbedding()
        self.vector_solidness = VectorSolidness(self.vector)

    def update_vector(self, transaction_data):
        transaction_vector = self.transaction_embedder.encode_transaction(transaction_data)
        
        if self.base_vector.size == 0:
            self.base_vector = transaction_vector
        else:
            self.transaction_vectors.append(transaction_vector)
        
        combined_vector = np.concatenate([self.base_vector] + self.transaction_vectors)
        
        if len(combined_vector) > self.max_vector_size:
            vector_splitter = VectorSplitting(combined_vector, self.max_vector_size)
            self.vector, new_child_vectors = vector_splitter.split_vector()
            self.child_vectors.extend(new_child_vectors)
        else:
            self.vector = combined_vector
        
        self.recalculate_solidness()
        self.log_change(transaction_data)

    def recalculate_solidness(self):
        self.vector_solidness = VectorSolidness(self.get_full_vector())

    def get_full_vector(self):
        return np.concatenate([self.vector] + self.child_vectors)

    def move_in_space(self, displacement):
        # Move the main vector and all child vectors
        self.vector += displacement
        for i in range(len(self.child_vectors)):
            self.child_vectors[i] += displacement

    def combine_vectors(self, customer_vector, time_stamp_vector=None): 
        """Combines vectors into a single representation."""
        data_object_vector =  np.concatenate([customer_vector, time_stamp_vector])  
        return data_object_vector 


    def pad_vector(vector, target_dim):
        """
        Pads a vector with zeros if it is shorter than the target dimension.
        """
        target_dim = int(1024) 
        if len(vector) < target_dim:
            return np.pad(vector, (0, target_dim - len(vector)), 'constant')
        return vector[:target_dim]  # Truncate if too long


    def normalize_vector_dimension(self, vector, target_dimension):
        """Normalizes the vector to a target dimension."""
        if len(vector) < target_dimension:
            normalized_vector = np.pad(vector, (0, target_dimension - len(vector)))
        else:
            normalized_vector = vector[:target_dimension]
        return normalized_vector

    def calculate_solidness(self): 
        """Calculates the solidness of the data object.
        """
        self.solidness = self.vector_solidness.calculate_solidness()   
        return self.solidness   

    
    def get_embedding(self): 
        """Get the embedding of the data object."""
        return None 
    
    def lock_vector(self):
        """Locks the vector, preventing modifications."""
        return None
    
    def unlock_vector(self):
        """Unlocks the vector, allowing modifications."""
        return None
    
    def log_change(self, transaction_data):
        """Logs changes to the data object."""
        logging.info(f"Data object updated with transaction: {transaction_data}")


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


class VectorSplitting:
    def __init__(self, vector, max_vector_size):
        self.vector = vector
        self.max_vector_size = max_vector_size

    def split_vector(self):
        main_vector = self.vector[:self.max_vector_size]
        child_vectors = [self.vector[self.max_vector_size:]]
        return main_vector, child_vectors
    
    def connect_vectors(self, vectors):
        return np.concatenate(vectors)
    

# class VectorSplitting: 
#     """Splits a vector into a main vector and child vectors."""
#     def __init__(self, combined_vector, max_vector_size=1000): 
#         self.combined_vector = combined_vector
#         self.max_vector_size = max_vector_size
    
#     def split_vector(self): 
#         """Split the combined vector into the main vector and child vectors."""
#         # Check vector size
#         if len(self.combined_vector) > self.max_vector_size: 
#             split_point = self.max_vector_size
#             main_vector = self.combined_vector[:split_point]
#             child_vector = self.combined_vector[split_point:]
#             return main_vector, [child_vector] # return as a list
#         else: 
#             return self.combined_vector, []



base_vector = np.array([
    -2.99544614e-02,  7.74931312e-02,  1.15252109e-02, -8.04022700e-02,
     2.70858612e-02,  2.38767788e-02,  1.37460791e-02, -6.10772846e-03,
    -3.65181118e-02,  3.55421118e-02,  5.15289716e-02, -1.80053055e-01,
    -9.27598327e-02,  3.39416647e-03,  1.06720207e-02, -2.30363738e-02,
     2.11187024e-02,  3.66944596e-02,  3.52025665e-02, -2.24371217e-02,
    -1.34397577e-02,  3.87426317e-02,  2.48337425e-02, -1.34683505e-01,
    -4.16973270e-02,  4.41534594e-02,  4.21745591e-02,  1.95484832e-02,
    -8.33901018e-03,  1.91946141e-02,  5.48228472e-02,  6.89471290e-02,
    -8.62928703e-02,  6.44042343e-02,  8.72470737e-02, -2.42929272e-02,
    -7.39091113e-02,  3.49527188e-02, -2.35123187e-02, -1.11247497e-02,
    -2.30590180e-02, -3.09941471e-02, -6.25899956e-02,  3.76952067e-03,
    -2.20282818e-03, -8.78056884e-03, -5.43859825e-02,  2.70316377e-02,
     3.36525440e-02, -5.60998498e-03, -8.76548663e-02, -3.92742604e-02,
     2.51188669e-02, -7.32544390e-03, -1.56181911e-02,  4.80663516e-02,
    -6.86088651e-02,  6.91495761e-02,  3.42985988e-02, -6.32330775e-02,
     8.03808030e-03,  2.12153085e-02, -1.01686895e-01,  2.59377342e-02,
     1.30237360e-02, -6.05778843e-02,  1.28404181e-02,  2.37946343e-02,
     2.29459573e-02,  1.37729291e-02,  2.55284030e-02, -1.37199229e-02,
    -2.51829643e-02, -1.59564205e-02,  1.43096521e-02,  1.38535341e-02,
    -6.62351623e-02, -5.57794832e-02,  4.29149158e-02,  2.14778036e-02,
    -1.34961996e-02, -2.88620312e-02,  1.54655809e-02,  1.49877435e-02,
     8.78041610e-03,  5.51732369e-02,  5.23806810e-02, -1.40396599e-02,
    -2.00463794e-02, -8.74858946e-02, -4.21682708e-02,  1.48306917e-02,
    -3.66332717e-02,  3.86506766e-02, -4.07265946e-02, -3.50502916e-02,
     1.06102124e-01,  8.57781172e-02, -4.15127836e-02,  5.34709655e-02,
    -1.12084597e-02,  5.06951734e-02,  7.81494081e-02,  8.73523727e-02,
    -5.53312227e-02,  5.36965355e-02, -2.70587429e-02,  1.97471287e-02,
     1.25821205e-02, -3.12791355e-02, -1.14125699e-01, -6.14934089e-03,
    -5.74477613e-02, -8.61332789e-02,  7.09217489e-02, -4.55739051e-02,
    -2.46301237e-02,  5.53811900e-02,  4.60277237e-02, -3.07723135e-02,
    -4.86125890e-03,  3.78956832e-03, -3.81493233e-02,  1.85884181e-02,
    -2.03428827e-02, -6.10506274e-02,  1.11294955e-01,  4.35798046e-34,
    -2.40014796e-03,  5.23870299e-03,  9.00179446e-02,  6.22753091e-02,
     1.43727018e-02,  3.14126164e-02, -4.44670394e-02,  5.85998083e-03,
    -7.07861409e-02,  2.90316716e-02, -1.81023758e-02, -8.04155394e-02,
     4.42748843e-03, -1.79431569e-02, -1.93363708e-02,  5.63893132e-02,
     6.59231544e-02, -2.31821812e-03, -3.95035669e-02,  4.74437177e-02,
     2.52384320e-02, -9.93028656e-02,  1.84642058e-02,  2.83764508e-02,
     2.37143058e-02,  4.92306566e-03,  1.00534642e-02, -6.98011070e-02,
     8.68199393e-02,  4.19130772e-02, -4.80721369e-02,  3.55700292e-02,
    -3.86165045e-02, -8.48107636e-02, -5.27026542e-02,  7.64052495e-02,
    -3.12383249e-02, -9.36776996e-02, -7.26250485e-02, -2.36529522e-02,
     5.89548200e-02, -8.21750902e-04, -5.00608534e-02,  1.38436467e-03,
    -6.05288558e-02,  4.27642837e-02,  6.11506365e-02, -3.12101413e-02,
     8.25946331e-02,  5.44780344e-02, -1.23765528e-01, -1.68450177e-02,
    -1.96637243e-01, -4.57731150e-02, -5.23275509e-02, -1.17090819e-02,
     2.14244407e-02,  5.96343838e-02,  1.17322141e-02, -1.59929805e-02,
     6.35620207e-02, -1.14881834e-02, -4.51977998e-02,  2.30578985e-02,
     3.87271680e-02, -1.30231649e-01,  2.44902838e-02, -1.22500628e-01,
     5.27332760e-02, -1.98952742e-02, -2.85284631e-02,  1.44301979e-02,
     8.93852785e-02, -2.34713759e-02,  3.66504900e-02,  6.77278191e-02,
    -8.31799861e-03,  2.38911603e-02, -3.30536589e-02,  1.80449355e-02,
    -5.13748266e-02,  2.38009244e-02, -1.26640305e-01, -4.72619608e-02,
     8.35246444e-02, -2.05461811e-02, -1.93895828e-02, -6.48498088e-02,
    -3.14721353e-02,  4.29482087e-02, -5.80327697e-02, -2.09215228e-02,
    -2.62965094e-02, -3.28362845e-02, -8.48082304e-02, -3.69838220e-33,
    -3.49696027e-03, -3.10857426e-02, -5.89057151e-03,  1.18333856e-02,
     9.52635631e-02,  7.15077296e-03, -6.98146224e-03,  1.27244189e-01,
     4.78420258e-02,  2.05097813e-03, -1.81980859e-02, -2.46850066e-02,
     9.67854261e-02,  3.16540934e-02,  1.83829572e-02,  1.34355605e-01,
     2.98132701e-03,  1.06270509e-02, -7.21223594e-04, -5.49167302e-03,
     1.75611489e-02,  7.58416951e-02,  7.64044374e-03,  5.59341684e-02,
    -6.61418065e-02,  1.49915309e-03,  8.36115256e-02, -2.92976107e-02,
    -8.36163908e-02,  2.44934224e-02, -6.86838478e-02,  4.96081710e-02,
    -9.54838097e-02,  9.51022729e-02, -4.37667929e-02,  7.81584345e-03,
     6.67118281e-02,  2.59250347e-02, -1.28517731e-03,  2.99480800e-02,
    -5.80420904e-02,  4.88545708e-02, -4.06091698e-02, -4.21950966e-03,
    -1.14709381e-02, -1.09712720e-01,  4.28720638e-02,  1.39689045e-02,
     6.47711828e-02,  8.27713404e-03,  9.00108293e-02,  6.50964454e-02,
     7.92875141e-03,  1.06618106e-02,  1.78171992e-02,  7.70539697e-03,
    -3.06806192e-02, -5.08095585e-02,  7.07829669e-02, -3.29501781e-04,
    -1.44501589e-02,  8.37879181e-02, -4.65539172e-02,  6.64437413e-02,
     5.78141659e-02, -2.81706806e-02, -5.43906242e-02,  3.99910510e-02,
    -1.14188343e-02, -1.10144503e-01,  3.63838077e-02, -3.07089332e-02,
     3.83219682e-02,  3.11097503e-02, -1.20747527e-02, -8.70016068e-02,
     5.92210656e-03,  6.25724271e-02, -6.32506190e-03,  2.46236436e-02,
     6.71281107e-03,  4.28783242e-03, -7.69105274e-04, -7.15313060e-03,
     1.10594823e-03,  9.34809819e-03,  1.06544290e-02, -4.15913314e-02,
     5.91242127e-03, -4.22251113e-02, -2.56975628e-02,  1.83455274e-02,
    -7.64460340e-02,  4.46652956e-02, -2.33392492e-02, -3.25599743e-08,
    -1.11105721e-02, -2.77497750e-02, -3.41551378e-02,  1.89045593e-02,
     5.69203719e-02, -3.30185518e-02,  1.16865337e-02,  6.51133657e-02,
     2.82165930e-02,  2.57699490e-02,  1.05523402e-02,  3.03671788e-02,
     4.65902127e-02,  6.34798482e-02,  4.06110287e-02,  9.64824017e-03,
    -2.49274652e-02,  2.94708014e-02, -8.94969143e-03,  4.25682077e-03,
     5.87835088e-02,  9.97069292e-03,  6.26475066e-02, -2.41643973e-02,
     4.25947234e-02,  3.22378501e-02,  3.79232392e-02, -1.90860666e-02,
    -8.04211348e-02, -4.80241291e-02, -3.26845497e-02,  1.09183520e-01,
     5.21602333e-02, -1.82963293e-02,  8.92694481e-03, -9.37292874e-02,
    -5.56323025e-03,  1.72871333e-02, -9.08524022e-02, -2.62858681e-02,
    -2.78294422e-02, -4.81109284e-02,  3.15085799e-02,  2.98612863e-02,
     1.10600493e-03,  4.18769196e-02,  2.12112702e-02, -6.01766147e-02,
     5.90547640e-03, -6.59016743e-02, -1.02021217e-01,  1.47908751e-03,
     7.30319396e-02, -2.50986777e-03, -7.49463663e-02, -5.06162569e-02,
    -4.49414179e-02,  5.74013777e-02, -2.46349834e-02, -1.15269339e-02,
     9.04167816e-02, -1.63741447e-02,  -9.73106921e-02,  3.26282047e-02, 
     6.74333333e-01,  5.00000000e-01,  4.83870968e-01,  5.83333333e-01,
  5.00000000e-01,  0.00000000e+00,  1.22464680e-16, -1.00000000e+00,
  1.01168322e-01, -9.94869323e-01])
