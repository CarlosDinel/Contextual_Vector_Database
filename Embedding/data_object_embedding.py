import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize
import text_embedding   

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

class DataObject:
    def __init__(self, entity_id, entity_type, attributes):
        """
        Creates a vector-based representation of a data object (e.g., customer, product, transaction).
        
        Args:
            entity_id (str): Unique identifier for the data object.
            entity_type (str): Type of the entity (e.g., 'customer', 'product', 'transaction').
            attributes (dict): Key-value pairs representing the entity's features.
        """
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.attributes = attributes
        self.vector = None  # Initialize without generating immediately
        self.locked = False
        self.update_embedding()
    
    def normalize_vector(self, vector):
        """Normalizes a vector to ensure numerical stability."""
        return normalize(vector.reshape(1, -1))[0]
    
    def update_embedding(self):
        """Updates the vector representation only when necessary."""
        if not self.locked:
            text_representation = f"{self.entity_type}: " + ", ".join(f"{k}: {v}" for k, v in self.attributes.items())
            self.vector = model.encode(text_representation, convert_to_numpy=True)
    
    def combine(self, customer_vector, note_vector=None, transaction_vector=None, weights=None, time_stamp_vector=None):
        """
        Combines vectors into a single representation.

        Args:
            customer_vector (numpy.ndarray): The main entity vector (e.g., customer).
            note_vector (numpy.ndarray, optional): The vector representation of a note.
            transaction_vector (numpy.ndarray, optional): The vector representation of a transaction.
            time_stamp_vector (numpy.ndarray, optional): The vector representation of a time stamp. 
            weights (list, optional): A list of weights to apply to each vector (default: equal weights).

        Returns:
            numpy.ndarray: The combined vector representation.
        """
        if self.locked:
            raise Exception("Vector is locked. Unlock before modifying.")

        vectors = [self.normalize_vector(customer_vector)]
        if note_vector is not None:
            vectors.append(self.normalize_vector(text_embedding.NotesEmbedding().encode(note_vector)))
        if transaction_vector is not None:
            vectors.append(self.normalize_vector(text_embedding.TransactionEmbedding().encode(transaction_vector)))
        if time_stamp_vector is not None:   
            vectors.append(self.normalize_vector(text_embedding.TimeStampEmbedding().encode(time_stamp_vector)))    
        if weights and len(weights) == len(vectors):
            combined_vector = np.average(vectors, axis=0, weights=weights)
        else:
            combined_vector = np.mean(vectors, axis=0)  # Default: simple mean

        return combined_vector, print(combined_vector)
    
    def lock(self):
        """Locks the vector, preventing modifications."""
        self.locked = True
    
    def unlock(self):
        """Unlocks the vector, allowing modifications."""
        # if modified, update the vector to locked 
        if self.locked:
            self.update_embedding()
        self.locked = False
    
    def update_attributes(self, new_attributes):
        """Updates the object's attributes and regenerates its vector when unlocked."""
        if not self.locked:
            self.attributes.update(new_attributes)
            self.update_embedding()

# Create some example data objects
customer = DataObject("C123", "customer", {"name": "Alice", "location": "NY", "preferences": "organic food"})
product = DataObject("P456", "product", {"name": "Organic Apple", "category": "Fruits", "price": "2.5"})
transaction = DataObject("T789", "transaction", {"customer": "C123", "product": "P456", "amount": "5"})

# Create FAISS index for fast similarity search
d = len(customer.vector)  # Vector dimension
index = faiss.IndexFlatIP(d)  # Cosine similarity index

# Add vectors to FAISS index
entity_vectors = np.array([customer.vector, product.vector, transaction.vector])
index.add(entity_vectors)

# Function to dynamically query vectors
def dynamic_query(query_text, k=2):
    """Encodes the query text into a vector and searches for the most similar entities."""
    query_vector = model.encode(query_text, convert_to_numpy=True)
    distances, indices = index.search(np.array([query_vector]), k=k)
    
    print("Most similar entities:")
    for idx, distance in zip(indices[0], distances[0]):
        print(f"Entity ID: {idx}, Distance: {distance}")

# Example dynamic query
dynamic_query("customer who likes organic fruits")

#  Test the embeddings
if __name__ == "__main__":
    # Customer data
    customer_data = {"name": "Jan de Vries", "age": 35, "location": "Amsterdam"}
    customer_embedder = text_embedding.CustomerEmbedding()
    customer_vector = customer_embedder.encode_customer(customer_data)
    
    # Note
    note = "This customer recently submitted a support query about their order."
    note_embedder = text_embedding.NotesEmbedding()
    note_vector = note_embedder.encode_note(note)
    
    # Transaction
    transaction_data = {"product": "Laptop", "amount": 1200, "date": "2023-10-01"}
    transaction_embedder = text_embedding.TransactionEmbedding()
    transaction_vector = transaction_embedder.encode_transaction(transaction_data)

    # Combine embeddings into a single vector
    data_object = DataObject(entity_id="1", entity_type="customer", attributes=customer_data)
    combined_vector = data_object.combine(customer_vector, note_vector, transaction_vector)

    # Print vectors for testing
    print("🔹 Customer vector:", customer_vector)
    print("🔹 Note vector:", note_vector)
    print("🔹 Transaction vector:", transaction_vector)
    print("🔹 Combined vector:", combined_vector)

    # Check the length of the combined vector
    print("Length of combined vector:", len(combined_vector))

