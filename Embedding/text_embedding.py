from sentence_transformers import SentenceTransformer
import numpy as np

customer_data = {
    "name": "Pieter de Vries",
    "age": 35,
    "location": "Amsterdam",
    "aankoop": "Laptop",
    "prijs": 999.99,
    "datum": "2023-06-15"
}

class TextEmbedding:    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text):
        if not text: 
            raise ValueError("Text is empty")
        try:
            return self.model.encode(text)
        except Exception as e:
            print(f"Error: {e}")
            return None

    def encode_customer(self, customer_data):   
        text_representation = f"Name: {customer_data['name']}, Age: {customer_data['age']}, Location: {customer_data['location']}, Aankoop: {customer_data['aankoop']}, Prijs: {customer_data['prijs']}, Datum: {customer_data['datum']}"
        customer_vector = self.encode_text(text_representation)
        return customer_vector  
    
    def decode_vector(self, vector):
        # This method would require a reverse mapping from vectors to text, which is not straightforward.
        # For simplicity, we'll return a placeholder text.
        return "Decoded text from vector"

    def batch_encode(self, texts):
        if not texts:
            raise ValueError("Input texts cannot be empty")
        try:
            return self.model.encode(texts)
        except Exception as e:  
            print(f"Error: {e}")
            return None

    def generate_vector(self, features):
        """
        Generate a vector from the extracted features.

        Args:
            features (dict): A dictionary of extracted features.

        Returns:
            numpy.ndarray: The generated vector representation of the features.
        """
        text_representation = ", ".join([f"{key}: {value}" for key, value in features.items()])
        return self.encode_text(text_representation)

    def update_model(self, model_name):
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Model updated to {model_name}")
        except Exception as e:
            print(f"Error: {e}")
            return None

    def save_model(self, path):
        try:
            self.model.save(path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error: {e}")
            return None

    def load_model(self, path): 
        try:
            self.model = SentenceTransformer(path)
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error: {e}")
            return None