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
        
        # tokenize text 
        # generate embedding
        # return vector

    def encode_customer(self, customer_data):   
        text_representation = f"Name: {customer_data['name']}, Age: {customer_data['age']}, Location: {customer_data['location']}, Aankoop: {customer_data['aankoop']}, Prijs: {customer_data['prijs']}, Datum: {customer_data['datum']}"
        customer_vector = self.encode_text(text_representation)
        return customer_vector  
    
    
    
    def decode_vector(self, vector):
        # convert vector to tokens
        # convert tokens to text
        # return text
        return None
    
    def batch_encode(self, text):
        if not text:
            raise ValueError("Input texts cannot be empty")
        try:
            return self.model.encode(text)
        except Exception as e:  
            print(f"Error: {e}")
            return None
    
        # encode multiple texts


    
    def update_model(self, model_name):
        # update model
        return None
    
    def save_model(self, path):
        # save model
        return None 
    
    def load_model(self, path): 
        # load model
        return None 
    



