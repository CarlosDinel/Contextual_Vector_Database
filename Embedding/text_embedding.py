from sentence_transformers import SentenceTransformer

class TextEmbedding:
    """
    General text embedding class for various types of data.
    
    Attributes:
        model_name (str): Name of the model used for text embedding.
        model (SentenceTransformer): The model used to convert text into vectors.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the TextEmbedding class with a specific model.

        Args:
            model_name (str): Name of the model used for text embedding.
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        """
        Convert a text into a vector.

        Args:
            text (str): The text to be converted into a vector.

        Returns:
            numpy.ndarray: The resulting vector representation of the text.
        """
        return self.model.encode(text)

class CustomerEmbedding(TextEmbedding):
    """
    Text embedding class specifically for customer data.
    """
    def encode_customer(self, customer_data):
        """
        Convert customer data into a vector.

        Args:
            customer_data (dict): A dictionary with customer data such as name, age, and location.

        Returns:
            numpy.ndarray: The resulting vector representation of the customer data.
        """
        text_representation = f"Name: {customer_data['name']}, Age: {customer_data['age']}, Location: {customer_data['location']}"
        return self.encode(text_representation)

class NotesEmbedding(TextEmbedding):
    """
    Text embedding class specifically for customer notes.
    """
    def encode_note(self, note):
        """
        Convert a customer note into a vector.

        Args:
            note (str): The note to be converted into a vector.

        Returns:
            numpy.ndarray: The resulting vector representation of the note.
        """
        return self.encode(note)

class TransactionEmbedding(TextEmbedding):
    """
    Text embedding class specifically for transactions.
    """
    def encode_transaction(self, transaction_data):
        """
        Convert a transaction into a vector.

        Args:
            transaction_data (dict): A dictionary with transaction data.

        Returns:
            numpy.ndarray: The resulting vector representation of the transaction.
        """
        text_representation = f"Product: {transaction_data['product']}, Amount: {transaction_data['amount']}, Date: {transaction_data['date']}"
        return self.encode(text_representation)



# 🔥 Test the embeddings
if __name__ == "__main__":
    # Customer data
    customer = {"name": "Jan de Vries", "age": 35, "location": "Amsterdam"}
    customer_embedder = CustomerEmbedding()
    customer_vector = customer_embedder.encode_customer(customer)
    
    # Note
    note = "This customer recently submitted a support query about their order."
    note_embedder = NotesEmbedding()
    note_vector = note_embedder.encode_note(note)
    
    # Transaction
    transaction = {"product": "Laptop", "amount": 1200, "date": "2023-10-01"}
    transaction_embedder = TransactionEmbedding()
    transaction_vector = transaction_embedder.encode_transaction(transaction)

    print("🔹 Klantvector:", customer_vector)
    print("🔹 Notitievector:", note_vector)
    print("🔹 Transactievector:", transaction_vector)

