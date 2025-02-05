from sentence_transformers import SentenceTransformer

"""from sentence_transformers import SentenceTransformer

class TextEmbedding:
    """
    Algemene tekst-embedding class voor verschillende soorten data.
    
    Attributes:
        model_name (str): Naam van het model dat wordt gebruikt voor tekst-embedding.
        model (SentenceTransformer): Het model dat wordt gebruikt om tekst om te zetten in vectoren.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialiseer de TextEmbedding class met een specifiek model.

        Args:
            model_name (str): Naam van het model dat wordt gebruikt voor tekst-embedding.
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        """
        Zet een tekst om in een vector.

        Args:
            text (str): De tekst die moet worden omgezet in een vector.

        Returns:
            numpy.ndarray: De resulterende vectorrepresentatie van de tekst.
        """
        return self.model.encode(text)

class CustomerEmbedding(TextEmbedding):
    """
    Tekst-embedding class specifiek voor klantgegevens.
    """
    def encode_customer(self, customer_data):
        """
        Zet klantgegevens om in een vector.

        Args:
            customer_data (dict): Een dictionary met klantgegevens zoals naam, leeftijd en locatie.

        Returns:
            numpy.ndarray: De resulterende vectorrepresentatie van de klantgegevens.
        """
        text_representation = f"Naam: {customer_data['name']}, Leeftijd: {customer_data['age']}, Locatie: {customer_data['location']}"
        return self.encode(text_representation)

class NotesEmbedding(TextEmbedding):
    """
    Tekst-embedding class specifiek voor klantnotities.
    """
    def encode_note(self, note):
        """
        Zet een klantnotitie om in een vector.

        Args:
            note (str): De notitie die moet worden omgezet in een vector.

        Returns:
            numpy.ndarray: De resulterende vectorrepresentatie van de notitie.
        """
        return self.encode(note)

class TransactionEmbedding(TextEmbedding):
    """
    Tekst-embedding class specifiek voor transacties.
    """
    def encode_transaction(self, transaction_data):
        """
        Zet een transactie om in een vector.

        Args:
            transaction_data (dict): Een dictionary met transactiegegevens.

        Returns:
            numpy.ndarray: De resulterende vectorrepresentatie van de transactie.
        """
        text_representation = f"Bedrag: {transaction_data['amount']}, Datum: {transaction_data['date']}, Beschrijving: {transaction_data['description']}"
        return self.encode(text_representation)"""
class TextEmbedding:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Algemene tekst-embedding class voor verschillende soorten data.
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        """
        Zet een tekst om in een vector.
        """
        return self.model.encode(text)

class CustomerEmbedding(TextEmbedding):
    def encode_customer(self, customer_data):
        """
        Zet klantgegevens om in een vector.
        """
        text_representation = f"Naam: {customer_data['name']}, Leeftijd: {customer_data['age']}, Locatie: {customer_data['location']}"
        return self.encode(text_representation)

class NotesEmbedding(TextEmbedding):
    def encode_note(self, note):
        """
        Zet een klantnotitie om in een vector.
        """
        return self.encode(note)

class TransactionEmbedding(TextEmbedding):
    def encode_transaction(self, transaction_data):
        """
        Zet een transactie om in een vector.
        """
        text_representation = f"Product: {transaction_data['product']}, Bedrag: {transaction_data['amount']}, Datum: {transaction_data['date']}"
        return self.encode(text_representation)

# 🔥 Test de embeddings
if __name__ == "__main__":
    # Klantgegevens
    customer = {"name": "Jan de Vries", "age": 35, "location": "Amsterdam"}
    customer_embedder = CustomerEmbedding()
    customer_vector = customer_embedder.encode_customer(customer)
    
    # Notitie
    note = "Deze klant heeft recent een supportvraag ingediend over zijn bestelling."
    note_embedder = NotesEmbedding()
    note_vector = note_embedder.encode_note(note)
    
    # Transactie
    transaction = {"product": "Laptop", "amount": "1200 EUR", "date": "2024-02-05"}
    transaction_embedder = TransactionEmbedding()
    transaction_vector = transaction_embedder.encode_transaction(transaction)

    print("🔹 Klantvector:", customer_vector[:5])
    print("🔹 Notitievector:", note_vector[:5])
    print("🔹 Transactievector:", transaction_vector[:5])
