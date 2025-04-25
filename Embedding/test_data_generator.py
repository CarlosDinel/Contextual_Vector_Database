""" Test Data Generator Module for Contextual Vector Database

This module provides utilities for generating test data for the embedding system,
including text, transactions, timestamps, and multimodal data.

Author: Carlos D. Almeida
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random
import string
import logging
from PIL import Image, ImageDraw
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDataGenerator:
    """
    Generates test data for the embedding system.
    """
    def __init__(self):
        """Initialize the test data generator."""
        # Sample data for text generation
        self.products = [
            "Laptop", "Smartphone", "Tablet", "Headphones", "Smartwatch",
            "Camera", "Printer", "Monitor", "Keyboard", "Mouse"
        ]
        
        self.categories = [
            "Electronics", "Computers", "Accessories", "Gaming", "Office",
            "Home", "Sports", "Fashion", "Books", "Food"
        ]
        
        self.transaction_types = [
            "purchase", "refund", "transfer", "deposit", "withdrawal"
        ]
        
        # Customer names for text generation
        self.first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Emma", "James",
            "Olivia", "William", "Sophia"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
            "Miller", "Davis", "Rodriguez", "Martinez"
        ]
        
    def generate_text(self, 
                     num_sentences: int = 3,
                     include_customer: bool = False) -> str:
        """
        Generate sample text data.
        
        Args:
            num_sentences: Number of sentences to generate
            include_customer: Whether to include customer information
            
        Returns:
            str: Generated text
        """
        sentences = []
        
        if include_customer:
            customer = self._generate_customer_info()
            sentences.append(customer)
            
        for _ in range(num_sentences):
            product = random.choice(self.products)
            category = random.choice(self.categories)
            price = random.uniform(10.0, 1000.0)
            sentences.append(
                f"Customer purchased a {product} in the {category} "
                f"category for ${price:.2f}."
            )
            
        return " ".join(sentences)
    
    def generate_transaction(self) -> Dict[str, Any]:
        """
        Generate sample transaction data.
        
        Returns:
            Dict[str, Any]: Transaction data
        """
        return {
            'amount': random.uniform(10.0, 1000.0),
            'type': random.choice(self.transaction_types),
            'product_id': f"PROD_{random.randint(1000, 9999)}",
            'category': random.choice(self.categories),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_timestamp(self, 
                         days_range: int = 30) -> str:
        """
        Generate a random timestamp within a specified range.
        
        Args:
            days_range: Number of days to look back
            
        Returns:
            str: ISO format timestamp
        """
        random_days = random.randint(0, days_range)
        random_hours = random.randint(0, 24)
        random_minutes = random.randint(0, 60)
        
        timestamp = datetime.now() - timedelta(
            days=random_days,
            hours=random_hours,
            minutes=random_minutes
        )
        
        return timestamp.isoformat()
    
    def generate_multimodal_data(self, 
                               include_image: bool = True) -> Dict[str, Any]:
        """
        Generate sample multimodal data.
        
        Args:
            include_image: Whether to include a generated image
            
        Returns:
            Dict[str, Any]: Multimodal data
        """
        data = {
            'text': self.generate_text(),
            'image': None
        }
        
        if include_image:
            data['image'] = self._generate_sample_image()
            
        return data
    
    def generate_data_object(self) -> Dict[str, Any]:
        """
        Generate a complete data object with all components.
        
        Returns:
            Dict[str, Any]: Complete data object
        """
        return {
            'text': self.generate_text(include_customer=True),
            'transaction': self.generate_transaction(),
            'timestamp': self.generate_timestamp(),
            'multimodal': self.generate_multimodal_data(include_image=True)
        }
    
    def generate_batch(self, 
                      num_items: int = 10,
                      data_type: str = 'all') -> List[Dict[str, Any]]:
        """
        Generate a batch of test data.
        
        Args:
            num_items: Number of items to generate
            data_type: Type of data to generate ('text', 'transaction', 'timestamp', 'multimodal', 'all')
            
        Returns:
            List[Dict[str, Any]]: List of generated data items
        """
        batch = []
        
        for _ in range(num_items):
            if data_type == 'text':
                item = self.generate_text(include_customer=True)
            elif data_type == 'transaction':
                item = self.generate_transaction()
            elif data_type == 'timestamp':
                item = self.generate_timestamp()
            elif data_type == 'multimodal':
                item = self.generate_multimodal_data()
            else:  # 'all'
                item = self.generate_data_object()
                
            batch.append(item)
            
        return batch
    
    def _generate_customer_info(self) -> str:
        """Generate customer information string."""
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)
        age = random.randint(18, 80)
        location = f"City_{random.randint(1, 100)}"
        
        return (
            f"Customer: {first_name} {last_name}, Age: {age}, "
            f"Location: {location}"
        )
    
    def _generate_sample_image(self, 
                             size: tuple = (224, 224)) -> Optional[Image.Image]:
        """
        Generate a sample image for testing.
        
        Args:
            size: Size of the generated image
            
        Returns:
            Optional[Image.Image]: Generated image
        """
        try:
            # Create a new image with random background
            image = Image.new('RGB', size, color=(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ))
            
            # Create a drawing object
            draw = ImageDraw.Draw(image)
            
            # Draw some random shapes
            for _ in range(5):
                # Random rectangle with proper coordinates
                x1 = random.randint(0, size[0] - 1)
                y1 = random.randint(0, size[1] - 1)
                x2 = random.randint(x1 + 1, size[0])
                y2 = random.randint(y1 + 1, size[1])
                
                draw.rectangle(
                    [x1, y1, x2, y2],
                    fill=(
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                )
                
            return image
            
        except Exception as e:
            logger.error(f"Failed to generate sample image: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Create test data generator
    generator = TestDataGenerator()
    
    # Generate various types of test data
    print("\nGenerating text data:")
    text = generator.generate_text(include_customer=True)
    print(text)
    
    print("\nGenerating transaction data:")
    transaction = generator.generate_transaction()
    print(transaction)
    
    print("\nGenerating timestamp:")
    timestamp = generator.generate_timestamp()
    print(timestamp)
    
    print("\nGenerating multimodal data:")
    multimodal = generator.generate_multimodal_data()
    print(multimodal)
    
    print("\nGenerating complete data object:")
    data_object = generator.generate_data_object()
    print(data_object)
    
    print("\nGenerating batch of test data:")
    batch = generator.generate_batch(num_items=3)
    print(f"Generated {len(batch)} items") 