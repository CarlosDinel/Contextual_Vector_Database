# import sys
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from flask import Flask, render_template, request, jsonify


# # Voeg de hoofdmap van het project toe aan de Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # Gebruik absolute imports
# from Contexter.reembedding_orchestra import ReembedingOrchestrator
# from Contexter.contexter_model import Contexter, Vector



  
# app = Flask(__name__)

# # Globale variabelen
# vectors = []
# orchestrator = None
# contexter = None

# def initialize_system():
#     """Initialiseer het systeem met enkele vectoren"""
#     global vectors, orchestrator, contexter
    
#     # Maak contexter en orchestrator
#     contexter = Contexter()
#     orchestrator = ReembedingOrchestrator(contexter=contexter)
    
#     # Maak enkele initiële vectoren
#     vectors = [
#         Vector("v1", np.array([0.9, 0.1, 0.0]), solidness=0.3),
#         Vector("v2", np.array([0.8, 0.2, 0.0]), solidness=0.5),
#         Vector("v3", np.array([0.1, 0.9, 0.0]), solidness=0.7),
#         Vector("v4", np.array([0.0, 0.1, 0.9]), solidness=0.1),
#     ]

# @app.route('/')
# def index():
#     """Render de hoofdpagina"""
#     return render_template('index.html')

# @app.route('/api/vectors', methods=['GET'])
# def get_vectors():
#     """API endpoint om vectoren op te halen"""
#     global vectors
    
#     # Converteer vectoren naar JSON
#     vector_data = []
#     for v in vectors:
#         vector_data.append({
#             'id': v.id,
#             'data': v.data.tolist(),
#             'solidness': v.solidness,
#             'last_update': v.last_update_time
#         })
    
#     return jsonify(vector_data)

# @app.route('/api/reembed', methods=['POST'])
# def reembed_vectors():
#     """API endpoint om vectoren te reembedden"""
#     global vectors, orchestrator
    
#     # Voer reembedding uit
#     vectors = orchestrator.orchestrate_reembedding(vectors)
    
#     # Converteer resultaten naar JSON
#     vector_data = []
#     for v in vectors:
#         vector_data.append({
#             'id': v.id,
#             'data': v.data.tolist(),
#             'solidness': v.solidness,
#             'last_update': v.last_update_time
#         })
    
#     return jsonify(vector_data)

# @app.route('/api/add_vector', methods=['POST'])
# def add_vector():
#     """API endpoint om een nieuwe vector toe te voegen"""
#     global vectors
    
#     # Haal data uit request
#     data = request.json
#     vector_data = np.array(data['data'])
#     solidness = float(data.get('solidness', 0.5))
    
#     # Genereer ID
#     vector_id = f"v{len(vectors) + 1}"
    
#     # Maak nieuwe vector
#     new_vector = Vector(vector_id, vector_data, solidness=solidness)
    
#     # Voeg toe aan vectoren
#     vectors.append(new_vector)
    
#     return jsonify({'id': vector_id, 'status': 'success'})

# @app.route('/api/context', methods=['GET'])
# def get_context():
#     """API endpoint om context informatie op te halen"""
#     global vectors, contexter
    
#     # Bepaal context
#     context_map = contexter.determine_context(vectors)
    
#     # Converteer naar JSON
#     context_data = {}
#     for vector_id, impacts in context_map.items():
#         context_data[vector_id] = impacts
    
#     return jsonify(context_data)

# if __name__ == '__main__':
#     # Initialiseer systeem
#     initialize_system()
    
#     # Start app
#     app.run(debug=True, host='0.0.0.0', port=5000)
# # templates/index.html


"""
Demo for Contexter functionality in Contextual Vector Database

This script demonstrates the core functionality of the Contexter component
by creating a set of test vectors and showing how they interact and reposition
based on contextual relationships.

Author: Carlos D. Almeida
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

# Import from the optimized Contexter implementation
from Contexter.contexter_model import Vector, Contexter

def generate_test_vectors(num_vectors=20, dimensions=3, seed=42):
    """
    Generate test vectors for demonstration.
    
    Args:
        num_vectors: Number of vectors to generate
        dimensions: Dimensionality of vectors
        seed: Random seed for reproducibility
        
    Returns:
        List of Vector objects
    """
    np.random.seed(seed)
    vectors = []
    
    # Generate random vectors
    for i in range(num_vectors):
        # Create vector data with random values
        vector_data = np.random.randn(dimensions)
        
        # Normalize to unit length
        vector_data = vector_data / np.linalg.norm(vector_data)
        
        # Create vector with random solidness
        solidness = np.random.uniform(0.1, 0.5)
        impact_radius = np.random.uniform(0.5, 2.0)
        
        # Create metadata
        metadata = {
            'created_at': time.time(),
            'category': np.random.choice(['A', 'B', 'C']),
            'importance': np.random.uniform(0.1, 1.0)
        }
        
        # Create vector
        vector = Vector(
            id=f"vector_{i}",
            data=vector_data,
            solidness=solidness,
            impact_radius=impact_radius,
            metadata=metadata
        )
        
        vectors.append(vector)
    
    return vectors

def visualize_vectors(vectors, title="Vector Visualization", filename=None):
    """
    Visualize vectors in 3D space.
    
    Args:
        vectors: List of Vector objects
        title: Plot title
        filename: If provided, save plot to this file
    """
    # Extract vector data
    positions = np.array([v.data for v in vectors])
    solidness = np.array([v.solidness for v in vectors])
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vectors
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=solidness,
        cmap='viridis',
        s=100,
        alpha=0.8
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Solidness')
    
    # Add labels
    for i, vector in enumerate(vectors):
        ax.text(
            positions[i, 0],
            positions[i, 1],
            positions[i, 2],
            vector.id,
            size=8
        )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def visualize_context(vectors, context_map, title="Context Visualization", filename=None):
    """
    Visualize contextual relationships between vectors.
    
    Args:
        vectors: List of Vector objects
        context_map: Dictionary mapping vector IDs to lists of (vector_id, impact) tuples
        title: Plot title
        filename: If provided, save plot to this file
    """
    # Create dictionary for easy lookup
    vector_dict = {v.id: v for v in vectors}
    
    # Extract vector data
    positions = np.array([v.data for v in vectors])
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vectors
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c='blue',
        s=50,
        alpha=0.8
    )
    
    # Add labels
    for i, vector in enumerate(vectors):
        ax.text(
            positions[i, 0],
            positions[i, 1],
            positions[i, 2],
            vector.id,
            size=8
        )
    
    # Plot connections
    for vector_id, impacts in context_map.items():
        if vector_id in vector_dict:
            source = vector_dict[vector_id]
            source_pos = source.data
            
            # Plot connections to top 3 influences
            for target_id, impact in impacts[:3]:
                if target_id in vector_dict:
                    target = vector_dict[target_id]
                    target_pos = target.data
                    
                    # Draw line with alpha based on impact
                    ax.plot(
                        [source_pos[0], target_pos[0]],
                        [source_pos[1], target_pos[1]],
                        [source_pos[2], target_pos[2]],
                        'r-',
                        alpha=min(1.0, impact * 5),
                        linewidth=impact * 3
                    )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def run_demo():
    """
    Run the Contexter demonstration.
    """
    print("Starting Contexter Demonstration")
    print("=" * 50)
    
    # Create output directory for visualizations
    output_dir = "contexter_demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate test vectors
    print("Generating test vectors...")
    vectors = generate_test_vectors(num_vectors=15, dimensions=3)
    
    # Visualize initial vectors
    print("Visualizing initial vectors...")
    visualize_vectors(
        vectors, 
        title="Initial Vector Positions",
        filename=os.path.join(output_dir, "initial_vectors.png")
    )
    
    # Create Contexter
    print("Creating Contexter...")
    contexter = Contexter()
    
    # Determine initial context
    print("Determining initial context...")
    context_map = contexter.determine_context(vectors)
    
    # Visualize initial context
    print("Visualizing initial context...")
    visualize_context(
        vectors, 
        context_map, 
        title="Initial Contextual Relationships",
        filename=os.path.join(output_dir, "initial_context.png")
    )
    
    # Reembed vectors
    print("Reembedding vectors...")
    reembedded_vectors = contexter.reembed_vectors(vectors, iterations=5)
    
    # Determine new context
    print("Determining new context...")
    new_context_map = contexter.determine_context(reembedded_vectors)
    
    # Visualize reembedded vectors
    print("Visualizing reembedded vectors...")
    visualize_vectors(
        reembedded_vectors, 
        title="Reembedded Vector Positions",
        filename=os.path.join(output_dir, "reembedded_vectors.png")
    )
    
    # Visualize new context
    print("Visualizing new context...")
    visualize_context(
        reembedded_vectors, 
        new_context_map, 
        title="Updated Contextual Relationships",
        filename=os.path.join(output_dir, "updated_context.png")
    )
    
    print("=" * 50)
    print("Contexter Demonstration Complete")
    print(f"Visualization images saved to {output_dir}/")

if __name__ == "__main__":
    run_demo()
