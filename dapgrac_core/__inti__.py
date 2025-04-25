"""
DaPGRaC core module for the DaPGRaC system.

This module contains the core components of DaPGRaC, which implement the five
key mechanisms: Shooting Edge Velocity, Relationship Magnetism, Relationship Decay,
Relationship Strength, and Segmental & Global Relationship Edges.
"""

from .shooting_edge_velocity import ShootingEdgeVelocity
from .relationship_magnetism import RelationshipMagnetism
from .relationship_decay import RelationshipDecay
from .relationship_strength import RelationshipStrength
from .relationship_edges import RelationshipEdges
from .dapgrac import DaPGRaC

__all__ = [
    'ShootingEdgeVelocity',
    'RelationshipMagnetism',
    'RelationshipDecay',
    'RelationshipStrength',
    'RelationshipEdges',
    'DaPGRaC',
]
