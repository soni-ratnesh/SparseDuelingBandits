"""
Sparse Dueling Bandits for Copeland Winner Identification

A library for comparing standard O(N²) dueling bandit algorithms
with sparse feature-based O(k log N) approaches.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_generator import generate_preference_matrix
from .naive_ccb import NaiveCCB
from .sparse_bandit import SparseDuelingBandit

__all__ = [
    "generate_preference_matrix",
    "NaiveCCB", 
    "SparseDuelingBandit"
]
