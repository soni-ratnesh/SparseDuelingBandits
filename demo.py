#!/usr/bin/env python
"""
Quick Demo

Simple demonstration of the sparse dueling bandits framework.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.data_generator import generate_preference_matrix, get_relevant_features
from src.naive_ccb import NaiveCCB
from src.sparse_bandit import SparseDuelingBandit


def main():
    print("=" * 60)
    print("SPARSE DUELING BANDITS - QUICK DEMO")
    print("=" * 60)
    
    # Parameters
    N = 8    # Items
    z = 15   # Features
    k = 3    # Sparse (only 3 features matter)
    max_samples = 200
    
    print(f"\nProblem Setup:")
    print(f"  N = {N} items")
    print(f"  z = {z} features per item")
    print(f"  k = {k} relevant features (sparse)")
    print(f"  Max comparisons = {max_samples}")
    
    # Generate problem
    print("\n" + "-" * 60)
    print("Generating sparse preference matrix...")
    P, X, w, true_winner, scores = generate_preference_matrix(N, z, k, seed=42)
    
    true_features = get_relevant_features(w)
    print(f"  True relevant features: {true_features}")
    print(f"  True Copeland winner: Item {true_winner}")
    print(f"  Copeland scores: {scores}")
    
    # Run Naive CCB
    print("\n" + "-" * 60)
    print("Running Naive CCB...")
    naive = NaiveCCB(N, delta=0.1)
    
    naive_first_correct = None
    for t in range(max_samples):
        i, j = naive.select_pair()
        i_wins = np.random.random() < P[i, j]
        naive.update(i, j, i_wins)
        
        if naive.get_current_winner() == true_winner and naive_first_correct is None:
            naive_first_correct = t + 1
    
    print(f"  Final estimate: Item {naive.get_current_winner()}")
    print(f"  Correct: {naive.get_current_winner() == true_winner}")
    print(f"  First correct at sample: {naive_first_correct or 'Never'}")
    
    # Run Sparse Bandit
    print("\n" + "-" * 60)
    print("Running Sparse Bandit...")
    sparse = SparseDuelingBandit(N, z, k, X, delta=0.1)
    
    sparse_first_correct = None
    for t in range(max_samples):
        i, j = sparse.select_pair()
        i_wins = np.random.random() < P[i, j]
        sparse.update(i, j, i_wins)
        
        if sparse.get_current_winner() == true_winner and sparse_first_correct is None:
            sparse_first_correct = t + 1
    
    estimated_features = sparse.get_estimated_relevant_features()
    print(f"  Final estimate: Item {sparse.get_current_winner()}")
    print(f"  Correct: {sparse.get_current_winner() == true_winner}")
    print(f"  First correct at sample: {sparse_first_correct or 'Never'}")
    print(f"  Estimated relevant features: {estimated_features}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<20} {'Samples to Correct':<20} {'Final Correct'}")
    print("-" * 60)
    print(f"{'Naive CCB':<20} {str(naive_first_correct or '>'+str(max_samples)):<20} {naive.get_current_winner() == true_winner}")
    print(f"{'Sparse Bandit':<20} {str(sparse_first_correct or '>'+str(max_samples)):<20} {sparse.get_current_winner() == true_winner}")
    
    if naive_first_correct and sparse_first_correct:
        speedup = naive_first_correct / sparse_first_correct
        print(f"\nSpeedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
