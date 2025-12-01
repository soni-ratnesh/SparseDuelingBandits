# Sparse Dueling Bandits

A framework for comparing standard O(N²) dueling bandit algorithms with sparse feature-based O(k log N) approaches for Copeland Winner identification.

## 📁 Project Structure

```
sparse_dueling_bandits/
│
├── src/                        # Source code
│   ├── __init__.py            # Package init
│   ├── data_generator.py      # Sparse preference matrix generation
│   ├── naive_ccb.py           # Naive CCB algorithm O(N²)
│   ├── sparse_bandit.py       # Sparse feature bandit O(k log N)
│   ├── evaluation.py          # Experiment utilities
│   └── plotting.py            # Visualization utilities
│
├── experiments/               # Experiment scripts (optional)
├── results/                   # Output plots and data
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_all.py
│
├── demo.py                    # Quick demonstration
├── run_experiment.py          # Main experiment runner
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Demo

```bash
python demo.py
```

### 3. Run Full Experiment

```bash
python run_experiment.py
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
# or
python tests/test_all.py
```

## 📖 Usage

### Basic Usage

```python
from src import generate_preference_matrix, NaiveCCB, SparseDuelingBandit

# Generate problem: N items, z features, k sparse
P, X, w, winner, scores = generate_preference_matrix(N=16, z=20, k=3, seed=42)

# Run Naive CCB
naive = NaiveCCB(N=16, delta=0.1)
naive_metrics = naive.run(P, max_samples=500)

# Run Sparse Bandit
sparse = SparseDuelingBandit(N=16, z=20, k=3, X=X, delta=0.1)
sparse_metrics = sparse.run(P, max_samples=500)

print(f"Naive final regret: {naive_metrics['cumulative_regret'][-1]}")
print(f"Sparse final regret: {sparse_metrics['cumulative_regret'][-1]}")
```

### Command Line Options

```bash
python run_experiment.py --help

Options:
  -N             Number of items (default: 16)
  -z             Feature dimension (default: 20)
  -k             Sparsity level (default: 3)
  --runs         Number of runs (default: 20)
  --max-samples  Max samples per run (default: 500)
  --scaling      Run scaling experiment
  --output-dir   Output directory (default: results)
```

### Examples

```bash
# Basic experiment
python run_experiment.py -N 16 -z 20 -k 3

# With scaling analysis
python run_experiment.py -N 16 -z 20 -k 3 --scaling

# Large scale
python run_experiment.py -N 32 -z 50 -k 5 --runs 30 --max-samples 1000
```

## 🔬 Problem Formulation

### Sparse Feature Model

- **N items**, each with **z features**: X ∈ ℝ^(N×z)
- **k-sparse weights**: w ∈ ℝ^z with only k non-zero entries
- **Item scores**: s_i = x_i^T w
- **Preferences**: P_ij = σ(s_i - s_j) (Bradley-Terry model)
- **Goal**: Find Copeland winner (item beating most others)

### Key Insight

Standard methods need O(N²) comparisons to estimate all pairwise preferences.
When preferences depend on only k features, we can identify the winner with O(k log N) comparisons.

## 📊 Algorithms

### 1. Naive CCB (Copeland Confidence Bound)

```
Complexity: O(N²)
Strategy: Estimate all pairwise preferences using UCB
```

### 2. Sparse Dueling Bandit

```
Complexity: O(k · poly(log N))
Strategy: 
  - Phase 1: Random exploration
  - Phase 2: Informative sampling for LASSO recovery
  - Phase 3: UCB exploitation
```

## 📈 Results

| N | N² | Naive CCB | Sparse Bandit | Speedup |
|---|-----|-----------|---------------|---------|
| 8 | 64 | 78 | 29 | **2.7x** |
| 16 | 256 | 331 | 112 | **3.0x** |
| 32 | 1024 | 762 | 78 | **9.8x** |

Speedup **increases** with N due to better scaling.

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_all.py::TestDataGenerator -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## 📚 API Reference

### `generate_preference_matrix(N, z, k, seed=None)`

Generate sparse preference matrix.

**Parameters:**
- `N`: Number of items
- `z`: Feature dimension  
- `k`: Sparsity (k ≤ z)
- `seed`: Random seed

**Returns:**
- `P`: Preference matrix (N×N)
- `X`: Feature matrix (N×z)
- `w`: Sparse weights (z,)
- `winner`: True Copeland winner
- `scores`: Copeland scores

### `NaiveCCB(N, delta=0.1)`

Naive Copeland Confidence Bound algorithm.

**Methods:**
- `run(P_true, max_samples)`: Run algorithm
- `get_current_winner()`: Get estimated winner
- `get_estimate()`: Get estimated preference matrix

### `SparseDuelingBandit(N, z, k, X, delta=0.1)`

Sparse feature-based algorithm.

**Methods:**
- `run(P_true, max_samples)`: Run algorithm
- `get_current_winner()`: Get estimated winner
- `get_estimated_relevant_features()`: Get identified features

## 📄 License

MIT License

## 🔗 References

1. Yue et al. "The K-armed Dueling Bandits Problem" (COLT 2012)
2. Zoghi et al. "Copeland Dueling Bandits" (NeurIPS 2015)
3. Tibshirani "Regression Shrinkage via Lasso" (JRSS 1996)
