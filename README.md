# QRE Calculations

A Python implementation for computing Quantal Response Equilibrium (QRE) branches in 3x3 symmetric 2-player games using continuation methods with analytical Jacobians.

## Overview

This project provides tools for analyzing Quantal Response Equilibrium (QRE) in game theory. QRE is a solution concept that generalizes Nash equilibrium by assuming players make noisy best responses rather than perfect best responses. The noise level is controlled by a rationality parameter λ (lambda), where:
- λ = 0 corresponds to completely random play
- λ → ∞ corresponds to Nash equilibrium

The implementation focuses on:
- Computing QRE branches using numerical continuation methods
- Analytical Jacobian calculations for improved numerical stability
- Support for 3x3 symmetric 2-player games
- Visualization of equilibrium paths

## Mathematical Background

### Quantal Response Equilibrium (QRE)
In QRE, players choose actions according to a logit response function:

```
σᵢ(λ) = exp(λ · Eᵢ) / Σⱼ exp(λ · Eⱼ)
```

Where:
- σᵢ is the probability of choosing action i
- λ is the rationality parameter
- Eᵢ is the expected payoff from action i

### Continuation Methods
The project uses predictor-corrector continuation methods to trace QRE branches:
1. **Predictor Step**: Tangent predictor using the nullspace of the Jacobian
2. **Corrector Step**: Newton-Raphson iteration to find the exact solution

### Analytical Jacobian
The implementation computes analytical Jacobians for the QRE system, providing:
- Improved numerical stability
- Faster convergence
- More accurate branch following

## Project Structure

```
QRECalculations/
├── src/                      # Source code
│   ├── qre_continuation.py   # Main QRE continuation implementation
│   └── nash_pygambit.py      # Nash equilibrium finding using PyGambit
├── tests/                    # Test suite
│   ├── test_analytical_jacobian.py  # Tests for Jacobian calculations
│   └── test_qre_validity.py         # QRE validity testing
├── scripts/                  # Analysis scripts
│   ├── analyze_battle_of_sexes.py   # Battle of Sexes game analysis
│   └── ...                          # Other game analyses
├── data/                     # Data files
├── notebooks/                # Jupyter notebooks
└── extras/                   # Additional resources
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/QRECalculations.git
cd QRECalculations
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install numpy scipy matplotlib pygambit pytest
```

## Usage

### Basic Example

```python
from src.qre_continuation import QREContinuation
import numpy as np

# Define a 3x3 symmetric game payoff matrix
payoff_matrix = np.array([
    [3, 0, 5],
    [5, 1, 1],
    [0, 1, 3]
])

# Create QRE solver
qre_solver = QREContinuation(payoff_matrix)

# Find QRE branches
branches = qre_solver.find_all_branches(
    lambda_min=0.0,
    lambda_max=10.0,
    num_points=100
)

# Visualize results
qre_solver.plot_branches(branches)
```

### Running Analysis Scripts

Analyze specific games:
```bash
python scripts/analyze_battle_of_sexes.py
```

### Running Tests

Run the test suite:
```bash
pytest tests/
```

## Key Features

- **Robust Continuation**: Handles branch detection and following with adaptive step sizes
- **Analytical Jacobians**: Improves numerical stability and convergence
- **Multiple Branch Detection**: Finds all QRE branches including unstable ones
- **Visualization Tools**: Built-in plotting for equilibrium paths
- **Nash Equilibrium Integration**: Uses PyGambit for Nash equilibrium computation
- **Comprehensive Testing**: Unit tests for all major components

## Documentation

- `qre_code_review.md` - Detailed code review and architecture overview
- `qre_algorithmic_fixes_detailed.md` - Technical details on algorithmic improvements

## Examples

The project includes analysis scripts for various game types:
- Battle of the Sexes variants
- Coordination games
- Rock-Paper-Scissors
- Matching Pennies

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- McKelvey, R. D., & Palfrey, T. R. (1995). Quantal response equilibria for normal form games. Games and Economic Behavior, 10(1), 6-38.
- Turocy, T. L. (2005). A dynamic homotopy interpretation of the logistic quantal response equilibrium correspondence. Games and Economic Behavior, 51(2), 243-263.

## Contact

For questions or collaborations, please open an issue on GitHub.