#!/usr/bin/env python3
"""
Simple test to verify QRE calculations for a specific game.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from qre_continuation import QREContinuation
from nash_pygambit import find_nash_with_pygambit


def verify_qre(payoff_matrix, lambda_val, strategy):
    """Verify a strategy is a valid QRE."""
    # Expected payoffs
    u = payoff_matrix @ strategy
    
    # Compute what the QRE should be
    if lambda_val == 0:
        qre_should_be = np.ones(3) / 3
    else:
        exp_vals = np.exp(lambda_val * u)
        qre_should_be = exp_vals / np.sum(exp_vals)
    
    # Check if strategy matches QRE
    error = np.linalg.norm(strategy - qre_should_be)
    
    return error < 1e-6, error, qre_should_be


def main():
    # Simple 3x3 game
    payoff_matrix = np.array([
        [3, 0, 0],
        [0, 2, 0],
        [0, 0, 1]
    ])
    
    print("Testing QRE for simple 3x3 game:")
    print("Payoff matrix:")
    print(payoff_matrix)
    print()
    
    # Find Nash equilibria
    nash_equilibria = find_nash_with_pygambit(payoff_matrix)
    print(f"Found {len(nash_equilibria)} Nash equilibria")
    
    # Initialize QRE solver
    qre_solver = QREContinuation(payoff_matrix)
    branches = qre_solver.find_all_branches(nash_equilibria, lambda_max=10.0)
    print(f"Found {len(branches)} QRE branches")
    print()
    
    # Test specific lambda values
    test_lambdas = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    for lam in test_lambdas:
        print(f"\nTesting λ = {lam}:")
        qre_solutions = qre_solver.find_qre_at_lambda(lam, branches)
        
        for i, qre in enumerate(qre_solutions):
            is_valid, error, expected = verify_qre(payoff_matrix, lam, qre.pi1)
            
            print(f"  Solution {i+1}: [{qre.pi1[0]:.6f}, {qre.pi1[1]:.6f}, {qre.pi1[2]:.6f}]")
            print(f"    Expected: [{expected[0]:.6f}, {expected[1]:.6f}, {expected[2]:.6f}]")
            print(f"    Error: {error:.2e}")
            print(f"    Valid: {'✓' if is_valid else '✗'}")
            
            if not is_valid:
                # Double check with direct calculation
                residual = qre_solver.residual_full(lam, qre.pi1)
                print(f"    Residual from solver: {np.linalg.norm(residual):.2e}")


if __name__ == "__main__":
    main()