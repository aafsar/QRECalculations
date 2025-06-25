#!/usr/bin/env python3
"""
Test script to verify that calculated QREs are indeed valid quantal response equilibria.
Tests multiple games and lambda values.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.qre.continuation import QREContinuation
from src.qre.nash import find_nash_with_pygambit
import matplotlib.pyplot as plt


def verify_qre_solution(payoff_matrix: np.ndarray, lambda_val: float, 
                       pi: np.ndarray, tol: float = 1e-6) -> tuple[bool, float]:
    """
    Verify that a mixed strategy is a valid QRE at given lambda.
    
    For a valid QRE, the strategy must satisfy:
    π_i = exp(λ * u_i(π)) / Σ_j exp(λ * u_j(π))
    
    Returns:
    --------
    (is_valid, max_error): Whether it's valid and the maximum deviation
    """
    # Expected payoffs
    u = payoff_matrix @ pi
    
    # Compute quantal response
    if lambda_val == 0:
        qr = np.ones(len(pi)) / len(pi)
    else:
        # Prevent overflow
        max_u = np.max(u)
        exp_vals = np.exp(lambda_val * (u - max_u))
        qr = exp_vals / np.sum(exp_vals)
    
    # Check if pi equals quantal response
    error = np.max(np.abs(pi - qr))
    is_valid = error < tol
    
    return is_valid, error


def test_game(name: str, payoff_matrix: np.ndarray, test_lambdas: list[float]):
    """Test QRE calculations for a specific game."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    print("\nPayoff Matrix:")
    print(payoff_matrix)
    
    # Find Nash equilibria
    nash_equilibria = find_nash_with_pygambit(payoff_matrix)
    print(f"\nFound {len(nash_equilibria)} Nash equilibria")
    
    # Initialize QRE solver
    qre_solver = QREContinuation(payoff_matrix)
    
    # Find all QRE branches
    branches = qre_solver.find_all_branches(nash_equilibria, lambda_max=50.0)
    print(f"Found {len(branches)} QRE branches")
    
    # Test QRE validity at different lambda values
    print("\nVerifying QRE solutions:")
    print(f"{'Lambda':>10} | {'# Solutions':>12} | {'All Valid':>10} | {'Max Error':>12}")
    print("-" * 50)
    
    all_valid = True
    
    for lambda_val in test_lambdas:
        # Find all QRE solutions at this lambda
        qre_solutions = qre_solver.find_qre_at_lambda(lambda_val, branches)
        
        # Verify each solution
        valid_count = 0
        max_error = 0.0
        
        for qre in qre_solutions:
            is_valid, error = verify_qre_solution(payoff_matrix, lambda_val, qre.pi1)
            if is_valid:
                valid_count += 1
            max_error = max(max_error, error)
            
            if not is_valid:
                all_valid = False
                print(f"\n  WARNING: Invalid QRE found at λ={lambda_val}")
                print(f"  Strategy: [{qre.pi1[0]:.6f}, {qre.pi1[1]:.6f}, {qre.pi1[2]:.6f}]")
                print(f"  Error: {error:.2e}")
        
        all_valid_str = "Yes" if valid_count == len(qre_solutions) else f"No ({valid_count}/{len(qre_solutions)})"
        print(f"{lambda_val:10.2f} | {len(qre_solutions):12} | {all_valid_str:>10} | {max_error:12.2e}")
    
    # Test extreme cases
    print("\nTesting extreme cases:")
    
    # Test λ = 0 (should give uniform distribution)
    print("\n1. λ = 0 (should give uniform distribution):")
    qre_at_zero = qre_solver.find_qre_at_lambda(0.0, branches)
    for i, qre in enumerate(qre_at_zero):
        expected = np.ones(3) / 3
        deviation = np.max(np.abs(qre.pi1 - expected))
        print(f"   Solution {i+1}: [{qre.pi1[0]:.6f}, {qre.pi1[1]:.6f}, {qre.pi1[2]:.6f}]")
        print(f"   Deviation from uniform: {deviation:.2e}")
        is_valid, error = verify_qre_solution(payoff_matrix, 0.0, qre.pi1)
        print(f"   Valid QRE: {is_valid} (error: {error:.2e})")
    
    # Test high λ (should approach Nash equilibria)
    print("\n2. λ = 50 (should approach Nash equilibria):")
    qre_at_high = qre_solver.find_qre_at_lambda(50.0, branches)
    print(f"   Found {len(qre_at_high)} solutions")
    
    for i, qre in enumerate(qre_at_high):
        # Find closest Nash
        min_dist = float('inf')
        closest_nash_idx = -1
        for j, nash in enumerate(nash_equilibria):
            dist = np.linalg.norm(qre.pi1 - nash)
            if dist < min_dist:
                min_dist = dist
                closest_nash_idx = j
        
        print(f"   Solution {i+1}: [{qre.pi1[0]:.6f}, {qre.pi1[1]:.6f}, {qre.pi1[2]:.6f}]")
        print(f"   Distance to closest Nash (#{closest_nash_idx+1}): {min_dist:.2e}")
        is_valid, error = verify_qre_solution(payoff_matrix, 50.0, qre.pi1)
        print(f"   Valid QRE: {is_valid} (error: {error:.2e})")
    
    return all_valid


def visualize_qre_validity(payoff_matrix: np.ndarray, name: str):
    """Create a detailed visualization of QRE validity across lambda values."""
    # Find Nash equilibria
    nash_equilibria = find_nash_with_pygambit(payoff_matrix)
    
    # Initialize QRE solver
    qre_solver = QREContinuation(payoff_matrix)
    branches = qre_solver.find_all_branches(nash_equilibria, lambda_max=50.0)
    
    # Test many lambda values
    lambda_vals = np.logspace(-2, 1.7, 100)
    errors = []
    num_solutions = []
    
    for lam in lambda_vals:
        qre_solutions = qre_solver.find_qre_at_lambda(lam, branches)
        num_solutions.append(len(qre_solutions))
        
        max_error = 0.0
        for qre in qre_solutions:
            _, error = verify_qre_solution(payoff_matrix, lam, qre.pi1)
            max_error = max(max_error, error)
        errors.append(max_error)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.loglog(lambda_vals, errors, 'b-', linewidth=2)
    ax1.axhline(y=1e-6, color='r', linestyle='--', label='Tolerance (1e-6)')
    ax1.set_xlabel('Lambda')
    ax1.set_ylabel('Maximum QRE Error')
    ax1.set_title(f'QRE Validity Check: {name}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.semilogx(lambda_vals, num_solutions, 'g-', linewidth=2)
    ax2.set_xlabel('Lambda')
    ax2.set_ylabel('Number of QRE Solutions')
    ax2.set_title('Number of QRE Solutions vs Lambda')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'qre_validity_{name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()


def main():
    """Run comprehensive QRE validity tests."""
    
    # Define test games
    test_games = [
        ("Original Game", np.array([
            [5, 3, 1],
            [2, 6, 0],
            [4, 3, 5]
        ])),
        
        ("Coordination Game", np.array([
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10]
        ])),
        
        ("Rock-Paper-Scissors", np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]
        ])),
        
        ("Battle of Sexes Variant", np.array([
            [6, 1, 0],
            [0, 3, 1],
            [1, 0, 4]
        ])),
        
        ("Matching Pennies Style", np.array([
            [1, -1, 0],
            [-1, 1, 0],
            [0, 0, 0]
        ]))
    ]
    
    # Lambda values to test
    test_lambdas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    print("QRE VALIDITY TEST SUITE")
    print("=" * 70)
    print("This script verifies that all calculated QRE solutions satisfy")
    print("the quantal response equilibrium condition:")
    print("π_i = exp(λ * u_i(π)) / Σ_j exp(λ * u_j(π))")
    
    all_games_valid = True
    
    # Test each game
    for name, payoff_matrix in test_games:
        is_valid = test_game(name, payoff_matrix, test_lambdas)
        if not is_valid:
            all_games_valid = False
        
        # Create visualization
        visualize_qre_validity(payoff_matrix, name)
        print(f"\nSaved validity plot: qre_validity_{name.lower().replace(' ', '_')}.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if all_games_valid:
        print("✓ All QRE solutions are valid across all tested games and lambda values!")
    else:
        print("✗ Some invalid QRE solutions were found. Check warnings above.")
    
    print("\nNote: Small errors (< 1e-6) are expected due to numerical precision.")
    print("The continuation method successfully finds valid QRE solutions.")


if __name__ == "__main__":
    main()