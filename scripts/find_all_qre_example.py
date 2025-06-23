#!/usr/bin/env python3
"""
Example: Finding all QRE solutions at a specific lambda value.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from qre_continuation import QREContinuation
from nash_pygambit import find_nash_with_pygambit, verify_nash_equilibrium

def main():
    # Define the game (same as in the notebooks)
    """
    payoff_matrix = np.array([
        [5, 3, 1],
        [2, 6, 0],
        [4, 3, 5]
    ])
    """
    payoff_matrix = np.array([
        [5, 0, 0],
        [0, 5, 0],
        [0, 0, -10]
    ])
    
    print("Payoff Matrix:")
    print(payoff_matrix)
    print()
    
    # First, find Nash equilibria using pygambit
    print("Finding Nash equilibria using pygambit...")
    nash_equilibria = find_nash_with_pygambit(payoff_matrix)
    print(f"Found {len(nash_equilibria)} Nash equilibria:")
    for i, nash in enumerate(nash_equilibria):
        print(f"Nash {i+1}: [{nash[0]:.6f}, {nash[1]:.6f}, {nash[2]:.6f}]")
    print()
    
    # Initialize the QRE solver
    qre_solver = QREContinuation(payoff_matrix)
    
    # Find all QRE branches using the Nash equilibria
    print("Finding all QRE branches...")
    branches = qre_solver.find_all_branches(nash_equilibria, lambda_max=50.0)
    print(f"Found {len(branches)} branches")
    print()
    
    # Find all QRE solutions at lambda = 1.0
    target_lambda = 1.0
    print(f"Finding all QRE solutions at λ = {target_lambda}:")
    print()
    
    qre_solutions = qre_solver.find_qre_at_lambda(target_lambda, branches)
    
    print(f"Found {len(qre_solutions)} QRE solution(s):")
    for i, qre in enumerate(qre_solutions):
        print(f"\nSolution {i+1}:")
        print(f"  Player 1 strategy: [{qre.pi1[0]:.6f}, {qre.pi1[1]:.6f}, {qre.pi1[2]:.6f}]")
        print(f"  Player 2 strategy: [{qre.pi2[0]:.6f}, {qre.pi2[1]:.6f}, {qre.pi2[2]:.6f}]")
        
        # Verify it's a valid QRE
        residual = qre_solver.residual_full(target_lambda, qre.pi1)
        print(f"  Residual norm: {np.linalg.norm(residual):.2e}")
        
        # Compute expected payoffs
        u1 = qre_solver.expected_payoff(qre.pi1)
        print(f"  Expected payoffs: [{u1[0]:.4f}, {u1[1]:.4f}, {u1[2]:.4f}]")
    
    print("\n" + "="*60)
    
    # Compare with different lambda values
    print("\nNumber of QRE solutions at different λ values:")
    print("λ        # Solutions")
    print("-" * 20)
    
    for lam in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        solutions = qre_solver.find_qre_at_lambda(lam, branches)
        print(f"{lam:6.1f}   {len(solutions)}")
    
    print("\n" + "="*60)
    
    # Display Nash equilibria details
    print("\nNash Equilibria Details:")
    
    for i, nash in enumerate(nash_equilibria):
        print(f"\nNash {i+1}: [{nash[0]:.6f}, {nash[1]:.6f}, {nash[2]:.6f}]")
        u = payoff_matrix @ nash
        print(f"  Expected payoffs: [{u[0]:.4f}, {u[1]:.4f}, {u[2]:.4f}]")
        print(f"  Best response payoff: {np.max(u):.4f}")
        print(f"  Verified as Nash: {verify_nash_equilibrium(payoff_matrix, nash)}")

if __name__ == "__main__":
    main()