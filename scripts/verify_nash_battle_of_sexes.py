#!/usr/bin/env python3
"""
Verify the Nash equilibria for Battle of Sexes variant.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from nash_pygambit import find_nash_with_pygambit, verify_nash_equilibrium


def check_nash_conditions(payoff_matrix, pi, tol=1e-9):
    """
    Check Nash equilibrium conditions manually.
    A strategy pi is a Nash equilibrium if:
    - For all strategies i with pi[i] > 0, u[i] = max(u)
    - For all strategies i with pi[i] = 0, u[i] <= max(u)
    """
    u = payoff_matrix @ pi
    max_u = np.max(u)
    
    print(f"  Strategy: [{pi[0]:.6f}, {pi[1]:.6f}, {pi[2]:.6f}]")
    print(f"  Expected payoffs: [{u[0]:.6f}, {u[1]:.6f}, {u[2]:.6f}]")
    print(f"  Max payoff: {max_u:.6f}")
    
    # Check support
    support = pi > tol
    print(f"  Support: {support}")
    
    # Check conditions
    is_nash = True
    
    # All strategies in support should have equal (max) payoff
    support_payoffs = u[support]
    if len(support_payoffs) > 0:
        if not np.allclose(support_payoffs, max_u, atol=tol):
            print(f"  ERROR: Payoffs in support not equal: {support_payoffs}")
            is_nash = False
    
    # All strategies outside support should have payoff <= max
    non_support_payoffs = u[~support]
    if len(non_support_payoffs) > 0:
        if np.any(non_support_payoffs > max_u + tol):
            print(f"  ERROR: Payoffs outside support exceed max: {non_support_payoffs}")
            is_nash = False
    
    return is_nash


def main():
    # Battle of Sexes variant
    payoff_matrix = np.array([
        [6, 1, 0],
        [0, 3, 1],
        [1, 0, 4]
    ])
    
    print("Battle of Sexes Variant Payoff Matrix:")
    print(payoff_matrix)
    print()
    
    # Find Nash equilibria using pygambit
    nash_equilibria = find_nash_with_pygambit(payoff_matrix)
    print(f"Found {len(nash_equilibria)} Nash equilibria according to pygambit:")
    print()
    
    # Verify each one
    for i, nash in enumerate(nash_equilibria):
        print(f"Nash {i+1}:")
        
        # Use pygambit's verification
        is_nash_pygambit = verify_nash_equilibrium(payoff_matrix, nash)
        print(f"  Pygambit verification: {is_nash_pygambit}")
        
        # Manual verification
        is_nash_manual = check_nash_conditions(payoff_matrix, nash)
        print(f"  Manual verification: {is_nash_manual}")
        
        # For symmetric games, also check if it's a symmetric Nash
        # (i.e., both players use the same strategy)
        payoff_matrix_T = payoff_matrix.T
        u2 = payoff_matrix_T @ nash
        print(f"  Player 2 payoffs if both use this: [{u2[0]:.6f}, {u2[1]:.6f}, {u2[2]:.6f}]")
        
        print()
    
    # Let's also look for symmetric Nash equilibria specifically
    print("="*60)
    print("Searching for symmetric Nash equilibria...")
    print("(Where both players use the same mixed strategy)")
    print()
    
    # Check pure strategies
    for i in range(3):
        pi = np.zeros(3)
        pi[i] = 1.0
        print(f"Pure strategy {i+1}:")
        is_nash = check_nash_conditions(payoff_matrix, pi)
        if is_nash:
            print("  This is a Nash equilibrium!")
        print()
    
    # The mixed Nash should satisfy:
    # If strategies i,j,k are all in support, then:
    # u[i] = u[j] = u[k]
    # This gives us linear equations to solve
    
    print("Looking for fully mixed Nash (all three strategies in support):")
    # For a fully mixed Nash, we need:
    # u[0] = u[1] = u[2]
    # pi[0] + pi[1] + pi[2] = 1
    
    # This gives us the system:
    # 6*pi[0] + 1*pi[1] + 0*pi[2] = 0*pi[0] + 3*pi[1] + 1*pi[2]
    # 6*pi[0] + 1*pi[1] + 0*pi[2] = 1*pi[0] + 0*pi[1] + 4*pi[2]
    # pi[0] + pi[1] + pi[2] = 1
    
    # Simplifying:
    # 6*pi[0] - 2*pi[1] - pi[2] = 0
    # 5*pi[0] + pi[1] - 4*pi[2] = 0
    # pi[0] + pi[1] + pi[2] = 1
    
    A = np.array([
        [6, -2, -1],
        [5, 1, -4],
        [1, 1, 1]
    ])
    b = np.array([0, 0, 1])
    
    try:
        pi_mixed = np.linalg.solve(A, b)
        print(f"  Solution: [{pi_mixed[0]:.6f}, {pi_mixed[1]:.6f}, {pi_mixed[2]:.6f}]")
        
        # Verify it's valid (all probabilities non-negative)
        if np.all(pi_mixed >= -1e-10) and np.all(pi_mixed <= 1 + 1e-10):
            print("  Valid probability distribution!")
            is_nash = check_nash_conditions(payoff_matrix, pi_mixed)
            if is_nash:
                print("  This is a Nash equilibrium!")
        else:
            print("  Invalid: negative probabilities")
    except:
        print("  No solution exists")


if __name__ == "__main__":
    main()