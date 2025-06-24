#!/usr/bin/env python3
"""
Analyze QRE branches for Battle of Sexes variant game to understand large errors.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from qre_continuation import QREContinuation
from nash_pygambit import find_nash_with_pygambit
import matplotlib.pyplot as plt


def verify_qre_point(payoff_matrix, lambda_val, pi):
    """Verify if a point is a valid QRE."""
    u = payoff_matrix @ pi
    
    if lambda_val == 0:
        qr = np.ones(3) / 3
    else:
        max_u = np.max(u)
        exp_vals = np.exp(lambda_val * (u - max_u))
        qr = exp_vals / np.sum(exp_vals)
    
    error = np.max(np.abs(pi - qr))
    return error, qr, u


def main():
    # Battle of Sexes variant
    payoff_matrix = np.array([
        [10, 1, 0],
        [0, 3, 5],
        [1, 4, 3]
    ])
    
    print("Made-up Game Payoff Matrix:")
    print(payoff_matrix)
    print()
    
    # Find Nash equilibria
    nash_equilibria = find_nash_with_pygambit(payoff_matrix)
    print(f"Found {len(nash_equilibria)} Nash equilibria:")
    for i, nash in enumerate(nash_equilibria):
        print(f"Nash {i+1}: [{nash[0]:.6f}, {nash[1]:.6f}, {nash[2]:.6f}]")
        error, _, u = verify_qre_point(payoff_matrix, 1000, nash)  # High lambda
        print(f"  Expected payoffs: [{u[0]:.4f}, {u[1]:.4f}, {u[2]:.4f}]")
        print(f"  Verification error at λ=1000: {error:.2e}")
    print()
    
    # Initialize QRE solver
    qre_solver = QREContinuation(payoff_matrix)
    
    # Find all branches
    branches = qre_solver.find_all_branches(nash_equilibria, lambda_max=50.0)
    print(f"\nFound {len(branches)} QRE branches")
    
    # Analyze each branch
    for i, branch in enumerate(branches):
        print(f"\n{'='*60}")
        print(f"Branch {i+1}: {len(branch)} points")
        print(f"Lambda range: [{branch[0].lambda_val:.3f}, {branch[-1].lambda_val:.3f}]")
        
        # Check start and end points
        print(f"Start: λ={branch[0].lambda_val:.3f}, π=[{branch[0].pi[0]:.6f}, {branch[0].pi[1]:.6f}, {branch[0].pi[2]:.6f}]")
        print(f"End:   λ={branch[-1].lambda_val:.3f}, π=[{branch[-1].pi[0]:.6f}, {branch[-1].pi[1]:.6f}, {branch[-1].pi[2]:.6f}]")
        
        # Sample some points along the branch and verify
        print("\nSampling points along branch:")
        sample_indices = np.linspace(0, len(branch)-1, min(10, len(branch)), dtype=int)
        
        max_error = 0
        worst_point = None
        
        for idx in sample_indices:
            point = branch[idx]
            error, qr, u = verify_qre_point(payoff_matrix, point.lambda_val, point.pi)
            
            if error > max_error:
                max_error = error
                worst_point = (idx, point, error, qr)
            
            if error > 1e-3:  # Flag large errors
                print(f"  idx={idx}, λ={point.lambda_val:.3f}: ERROR={error:.2e}")
                print(f"    π = [{point.pi[0]:.6f}, {point.pi[1]:.6f}, {point.pi[2]:.6f}]")
                print(f"    QR= [{qr[0]:.6f}, {qr[1]:.6f}, {qr[2]:.6f}]")
                print(f"    u = [{u[0]:.4f}, {u[1]:.4f}, {u[2]:.4f}]")
        
        if worst_point:
            idx, point, error, qr = worst_point
            print(f"\nWorst error on this branch: {error:.2e} at index {idx}")
    
    # Now specifically check problematic lambda values
    print(f"\n{'='*60}")
    print("Checking specific lambda values with high errors:")
    
    problematic_lambdas = [1.0, 2.0, 5.0]
    
    for lambda_val in problematic_lambdas:
        print(f"\nλ = {lambda_val}:")
        qre_solutions = qre_solver.find_qre_at_lambda(lambda_val, branches)
        print(f"Found {len(qre_solutions)} solutions")
        
        for j, qre in enumerate(qre_solutions):
            error, qr, u = verify_qre_point(payoff_matrix, lambda_val, qre.pi)

            print(f"{j+1}. π = [{qre.pi[0]:.6f}, {qre.pi[1]:.6f}, {qre.pi[2]:.6f}]")
            
            if error > 1e-4:
                print(f"\n  Solution {j+1} has large error: {error:.2e}")
                print(f"    π = [{qre.pi[0]:.6f}, {qre.pi[1]:.6f}, {qre.pi[2]:.6f}]")
                print(f"    QR= [{qr[0]:.6f}, {qr[1]:.6f}, {qr[2]:.6f}]")
                print(f"    Δ = [{qre.pi[0]-qr[0]:.6f}, {qre.pi[1]-qr[1]:.6f}, {qre.pi[2]-qr[2]:.6f}]")
                print(f"    u = [{u[0]:.4f}, {u[1]:.4f}, {u[2]:.4f}]")
                
                # Check which branch this came from
                for bi, branch in enumerate(branches):
                    for pi in range(len(branch)-1):
                        if branch[pi].lambda_val <= lambda_val <= branch[pi+1].lambda_val:
                            dist1 = np.linalg.norm(branch[pi].pi - qre.pi)
                            dist2 = np.linalg.norm(branch[pi+1].pi - qre.pi)
                            if dist1 < 0.1 or dist2 < 0.1:
                                print(f"    From branch {bi+1}, between indices {pi} and {pi+1}")
                                print(f"    Branch point errors:")
                                e1, _, _ = verify_qre_point(payoff_matrix, branch[pi].lambda_val, branch[pi].pi1)
                                e2, _, _ = verify_qre_point(payoff_matrix, branch[pi+1].lambda_val, branch[pi+1].pi1)
                                print(f"      Point {pi}: λ={branch[pi].lambda_val:.3f}, error={e1:.2e}")
                                print(f"      Point {pi+1}: λ={branch[pi+1].lambda_val:.3f}, error={e2:.2e}")
                                break
    
    # Create visualization
    print(f"\n{'='*60}")
    print("Creating branch visualization...")
    
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    # Plot branches in strategy space
    for i, branch in enumerate(branches):
        pi1_vals = [p.pi[0] for p in branch]
        pi2_vals = [p.pi[1] for p in branch]
        lambda_vals = [p.lambda_val for p in branch]
        
        # Color by lambda value
        sc = ax1.scatter(pi1_vals, pi2_vals,
                         s=5, label=f'Branch {i+1}')
    
    ax1.set_xlabel('π₁')
    ax1.set_ylabel('π₂')
    ax1.set_title('QRE Branches in Strategy Space')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Add Nash equilibria
    for i, nash in enumerate(nash_equilibria):
        ax1.plot(nash[0], nash[1], 'r*', markersize=10, label=f'Nash {i+1}' if i == 0 else '')
    
    # Plot error along branches
    """
    for i, branch in enumerate(branches):
        lambda_vals = []
        errors = []
        
        for point in branch:
            error, _, _ = verify_qre_point(payoff_matrix, point.lambda_val, point.pi1)
            lambda_vals.append(point.lambda_val)
            errors.append(error)
        
        ax2.semilogy(lambda_vals, errors, '-', label=f'Branch {i+1}')
    
    ax2.axhline(y=1e-6, color='r', linestyle='--', label='Tolerance')
    ax2.set_xlabel('λ')
    ax2.set_ylabel('QRE Error')
    ax2.set_title('QRE Verification Error by Lambda')
    ax2.grid(True, alpha=0.3)
    """
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig('branches_other_game.png', dpi=150)
    print("\nSaved visualization to battle_of_sexes_branch_analysis.png")


if __name__ == "__main__":
    main()