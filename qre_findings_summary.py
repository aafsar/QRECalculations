"""
Summary of findings about pygambit QRE methods for finding all solutions.
"""

import pygambit as gbt
import numpy as np
import matplotlib.pyplot as plt

def summarize_qre_methods():
    """Summarize available QRE methods and their capabilities."""
    
    print("=== PYGAMBIT QRE METHODS SUMMARY ===\n")
    
    print("AVAILABLE FUNCTIONS:")
    print("-------------------")
    
    print("\n1. gbt.qre.logit_solve_lambda(game, lam)")
    print("   - Computes QRE profiles at specific lambda values")
    print("   - Parameters:")
    print("     * game: The game to analyze")
    print("     * lam: List of lambda values")
    print("   - Returns: List of LogitQREMixedStrategyProfile objects")
    print("   - Use case: When you need QRE at specific rationality levels")
    
    print("\n2. gbt.qre.logit_solve_branch(game, **kwargs)")
    print("   - Traces the principal QRE branch from lambda=0 to Nash equilibrium")
    print("   - Parameters:")
    print("     * game: The game to analyze")
    print("     * use_strategic: bool (default=False)")
    print("     * maxregret: float (default=1e-08)")
    print("     * first_step: float (default=0.03) - Initial step size")
    print("     * max_accel: float (default=1.1) - Maximum step acceleration")
    print("   - Returns: List of profiles along the branch")
    print("   - Note: Step size is based on arc length, not lambda")
    
    print("\n3. gbt.qre.logit_solve() - NOT AVAILABLE in current version")
    print("   - Would return only the limiting Nash equilibrium")
    
    print("\n\nLIMITATIONS FOR FINDING ALL QRE SOLUTIONS:")
    print("------------------------------------------")
    
    print("\n1. Single Branch Only:")
    print("   - Both functions follow only the PRINCIPAL branch")
    print("   - This branch starts at centroid (uniform distribution)")
    print("   - Converges to ONE specific Nash equilibrium")
    
    print("\n2. No Multiple Branch Support:")
    print("   - Cannot find other QRE branches automatically")
    print("   - Cannot specify starting points for branch tracing")
    print("   - No bifurcation detection or branch switching")
    
    print("\n3. Profile Object Attributes:")
    print("   - profile.lam: The lambda value")
    print("   - profile.profile: The mixed strategy profile")
    print("   - profile.game: Reference to the game")
    print("   - profile.log_like: Log-likelihood value")

def demonstrate_limitations():
    """Demonstrate the limitations with a concrete example."""
    
    print("\n\n=== DEMONSTRATION: Multiple Nash Equilibria ===\n")
    
    # Create a game with 3 Nash equilibria
    game = gbt.Game.from_arrays(
        [[3, 0], [0, 2]],
        [[3, 0], [0, 2]],
        title="Coordination Game"
    )
    
    # Find all Nash equilibria
    nash_result = gbt.nash.enummixed_solve(game)
    nash_eqs = nash_result.equilibria if hasattr(nash_result, 'equilibria') else [nash_result]
    
    print(f"Game has {len(nash_eqs)} Nash equilibria:")
    for i, eq in enumerate(nash_eqs):
        probs = [float(eq[game.players[0]][s]) for s in game.players[0].strategies]
        print(f"  Nash {i+1}: [{probs[0]:.4f}, {probs[1]:.4f}]")
    
    # Trace QRE branch
    branch = gbt.qre.logit_solve_branch(game)
    
    # Extract lambda values and first player's strategy probabilities
    lambdas = [p.lam for p in branch]
    probs_s1 = [float(p.profile[game.players[0]][game.players[0].strategies[0]]) for p in branch]
    
    print(f"\nQRE branch traced {len(branch)} points")
    print(f"Lambda range: {lambdas[0]:.4f} to {lambdas[-1]:.4f}")
    print(f"Starting point: [{probs_s1[0]:.4f}, {1-probs_s1[0]:.4f}]")
    print(f"Ending point: [{probs_s1[-1]:.4f}, {1-probs_s1[-1]:.4f}]")
    
    print("\nOBSERVATION:")
    print("- QRE branch converges to Nash equilibrium [1.0, 0.0]")
    print("- Other Nash equilibria ([0.4, 0.6] and [0.0, 1.0]) are NOT reached")
    print("- No way to find QRE branches leading to these equilibria")
    
    # Plot the QRE correspondence
    plt.figure(figsize=(10, 6))
    plt.semilogx(lambdas, probs_s1, 'b-', linewidth=2, label='Principal QRE Branch')
    
    # Mark Nash equilibria
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Nash: [1.0, 0.0]')
    plt.axhline(y=0.4, color='g', linestyle='--', alpha=0.5, label='Nash: [0.4, 0.6]')
    plt.axhline(y=0.0, color='orange', linestyle='--', alpha=0.5, label='Nash: [0.0, 1.0]')
    
    plt.xlabel('Lambda (log scale)', fontsize=12)
    plt.ylabel('Probability of Strategy 1', fontsize=12)
    plt.title('QRE Correspondence: Only Principal Branch Found', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('/Users/atahan/Documents/Research_local/Projects_local/QRECalculations/qre_single_branch.png', dpi=150)
    print("\nPlot saved as 'qre_single_branch.png'")

def suggest_workarounds():
    """Suggest potential workarounds for finding all QRE solutions."""
    
    print("\n\n=== POTENTIAL WORKAROUNDS ===\n")
    
    print("1. THEORETICAL APPROACH:")
    print("   - Study the game structure to predict multiple branches")
    print("   - Use symmetry arguments to identify potential branches")
    print("   - Apply bifurcation theory to find branch points")
    
    print("\n2. NUMERICAL METHODS:")
    print("   - Implement custom QRE solver with continuation methods")
    print("   - Use AUTO or MatCont for numerical bifurcation analysis")
    print("   - Solve QRE fixed-point equations directly")
    
    print("\n3. HEURISTIC SEARCH:")
    print("   - Sample many lambda values with logit_solve_lambda()")
    print("   - Look for discontinuities or jumps in solutions")
    print("   - Try perturbations of the game to explore nearby branches")
    
    print("\n4. ALTERNATIVE TOOLS:")
    print("   - Consider other game theory software (e.g., Gambit GUI)")
    print("   - Use specialized QRE packages if available")
    print("   - Implement QRE equations in optimization software")

def main():
    """Main function."""
    
    summarize_qre_methods()
    demonstrate_limitations()
    suggest_workarounds()
    
    print("\n\n=== CONCLUSION ===")
    print("\nPygambit provides robust tools for computing QRE along the principal branch,")
    print("but does NOT have built-in functionality to find ALL QRE solutions or branches.")
    print("For games with multiple equilibria, additional branches may exist but require")
    print("custom implementation or alternative approaches to discover.")

if __name__ == "__main__":
    main()