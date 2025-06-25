import sys
sys.path.append('.')

from src.qre.game_loader import load_3x3_20_games
from src.qre.continuation import QREContinuation
from src.qre.plotter import plot_qre_branches
from src.qre.nash import find_nash_with_pygambit
import numpy as np
import matplotlib.pyplot as plt


def analyze_game1_with_scaling():
    """Analyze Game 1 with different scalings to find all branches."""
    
    # Load Game 1
    games = load_3x3_20_games()
    game1_original = games[0]
    
    print("GAME 1 - ORIGINAL")
    print("="*60)
    print("Payoff Matrix:")
    print(game1_original)
    
    # Find Nash equilibria
    nash_list = find_nash_with_pygambit(game1_original)
    print(f"\nNash Equilibria: {len(nash_list)} found")
    for i, nash in enumerate(nash_list):
        print(f"  Nash {i+1}: {nash}")
    
    # Test different scalings
    scaling_factors = [1.0, 0.1, 0.01]
    
    for scale in scaling_factors:
        print(f"\n\nTESTING WITH SCALING FACTOR: {scale}")
        print("="*60)
        
        # Scale the payoffs
        game1_scaled = game1_original * scale
        print(f"Scaled payoffs range: [{game1_scaled.min():.2f}, {game1_scaled.max():.2f}]")
        
        # Create QRE object
        qre = QREContinuation(game1_scaled)
        
        # Try different lambda ranges (need to adjust for scaling)
        # When payoffs are scaled down, we need larger lambda values for same effect
        lambda_max = 50.0 / scale if scale > 0 else 50.0
        
        print(f"Using lambda_max = {lambda_max:.1f}")
        
        # Find branches
        branches = qre.find_all_branches(lambda_max=lambda_max)
        
        print(f"\nBranches Found: {len(branches)}")
        
        for i, branch in enumerate(branches):
            print(f"\nBranch {i+1}:")
            print(f"  Points: {len(branch)}")
            print(f"  Lambda range: [{branch[0].lambda_val:.3f}, {branch[-1].lambda_val:.3f}]")
            print(f"  Start: {branch[0].pi}")
            print(f"  End: {branch[-1].pi}")
            
            # For scaled game, lambda * payoff is what matters
            # So effective lambda range is lambda * scale
            effective_lambda_start = branch[0].lambda_val * scale
            effective_lambda_end = branch[-1].lambda_val * scale
            print(f"  Effective lambda range: [{effective_lambda_start:.3f}, {effective_lambda_end:.3f}]")
        
        # Plot the branches for best scaling
        if scale == 0.01:
            fig = plot_qre_branches(
                game1_scaled,
                lambda_range=(0.0, lambda_max),
                title=f"Game 1 - Scaled by {scale}"
            )
            plt.savefig(f'game1_scaled_{scale}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    # Also try the Battle of Sexes game from the script
    print(f"\n\nCOMPARISON: BATTLE OF SEXES FROM analyze_battle_of_sexes.py")
    print("="*60)
    
    # This is the game that works in the original script
    bos = np.array([[3, 0, 0],
                    [5, 1, 5],
                    [0, 0, 3]])
    
    print("Payoff Matrix:")
    print(bos)
    
    qre_bos = QREContinuation(bos)
    branches_bos = qre_bos.find_all_branches(lambda_max=10.0)
    
    print(f"\nBranches Found: {len(branches_bos)}")
    for i, branch in enumerate(branches_bos):
        print(f"\nBranch {i+1}:")
        print(f"  Points: {len(branch)}")
        print(f"  Lambda range: [{branch[0].lambda_val:.3f}, {branch[-1].lambda_val:.3f}]")


if __name__ == "__main__":
    analyze_game1_with_scaling()