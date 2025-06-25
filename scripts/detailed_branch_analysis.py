import sys
sys.path.append('.')

from src.qre.game_loader import load_3x3_20_games
from src.qre.continuation import QREContinuation
from src.qre.nash import find_nash_with_pygambit
import numpy as np


def analyze_game_branches(game_idx, game):
    """Detailed analysis of branches for a specific game."""
    
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS - GAME {game_idx + 1}")
    print(f"{'='*80}")
    
    print("\nPayoff Matrix:")
    print(game)
    
    # Find Nash equilibria
    nash_list = find_nash_with_pygambit(game)
    print(f"\nNash Equilibria: {len(nash_list)} found")
    for i, nash in enumerate(nash_list):
        print(f"  Nash {i+1}: {nash}")
    
    # Create QRE object and find branches
    qre = QREContinuation(game)
    
    # Check which Nash are QRE limit points
    print("\nQRE Limit Point Check:")
    qre_limit_points = []
    for i, nash in enumerate(nash_list):
        is_limit = qre.is_qre_limit_point(nash, lambda_test=50.0)
        print(f"  Nash {i+1}: {'YES' if is_limit else 'NO'} (QRE limit point)")
        if is_limit:
            qre_limit_points.append(nash)
    
    # Find branches
    branches = qre.find_all_branches(lambda_max=10.0)
    
    print(f"\nBranches Found: {len(branches)}")
    for i, branch in enumerate(branches):
        print(f"\nBranch {i+1}:")
        print(f"  Points: {len(branch)}")
        print(f"  Lambda range: [{branch[0].lambda_val:.3f}, {branch[-1].lambda_val:.3f}]")
        print(f"  Start: {branch[0].pi}")
        print(f"  End: {branch[-1].pi}")
        
        # Check which Nash this branch connects to
        endpoint = branch[-1]
        min_dist = float('inf')
        closest_nash = None
        for j, nash in enumerate(nash_list):
            dist = np.linalg.norm(endpoint.pi - nash)
            if dist < min_dist:
                min_dist = dist
                closest_nash = j
        print(f"  Closest Nash: Nash {closest_nash+1} (distance: {min_dist:.6f})")
    
    # Check if we're missing branches
    if len(qre_limit_points) > len(branches):
        print(f"\nWARNING: Found {len(qre_limit_points)} QRE limit points but only {len(branches)} branches!")
        print("Some branches may be missing.")


def main():
    """Analyze a few selected games in detail."""
    
    games = load_3x3_20_games()
    
    # Analyze games with known multiple Nash equilibria
    test_indices = [0, 4, 12]  # Games 1, 5, and 13
    
    for idx in test_indices:
        analyze_game_branches(idx, games[idx])
    
    # Also test a custom game with known bifurcation
    print(f"\n{'='*80}")
    print("TESTING CUSTOM GAME WITH KNOWN BIFURCATION")
    print(f"{'='*80}")
    
    # A game with known bifurcation
    custom_game = np.array([[3, 0, 0],
                           [0, 2, 0],
                           [0, 0, 1]])
    
    analyze_game_branches(-1, custom_game)


if __name__ == "__main__":
    main()