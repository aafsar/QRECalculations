import sys
sys.path.append('.')

from src.qre.game_loader import load_3x3_20_games
from src.qre.continuation import QREContinuation
from src.qre.nash import find_nash_with_pygambit
import numpy as np


def analyze_all_games_branches():
    """Calculate and print number of branches for all 20 games."""
    
    # Load all games
    games = load_3x3_20_games()
    
    print("BRANCH ANALYSIS FOR ALL 20 GAMES")
    print("="*80)
    print(f"{'Game':<6} {'Branches':<10} {'Nash Eq.':<10} {'QRE Limits':<12} {'Branch Points':<15}")
    print("-"*80)
    
    total_branches = 0
    total_nash = 0
    games_with_multiple_branches = 0
    
    for i, game in enumerate(games):
        game_num = i + 1
        
        # Find Nash equilibria
        nash_list = find_nash_with_pygambit(game)
        
        # Create QRE object
        qre = QREContinuation(game)
        
        # Count QRE limit points
        qre_limit_count = 0
        for nash in nash_list:
            if qre.is_qre_limit_point(nash, lambda_test=50.0):
                qre_limit_count += 1
        
        # Find branches
        branches = qre.find_all_branches(lambda_max=50.0)
        
        # Count total points across all branches
        total_points = sum(len(branch) for branch in branches)
        
        # Update totals
        total_branches += len(branches)
        total_nash += len(nash_list)
        if len(branches) > 1:
            games_with_multiple_branches += 1
        
        # Print row
        print(f"{game_num:<6} {len(branches):<10} {len(nash_list):<10} {qre_limit_count:<12} {total_points:<15}")
    
    print("-"*80)
    print(f"{'TOTAL':<6} {total_branches:<10} {total_nash:<10}")
    print(f"\nAverage branches per game: {total_branches/len(games):.2f}")
    print(f"Average Nash equilibria per game: {total_nash/len(games):.2f}")
    print(f"Games with multiple branches: {games_with_multiple_branches} / {len(games)}")
    
    # Additional analysis
    print("\n\nDETAILED ANALYSIS OF GAMES WITH MULTIPLE NASH EQUILIBRIA")
    print("="*80)
    
    for i, game in enumerate(games):
        game_num = i + 1
        nash_list = find_nash_with_pygambit(game)
        
        if len(nash_list) > 1:
            print(f"\nGame {game_num}: {len(nash_list)} Nash equilibria")
            
            # Create QRE object
            qre = QREContinuation(game)
            
            # Check each Nash
            for j, nash in enumerate(nash_list):
                is_limit = qre.is_qre_limit_point(nash, lambda_test=50.0)
                print(f"  Nash {j+1}: {nash} - QRE limit: {'YES' if is_limit else 'NO'}")
            
            # Find branches
            branches = qre.find_all_branches(lambda_max=50.0)
            print(f"  Branches found: {len(branches)}")
            
            if len(branches) < len(nash_list):
                print(f"  WARNING: Fewer branches ({len(branches)}) than Nash equilibria ({len(nash_list)})")


if __name__ == "__main__":
    analyze_all_games_branches()