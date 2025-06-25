import sys
sys.path.append('.')

from src.qre.game_loader import load_3x3_20_games
from src.qre.plotter import plot_qre_branches
import matplotlib.pyplot as plt
import os


def plot_all_games():
    """Plot QRE branches for all 20 games and save to data/plots/."""
    
    # Load all games
    games = load_3x3_20_games()
    print(f"Loaded {len(games)} games")
    
    # Create plots directory if it doesn't exist
    os.makedirs('data/plots', exist_ok=True)
    
    # Plot each game
    for i, game in enumerate(games):
        game_number = i + 1
        print(f"\nProcessing Game {game_number}...")
        
        try:
            # Plot QRE branches
            fig = plot_qre_branches(
                game,
                lambda_range=(0.0, 50.0),  # Increased range to capture all branches
                title=f"QRE Branches - Game {game_number}"
            )
            
            # Save the plot
            filename = f"data/plots/qre_branches_game{game_number}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Close figure to free memory
            
            print(f"Saved: {filename}")
            
        except Exception as e:
            print(f"Error processing Game {game_number}: {str(e)}")
            continue
    
    print("\nAll games processed!")


if __name__ == "__main__":
    plot_all_games()