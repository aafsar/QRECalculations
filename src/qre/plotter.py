import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from .continuation import QREContinuation


def plot_qre_branches(payoff_matrix: np.ndarray, 
                     lambda_range: Tuple[float, float] = (0.0, 10.0),
                     title: Optional[str] = None) -> plt.Figure:
    """
    Plot QRE branches for a given game matrix.
    
    Args:
        payoff_matrix: The payoff matrix for the game
        lambda_range: Range of lambda values to explore (min, max)
        title: Optional title for the plot
        
    Returns:
        matplotlib Figure object
    """
    # Initialize QRE continuation
    qre = QREContinuation(payoff_matrix)
    
    # Find and trace branches
    _, lambda_max = lambda_range
    branches_raw = qre.find_all_branches(lambda_max=lambda_max)
    
    # Convert QREPoint objects to tuples of (lambda, strategy)
    branches = []
    for branch in branches_raw:
        branch_data = [(point.lambda_val, point.pi) for point in branch]
        branches.append(branch_data)
    
    # Create figure with only simplex diagram
    n_actions = payoff_matrix.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Strategy simplex projection (for 3x3 games)
    if n_actions == 3:
        # Plot all branches on the same simplex
        for branch_idx, branch in enumerate(branches):
            if len(branch) > 1:
                strategies = np.array([point[1] for point in branch])
                
                # Convert to 2D simplex coordinates with s1 on top, s2 on right
                # s1 at (0.5, sqrt(3)/2), s2 at (1, 0), s3 at (0, 0)
                x = strategies[:, 1] + 0.5 * strategies[:, 0]
                y = np.sqrt(3) / 2 * strategies[:, 0]
                
                ax.plot(x, y, label=f'Branch {branch_idx+1}', 
                        linewidth=2, alpha=0.8)
                
                # Mark start and end points
                ax.scatter(x[0], y[0], marker='o', s=100, 
                           label='Start' if branch_idx == 0 else '')
                ax.scatter(x[-1], y[-1], marker='s', s=100,
                           label='End' if branch_idx == 0 else '')
        
        # Draw simplex boundary
        simplex_x = [0, 1, 0.5, 0]
        simplex_y = [0, 0, np.sqrt(3)/2, 0]
        ax.plot(simplex_x, simplex_y, 'k-', linewidth=2, alpha=0.5)
        
        # Label vertices with s_1, s_2, s_3
        ax.text(0.5, np.sqrt(3)/2 + 0.05, r'$s_1$', fontsize=14, ha='center')
        ax.text(1.05, -0.05, r'$s_2$', fontsize=14)
        ax.text(-0.05, -0.05, r'$s_3$', fontsize=14, ha='right')
        
        ax.set_aspect('equal')
        ax.set_title('Strategy Simplex', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.15, np.sqrt(3)/2 + 0.15)
    else:
        # For non-3x3 games, show phase portrait of first two actions
        for branch_idx, branch in enumerate(branches):
            if len(branch) > 1:
                strategies = np.array([point[1] for point in branch])
                ax.plot(strategies[:, 0], strategies[:, 1], 
                        label=f'Branch {branch_idx+1}',
                        linewidth=2, alpha=0.8)
        
        ax.set_xlabel('P(Action 1)', fontsize=12)
        ax.set_ylabel('P(Action 2)', fontsize=12)
        ax.set_title('Phase Portrait (Actions 1 & 2)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    
    # Add legend if not too many branches
    if len(branches) <= 5:
        ax.legend(loc='best', fontsize=10)
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f'QRE Branches for {n_actions}x{n_actions} Game', fontsize=16)
    
    plt.tight_layout()
    
    # Print branch information
    print(f"Found {len(branches)} branches")
    for i, branch in enumerate(branches):
        print(f"Branch {i+1}: {len(branch)} points, λ range: [{branch[0][0]:.3f}, {branch[-1][0]:.3f}]")
    
    return fig


def plot_multiple_games(games: List[np.ndarray], 
                       lambda_range: Tuple[float, float] = (0.0, 10.0)) -> List[plt.Figure]:
    """
    Plot QRE branches for multiple games.
    
    Args:
        games: List of payoff matrices
        lambda_range: Range of lambda values to explore
        
    Returns:
        List of matplotlib Figure objects
    """
    figures = []
    
    for i, game in enumerate(games):
        fig = plot_qre_branches(
            game, 
            lambda_range=lambda_range,
            title=f'Game {i+1}'
        )
        figures.append(fig)
    
    return figures