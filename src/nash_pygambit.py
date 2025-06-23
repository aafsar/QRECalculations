"""
Nash equilibrium solver using pygambit.
"""

import numpy as np
import pygambit as gbt
from typing import List


def find_nash_with_pygambit(payoff_matrix: np.ndarray) -> List[np.ndarray]:
    """
    Find all Nash equilibria using pygambit for a symmetric 3x3 game.
    
    Parameters:
    -----------
    payoff_matrix : np.ndarray
        3x3 payoff matrix for player 1 (player 2's payoffs are the transpose)
    
    Returns:
    --------
    List[np.ndarray]
        List of Nash equilibria, each as a numpy array of probabilities
    """
    # Create the game in pygambit
    g = gbt.Game.from_arrays(payoff_matrix, payoff_matrix.T)
    
    # Find all Nash equilibria
    nash_result = gbt.nash.enummixed_solve(g, rational=False)
    
    # Convert pygambit profiles to numpy arrays
    nash_equilibria = []
    
    # Access the equilibria from the result object
    for profile in nash_result.equilibria:
        # Extract player 1's strategy (symmetric game, so both players have same strategy)
        player1_strategy = convert_pygambit_profile_to_array(profile, g, player_index=0)
        
        # Check if this equilibrium is already in our list (avoid duplicates)
        is_duplicate = False
        for existing in nash_equilibria:
            if np.linalg.norm(existing - player1_strategy) < 1e-4:
                is_duplicate = True
                break
        
        if not is_duplicate:
            nash_equilibria.append(player1_strategy)
    
    return nash_equilibria


def convert_pygambit_profile_to_array(profile, game, player_index=0, cleanup_tol=1e-10) -> np.ndarray:
    """
    Convert a single pygambit mixed strategy profile to numpy array.
    
    Parameters:
    -----------
    profile : pygambit MixedStrategyProfile
        The profile to convert
    game : pygambit Game
        The game object
    player_index : int
        Which player's strategy to extract (default: 0)
    cleanup_tol : float
        Threshold below which probabilities are set to zero
    
    Returns:
    --------
    np.ndarray
        Strategy as numpy array
    """
    player = game.players[player_index]
    strategy = np.array([
        float(profile[player][s]) 
        for s in player.strategies
    ])
    
    # Clean up minute probabilities
    strategy[strategy < cleanup_tol] = 0.0
    if np.sum(strategy) > 0:
        strategy = strategy / np.sum(strategy)  # Renormalize
    
    return strategy


def verify_nash_equilibrium(payoff_matrix: np.ndarray, strategy: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Verify that a strategy profile is indeed a Nash equilibrium for a symmetric game.
    
    Parameters:
    -----------
    payoff_matrix : np.ndarray
        Payoff matrix for player 1
    strategy : np.ndarray
        Mixed strategy to verify
    tol : float
        Tolerance for numerical comparisons
    
    Returns:
    --------
    bool
        True if the strategy is a Nash equilibrium
    """
    # Expected payoffs for each pure strategy
    expected_payoffs = payoff_matrix @ strategy
    
    # Best response payoff
    best_payoff = np.max(expected_payoffs)
    
    # Check Nash condition: all strategies with positive probability must yield best payoff
    for i in range(len(strategy)):
        if strategy[i] > tol:
            if abs(expected_payoffs[i] - best_payoff) > tol:
                return False
    
    return True


