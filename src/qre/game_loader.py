import json
import numpy as np
from typing import List, Dict, Any


def load_games_from_json(filepath: str) -> List[np.ndarray]:
    """
    Load a list of payoff matrices from a JSON file.
    
    Args:
        filepath: Path to the JSON file containing game data
        
    Returns:
        List of numpy arrays, each representing a payoff matrix
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    payoff_matrices = []
    
    # Iterate through all games in the JSON
    for game_id, game_data in data.items():
        # Extract the row player's payoff matrix
        # The JSON stores it as rows, but we need columns for actions
        # So we transpose it
        row_payoffs = np.array(game_data['row']).T
        payoff_matrices.append(row_payoffs)
    
    return payoff_matrices


def load_3x3_20_games() -> List[np.ndarray]:
    """
    Load the 20 3x3 symmetric games from the default location.
    
    Returns:
        List of 20 3x3 payoff matrices
    """
    return load_games_from_json('data/selected_games/3x3_20.json')