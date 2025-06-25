"""
QRE (Quantal Response Equilibrium) calculations package.

This package provides tools for computing and analyzing Quantal Response Equilibria
in game theory, particularly for 3x3 symmetric games.
"""

from .continuation import QREContinuation, QREPoint
from .nash import find_nash_with_pygambit
from .game_loader import load_3x3_20_games, load_games_from_json
from .plotter import plot_qre_branches, plot_multiple_games

__version__ = "0.1.0"

__all__ = [
    "QREContinuation",
    "QREPoint",
    "find_nash_with_pygambit",
    "load_3x3_20_games", 
    "load_games_from_json",
    "plot_qre_branches",
    "plot_multiple_games",
]