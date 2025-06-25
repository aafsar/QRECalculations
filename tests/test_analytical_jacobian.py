#!/usr/bin/env python3
"""
Test suite for analytical Jacobian calculations in QRE for 3x3 symmetric games.
Tests compare analytical Jacobians against numerical finite difference approximations.
"""

import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.qre.continuation import QREContinuation


class TestAnalyticalJacobian:
    """Test cases for analytical Jacobian calculations in 3x3 symmetric games."""
    
    def numerical_jacobian(self, qre_solver, lambda_val, x, h=1e-8):
        """
        Compute numerical Jacobian using finite differences.
        This serves as the ground truth for testing analytical Jacobians.
        """
        J = np.zeros((2, 3))
        
        # Current residual
        f0 = qre_solver.residual_reduced(lambda_val, x)
        
        # Derivative with respect to lambda
        f_lambda = qre_solver.residual_reduced(lambda_val + h, x)
        J[:, 0] = (f_lambda - f0) / h
        
        # Derivatives with respect to x1, x2
        for i in range(2):
            x_perturb = x.copy()
            x_perturb[i] += h
            f_perturb = qre_solver.residual_reduced(lambda_val, x_perturb)
            J[:, i+1] = (f_perturb - f0) / h
            
        return J
    
    def test_3x3_symmetric_game_jacobian(self):
        """Test Jacobian for various 3x3 symmetric games."""
        # Test multiple game structures
        games = [
            # Game 1: Rock-Paper-Scissors like
            np.array([
                [0, -1, 1],
                [1, 0, -1],
                [-1, 1, 0]
            ]),
            # Game 2: Coordination game
            np.array([
                [5, 0, 0],
                [0, 5, 0],
                [0, 0, 5]
            ]),
            # Game 3: Mixed game
            np.array([
                [3, 1, 2],
                [2, 4, 1],
                [1, 2, 3]
            ])
        ]
        
        for game_idx, payoff_matrix in enumerate(games):
            qre_solver = QREContinuation(payoff_matrix)
            
            # Test points covering the simplex
            test_points = [
                (1.0, np.array([1/3, 1/3])),      # Centroid
                (0.5, np.array([0.5, 0.3])),       # Generic interior
                (2.0, np.array([0.8, 0.1])),       # Near vertex 1
                (1.5, np.array([0.1, 0.8])),       # Near vertex 2
                (1.0, np.array([0.1, 0.1])),       # Near vertex 3
                (0.0, np.array([1/3, 1/3])),       # Lambda = 0
                (10.0, np.array([0.6, 0.3])),      # High lambda
            ]
            
            for lambda_val, x in test_points:
                # Verify point is in valid region
                assert x[0] >= 0 and x[1] >= 0 and x[0] + x[1] <= 1
                
                # Get numerical Jacobian
                J_numerical = self.numerical_jacobian(qre_solver, lambda_val, x)
                
                # Verify shape
                assert J_numerical.shape == (2, 3)
                
                # Test analytical Jacobian
                J_analytical = qre_solver.jacobian_analytical(lambda_val, x)
                np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-5, atol=1e-7,
                                         err_msg=f"Game {game_idx}, lambda={lambda_val}, x={x}")
    
    def test_jacobian_at_special_points(self):
        """Test Jacobian at special points like Nash equilibria and boundaries."""
        payoff_matrix = np.array([
            [5, 0, 0],
            [0, 5, 0],
            [0, 0, -10]
        ])
        
        qre_solver = QREContinuation(payoff_matrix)
        
        # Special test cases
        special_cases = [
            # Pure strategy Nash equilibria
            (50.0, np.array([1.0, 0.0])),     # Pure strategy 1
            (50.0, np.array([0.0, 1.0])),     # Pure strategy 2
            # Mixed strategy Nash (approximate)
            (20.0, np.array([0.5, 0.5])),      # Mixed between 1 and 2
            # Boundary points
            (1.0, np.array([0.5, 0.5])),       # On edge between vertices 1 and 2
            (2.0, np.array([0.0, 0.5])),       # On edge between vertices 2 and 3
            (1.5, np.array([0.5, 0.0])),       # On edge between vertices 1 and 3
        ]
        
        for lambda_val, x in special_cases:
            # Skip invalid points
            if x[0] < 0 or x[1] < 0 or x[0] + x[1] > 1:
                continue
                
            J_numerical = self.numerical_jacobian(qre_solver, lambda_val, x)
            assert J_numerical.shape == (2, 3)
            
            # Test analytical Jacobian
            J_analytical = qre_solver.jacobian_analytical(lambda_val, x)
            np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-5, atol=1e-7,
                                     err_msg=f"Special point: lambda={lambda_val}, x={x}")
    
    def test_jacobian_continuity(self):
        """Test that Jacobian varies continuously with parameters."""
        payoff_matrix = np.array([
            [2, 1, 0],
            [1, 2, 1],
            [0, 1, 2]
        ])
        
        qre_solver = QREContinuation(payoff_matrix)
        
        # Test continuity along a path
        x_fixed = np.array([0.4, 0.3])
        lambdas = np.linspace(0.1, 5.0, 20)
        
        J_prev = None
        for lambda_val in lambdas:
            J_curr = self.numerical_jacobian(qre_solver, lambda_val, x_fixed)
            
            if J_prev is not None:
                # Jacobian should change continuously
                diff = np.linalg.norm(J_curr - J_prev)
                assert diff < 0.5, f"Jacobian changed too much: {diff}"
            
            J_prev = J_curr
            
            # Test analytical Jacobian
            J_analytical = qre_solver.jacobian_analytical(lambda_val, x_fixed)
            np.testing.assert_allclose(J_analytical, J_curr, rtol=1e-6, atol=1e-8,
                                     err_msg=f"Continuity test: lambda={lambda_val}")
    
    def test_jacobian_symmetry(self):
        """Test Jacobian properties for perfectly symmetric games."""
        # Perfectly symmetric game where all strategies are equivalent
        payoff_matrix = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        
        qre_solver = QREContinuation(payoff_matrix)
        
        # At the centroid with symmetric game, certain properties should hold
        lambda_val = 1.0
        x = np.array([1/3, 1/3])
        
        J = self.numerical_jacobian(qre_solver, lambda_val, x)
        
        # For this symmetric game, residual should be zero at centroid
        residual = qre_solver.residual_reduced(lambda_val, x)
        np.testing.assert_allclose(residual, np.zeros(2), atol=1e-10)
        
        # Test analytical Jacobian
        J_analytical = qre_solver.jacobian_analytical(lambda_val, x)
        np.testing.assert_allclose(J_analytical, J, rtol=1e-6, atol=1e-8)
    
    def test_jacobian_lambda_zero(self):
        """Test Jacobian behavior when lambda = 0."""
        payoff_matrix = np.array([
            [3, 1, 0],
            [2, 4, 1],
            [1, 0, 3]
        ])
        
        qre_solver = QREContinuation(payoff_matrix)
        
        # When lambda = 0, all strategies should have equal probability
        lambda_val = 0.0
        test_points = [
            np.array([0.2, 0.3]),
            np.array([0.5, 0.4]),
            np.array([0.1, 0.1]),
        ]
        
        for x in test_points:
            J = self.numerical_jacobian(qre_solver, lambda_val, x)
            
            # At lambda=0, quantal response is always [1/3, 1/3, 1/3]
            qr = qre_solver.quantal_response(lambda_val, 
                                            np.array([x[0], x[1], 1-x[0]-x[1]]))
            np.testing.assert_allclose(qr, np.ones(3)/3, atol=1e-10)
            
            # Test analytical Jacobian
            J_analytical = qre_solver.jacobian_analytical(lambda_val, x)
            np.testing.assert_allclose(J_analytical, J, rtol=1e-6, atol=1e-8,
                                     err_msg=f"Lambda=0 test: x={x}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])