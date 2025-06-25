"""
Continuation method for finding all QRE branches in 3x3 symmetric games.

This module implements a pseudo-arclength predictor-corrector algorithm to trace
Quantal Response Equilibrium (QRE) branches in symmetric 3x3 games. The algorithm
finds all solution branches by starting from both the centroid and Nash equilibria.
"""

import numpy as np
from scipy.linalg import solve, svd
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from .nash import find_nash_with_pygambit


@dataclass
class QREPoint:
    """Represents a point on a symmetric QRE branch."""
    lambda_val: float
    pi: np.ndarray  # Mixed strategy (same for both players in symmetric game)


@dataclass
class Diagnostic:
    """Diagnostic information for bifurcation detection."""
    point: np.ndarray      # [λ, x₁, x₂]
    tangent: np.ndarray    # [dλ/ds, dx₁/ds, dx₂/ds]
    sigma_min: float       # Minimum singular value
    det_sign: int          # Sign of determinant
    jacobian: np.ndarray   # Full Jacobian for branch switching
    
    
class QREContinuation:
    """
    Computes symmetric QRE branches for a 3x3 symmetric game using continuation methods.
    
    In symmetric games, we look for symmetric equilibria where both players use the same
    mixed strategy. This simplifies the computation as we only need to track one strategy
    vector instead of two.
    
    Attributes:
        payoff_matrix: The 3x3 payoff matrix for the game
        tolerance: Numerical tolerance for convergence checks
    """
    
    # Algorithm parameters grouped by function
    
    # Continuation parameters
    DEFAULT_LAMBDA_FALLBACK = 1e-2      # λ value when centroid fails at λ=0
    LOOP_DETECTION_TOL = 1e-7           # Tolerance for detecting revisited points
    MAX_STEP_SIZE = 0.01                # Maximum continuation step size
    MIN_STEP_SIZE = 1e-8                # Minimum step size before giving up
    STEP_GROWTH_RATE = 1.1              # Step size growth rate
    STEP_REDUCTION_RATE = 0.5           # Step size reduction on failure
    
    # Newton-Raphson corrector parameters
    ARMIJO_MIN_STEP = 1e-3              # Minimum Armijo line search step
    ARMIJO_REDUCTION_RATE = 0.5         # Step reduction factor in line search
    
    # Nash equilibrium tracing parameters
    NASH_START_LAMBDA = 20.0            # Starting λ for Nash equilibrium search
    NASH_LAMBDA_INCREMENT = 5.0         # λ increment when projection fails
    NASH_MAX_ATTEMPTS = 6               # Max attempts to find branch from Nash
    
    # Branch validation parameters
    MIN_BRANCH_LENGTH = 10              # Minimum points for a significant branch
    
    # Bifurcation detection parameters
    SINGULAR_VALUE_THRESHOLD = 1e-2     # Threshold for detecting singular Jacobian
    DETERMINANT_THRESHOLD = 1e-8        # Threshold for determinant sign changes
    MIN_ANGLE_CHANGE = 0.1              # Minimum angle change for turning point (radians)
    LARGE_ANGLE_THRESHOLD = 0.5         # cos(60°) for large angle detection
    
    def __init__(self, payoff_matrix: np.ndarray, tolerance: float = 1e-12):
        """
        Initialize QRE continuation solver.
        
        Args:
            payoff_matrix: 3x3 payoff matrix for player 1 (player 2's payoffs are transpose)
            tolerance: Numerical tolerance for convergence checks
        """
        self.payoff_matrix = payoff_matrix
        self.tolerance = tolerance
        self.num_strategies = 3
        
    def expected_payoff(self, strategy: np.ndarray) -> np.ndarray:
        """
        Compute expected payoffs for each pure strategy.
        
        Args:
            strategy: Mixed strategy probability vector
            
        Returns:
            Expected payoff for each pure strategy
        """
        return self.payoff_matrix @ strategy
    
    def quantal_response(self, lambda_val: float, strategy: np.ndarray) -> np.ndarray:
        """
        Compute the quantal response for given lambda and mixed strategy.
        
        The quantal response function gives the probability of choosing each strategy
        based on exponential response to expected payoffs.
        
        Args:
            lambda_val: Rationality parameter (0 = random, ∞ = best response)
            strategy: Current mixed strategy
            
        Returns:
            Quantal response probabilities
        """
        if lambda_val == 0:
            return np.ones(self.num_strategies) / self.num_strategies
        
        payoffs = self.expected_payoff(strategy)
        # Prevent numerical overflow by subtracting max payoff
        max_payoff = np.max(payoffs)
        exp_values = np.exp(lambda_val * (payoffs - max_payoff))
        return exp_values / np.sum(exp_values)
    
    def residual_full(self, lambda_val: float, strategy: np.ndarray) -> np.ndarray:
        """
        Full residual F(λ, π) = π - QR(λ, π).
        
        The residual measures how far a strategy is from being a quantal response
        to itself (fixed point condition).
        """
        return strategy - self.quantal_response(lambda_val, strategy)
    
    def residual_reduced(self, lambda_val: float, strategy_reduced: np.ndarray) -> np.ndarray:
        """
        Reduced residual in (π₁, π₂) coordinates.
        
        Since π₃ = 1 - π₁ - π₂, we work in reduced 2D space.
        
        Args:
            lambda_val: Rationality parameter
            strategy_reduced: [π₁, π₂] reduced strategy vector
            
        Returns:
            2D residual vector
        """
        strategy_full = np.array([
            strategy_reduced[0], 
            strategy_reduced[1], 
            1 - strategy_reduced[0] - strategy_reduced[1]
        ])
        residual = self.residual_full(lambda_val, strategy_full)
        return residual[:2]  # Return only first 2 components
    
    def jacobian_analytical(self, lambda_val: float, strategy_reduced: np.ndarray) -> np.ndarray:
        """
        Compute analytical Jacobian of reduced residual.
        
        The Jacobian matrix contains partial derivatives of the residual function
        with respect to (λ, π₁, π₂). This is used for Newton's method and
        tangent computation.
        
        Args:
            lambda_val: Rationality parameter
            strategy_reduced: [π₁, π₂] reduced strategy vector
            
        Returns:
            2x3 Jacobian matrix
        """
        # Reconstruct full probability vector
        strategy_full = np.array([
            strategy_reduced[0], 
            strategy_reduced[1], 
            1 - strategy_reduced[0] - strategy_reduced[1]
        ])

        # Expected payoffs
        payoffs = self.payoff_matrix @ strategy_full
        
        if lambda_val == 0:
            # Special case: uniform distribution at λ = 0
            jacobian = np.zeros((2, 3))
            
            # ∂F_i/∂λ at λ=0
            payoff_mean = np.mean(payoffs)
            for i in range(2):
                jacobian[i, 0] = -(payoffs[i] - payoff_mean) / 3
            
            # ∂F_i/∂π_j at λ=0 (only diagonal terms)
            jacobian[0, 1] = 1.0  # ∂F₁/∂π₁
            jacobian[1, 2] = 1.0  # ∂F₂/∂π₂
            
            return jacobian
        
        # General case: λ > 0
        qr_probs = self.quantal_response(lambda_val, strategy_full)
        jacobian = np.zeros((2, 3))
        
        # Derivatives with respect to λ
        weighted_avg_payoff = np.sum(qr_probs * payoffs)
        for i in range(2):
            jacobian[i, 0] = -qr_probs[i] * (payoffs[i] - weighted_avg_payoff)
        
        # Derivatives with respect to π₁ and π₂
        for i in range(2):  # For F₁ and F₂
            for k in range(2):  # For π₁ and π₂
                # Direct contribution
                if i == k:
                    jacobian[i, k+1] = 1.0
                
                # Contribution through quantal response
                weighted_col_k = np.sum(qr_probs * self.payoff_matrix[:, k])
                weighted_col_2 = np.sum(qr_probs * self.payoff_matrix[:, 2])
                
                # Total derivative considering π₃ = 1 - π₁ - π₂
                jacobian[i, k+1] -= lambda_val * qr_probs[i] * (
                    self.payoff_matrix[i, k] - weighted_col_k - 
                    (self.payoff_matrix[i, 2] - weighted_col_2)
                )
        
        return jacobian
    
    def find_tangent(self, lambda_val: float, strategy_reduced: np.ndarray) -> np.ndarray:
        """
        Find unit tangent vector to the QRE curve.
        
        The tangent vector lies in the null space of the Jacobian and indicates
        the direction to continue tracing the branch.
        
        Args:
            lambda_val: Current λ value
            strategy_reduced: Current strategy in reduced coordinates
            
        Returns:
            3D unit tangent vector [dλ/ds, dπ₁/ds, dπ₂/ds]
            
        Raises:
            ValueError: If tangent vector has zero norm
        """
        jacobian = self.jacobian_analytical(lambda_val, strategy_reduced)
        _, _, vt = svd(jacobian)
        
        # The last row of Vt is the null vector (smallest singular value)
        tangent = vt[-1, :]
        
        # Normalize
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-14:
            raise ValueError(
                f"Zero tangent vector at λ={lambda_val}, x={strategy_reduced}. "
                "This may indicate a bifurcation point or numerical issues."
            )
            
        return tangent / tangent_norm
    
    def corrector_step(self, prediction: np.ndarray, previous: np.ndarray, 
                      previous_tangent: np.ndarray, step_size: float, 
                      max_iterations: int = 500) -> tuple:
        """
        Newton corrector step with Armijo line search.
        
        After the predictor step estimates the next point, the corrector refines it
        to lie exactly on the QRE manifold while maintaining the arclength constraint.
        
        Args:
            prediction: Predicted point from predictor step
            previous: Previous point on the branch
            previous_tangent: Tangent at previous point
            step_size: Desired arclength step
            max_iterations: Maximum Newton iterations
            
        Returns:
            (corrected_point, num_iterations) or (None, None) on failure
        """
        from numpy.linalg import LinAlgError
        
        current = prediction.copy()
        
        for iteration in range(max_iterations):
            lambda_val, strategy_reduced = current[0], current[1:3]
            
            # Compute residuals
            strategy_residual = self.residual_reduced(lambda_val, strategy_reduced)
            arclength_residual = previous_tangent @ (current - previous) - step_size
            
            # Scale arclength constraint for numerical stability
            scale = max(step_size, 1e-6)
            combined_residual = np.hstack((strategy_residual, arclength_residual / scale))
            
            # Check convergence
            if np.linalg.norm(combined_residual, np.inf) < self.tolerance:
                return current, iteration
            
            # Build augmented Jacobian
            jacobian = self.jacobian_analytical(lambda_val, strategy_reduced)
            jacobian_augmented = np.vstack((
                jacobian, 
                (previous_tangent / scale).reshape(1, -1)
            ))
            
            # Solve Newton system
            try:
                delta = solve(jacobian_augmented, -combined_residual)
            except LinAlgError:
                # Singular system - possible bifurcation
                return None, None
            except ValueError as e:
                raise ValueError(
                    f"Invalid values in corrector at λ={lambda_val}, x={strategy_reduced}. "
                    f"Error: {e}"
                )
            
            # Armijo line search
            line_search_step = 1.0
            while line_search_step > self.ARMIJO_MIN_STEP:
                candidate = current + line_search_step * delta
                candidate_lambda = candidate[0]
                candidate_strategy = candidate[1:3]
                
                # Check simplex constraints
                if (candidate_strategy[0] < -1e-12 or 
                    candidate_strategy[1] < -1e-12 or 
                    candidate_strategy.sum() > 1 + 1e-12):
                    line_search_step *= self.ARMIJO_REDUCTION_RATE
                    continue
                
                # Evaluate residual at candidate
                new_strategy_residual = self.residual_reduced(candidate_lambda, candidate_strategy)
                new_arclength_residual = previous_tangent @ (candidate - previous) - step_size
                new_combined_residual = np.hstack((
                    new_strategy_residual, 
                    new_arclength_residual / scale
                ))
                
                # Accept if improved
                if np.linalg.norm(new_combined_residual, np.inf) < np.linalg.norm(combined_residual, np.inf):
                    current = candidate
                    combined_residual = new_combined_residual
                    break
                    
                line_search_step *= self.ARMIJO_REDUCTION_RATE
            else:
                # Line search failed
                return None, None

            # Check for convergence based on step size
            if line_search_step * np.linalg.norm(delta) < self.tolerance:
                return current, iteration + 1
                
        # Return result even if not fully converged
        if np.linalg.norm(combined_residual, np.inf) < self.tolerance:
            return current, max_iterations
        else:
            return None, None
    
    def trace_branch(self, start_point: np.ndarray, start_tangent: np.ndarray, 
                    lambda_max: float = 50.0, initial_step: float = 0.001, 
                    max_steps: int = 20000,
                    branches: Optional[List[List[QREPoint]]] = None,
                    visited: Optional[Set[Tuple]] = None) -> List[QREPoint]:
        """
        Trace a QRE branch using predictor-corrector continuation.
        
        Starting from an initial point and tangent direction, this method traces
        the branch until reaching a boundary or completing a loop.
        
        Args:
            start_point: Starting point [λ, π₁, π₂]
            start_tangent: Initial tangent direction
            lambda_max: Maximum lambda value to trace
            initial_step: Initial step size
            max_steps: Maximum number of continuation steps
            branches: Existing branches (for bifurcation detection)
            visited: Set of visited branch keys
            
        Returns:
            List of QREPoints forming the branch
        """
        branch = []
        current_point = start_point.copy()
        current_tangent = start_tangent.copy()
        step_size = initial_step
        diagnostics = []  # Rolling buffer for bifurcation detection
        
        # Add initial point
        branch.append(self._make_qre_point(current_point))
        
        for step_num in range(max_steps):
            # Adaptive step size parameters
            max_step_local = self.MAX_STEP_SIZE
            
            # Predictor step
            predicted_point = current_point + step_size * current_tangent
            
            # Corrector step
            corrected_point, iterations = self.corrector_step(
                predicted_point, current_point, current_tangent, step_size
            )
            
            if corrected_point is None:
                # Failed to converge, reduce step size
                step_size *= self.STEP_REDUCTION_RATE
                if step_size < self.MIN_STEP_SIZE:
                    break
                continue
            
            # Check bounds
            lambda_val = corrected_point[0]
            strategy_reduced = corrected_point[1:3]
            
            if lambda_val < 0 or lambda_val > lambda_max:
                break
                
            # Check simplex constraints
            if (strategy_reduced[0] < -self.tolerance or 
                strategy_reduced[1] < -self.tolerance or 
                strategy_reduced[0] + strategy_reduced[1] > 1 + self.tolerance):
                break
            
            # Accept step
            branch.append(self._make_qre_point(corrected_point))
            
            # Update tangent
            new_tangent = self.find_tangent(corrected_point[0], corrected_point[1:3])
            
            # Ensure continuity of tangent direction
            if np.dot(current_tangent, new_tangent) < 0:
                new_tangent = -new_tangent
            
            # Adaptive step size based on tangent properties
            strategy_component_norm = np.linalg.norm(new_tangent[1:3])
            cos_angle = np.dot(current_tangent, new_tangent)
            curvature = 1 - abs(cos_angle)
            
            # Near Nash equilibria, strategy components become very small
            if strategy_component_norm < 1e-5:
                # Aggressive acceleration near Nash
                if strategy_component_norm > 1e-10:
                    max_step_local = min(1.0, 0.1 / strategy_component_norm)
                else:
                    max_step_local = 1.0
                growth_factor = 2.0
            elif strategy_component_norm < 1e-3:
                # Moderate acceleration
                max_step_local = 0.1
                growth_factor = 1.5
            else:
                # Normal region
                max_step_local = self.MAX_STEP_SIZE
                growth_factor = self.STEP_GROWTH_RATE
            
            # Apply curvature penalty
            if curvature > 0.3:
                step_size *= 0.5  # Sharp turn
            elif curvature > 0.1:
                step_size *= 0.8  # Moderate turn
            else:
                step_size *= growth_factor  # Straight or mild curve
            
            # Compute diagnostics for bifurcation detection
            full_jacobian = self.jacobian_analytical(corrected_point[0], corrected_point[1:3])
            reduced_jacobian = full_jacobian[:, 1:3]  # 2x2 Jacobian wrt strategies only
            
            # Check singularity of reduced Jacobian
            min_singular_value = np.linalg.svd(reduced_jacobian, compute_uv=False)[-1]
            determinant_sign = np.sign(np.linalg.det(reduced_jacobian))
            
            current_diagnostic = Diagnostic(
                point=corrected_point.copy(),
                tangent=new_tangent.copy(),
                sigma_min=min_singular_value,
                det_sign=determinant_sign,
                jacobian=full_jacobian.copy()
            )
            
            # Maintain rolling buffer of diagnostics
            diagnostics.append(current_diagnostic)
            if len(diagnostics) > 3:
                diagnostics.pop(0)
                
            # Update for next iteration
            current_point = corrected_point
            current_tangent = new_tangent
            
            # Apply step size limit
            step_size = min(max_step_local, step_size)
            
        return branch
    
    def _make_qre_point(self, point: np.ndarray) -> QREPoint:
        """Convert reduced coordinates to QREPoint."""
        lambda_val = point[0]
        strategy_reduced = point[1:3]
        strategy_full = np.array([
            strategy_reduced[0], 
            strategy_reduced[1], 
            1 - strategy_reduced[0] - strategy_reduced[1]
        ])
        return QREPoint(lambda_val, strategy_full)
    
    def _project_onto_manifold(self, point: np.ndarray, max_iterations: int = 30) -> Optional[np.ndarray]:
        """
        Project a point onto the QRE manifold.
        
        Given a point near the manifold, find the closest point that satisfies
        the QRE fixed-point condition by keeping λ fixed and adjusting the strategy.
        
        Args:
            point: Initial point [λ, π₁, π₂]
            max_iterations: Maximum Newton iterations
            
        Returns:
            Projected point or None if projection fails
        """
        projected = point.copy()
        lambda_val = projected[0]
        
        for _ in range(max_iterations):
            residual = self.residual_reduced(lambda_val, projected[1:3])
            if np.linalg.norm(residual) < self.tolerance:
                return projected
                
            # Jacobian with respect to strategy only
            jacobian = self.jacobian_analytical(lambda_val, projected[1:3])[:, 1:3]
            
            # Solve or use pseudo-inverse
            try:
                delta = solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(jacobian) @ (-residual)
                
            # Line search
            line_step = 1.0
            while line_step > self.ARMIJO_MIN_STEP:
                candidate = projected.copy()
                candidate[1:3] += line_step * delta
                
                # Project back into simplex if needed
                candidate[1:3] = np.clip(candidate[1:3], 0, 1)
                total = candidate[1] + candidate[2]
                if total > 1:
                    candidate[1:3] /= total
                
                new_residual = self.residual_reduced(lambda_val, candidate[1:3])
                if np.linalg.norm(new_residual) < np.linalg.norm(residual):
                    projected = candidate
                    residual = new_residual
                    break
                    
                line_step *= self.ARMIJO_REDUCTION_RATE
            else:
                return None
                
            if line_step * np.linalg.norm(delta) < self.tolerance:
                return projected
                
        return None
    
    def _hash_branch_key(self, point: np.ndarray, tangent: np.ndarray) -> tuple:
        """
        Create a hash key for branch tracking.
        
        The key includes the full strategy vector and the sign of dλ/ds to
        distinguish branches at the same point going in different λ directions.
        """
        strategy_full = np.array([
            point[1], 
            point[2], 
            1 - point[1] - point[2]
        ])
        return tuple(round(v, 6) for v in [point[0]] + list(strategy_full)) + (int(np.sign(tangent[0])),)
    
    def _is_duplicate_branch(self, new_branch: List[QREPoint], 
                            existing_branches: List[List[QREPoint]], 
                            endpoint_tolerance: float = 1e-1) -> bool:
        """
        Check if a branch duplicates an existing branch.
        
        Compares endpoints to detect branches traced in opposite directions
        or branches that are essentially the same.
        """
        if len(new_branch) < 2:
            return False
            
        new_start = new_branch[0].pi
        new_end = new_branch[-1].pi
        
        for existing in existing_branches:
            if len(existing) < 2:
                continue
                
            exist_start = existing[0].pi
            exist_end = existing[-1].pi
            
            # Check if endpoints match (same or opposite direction)
            if (np.linalg.norm(new_start - exist_start) < endpoint_tolerance and 
                np.linalg.norm(new_end - exist_end) < endpoint_tolerance):
                return True
                
            if (np.linalg.norm(new_start - exist_end) < endpoint_tolerance and 
                np.linalg.norm(new_end - exist_start) < endpoint_tolerance):
                return True
                
        return False
    
    def is_qre_limit_point(self, nash: np.ndarray, lambda_test: float = 50.0) -> bool:
        """
        Check if a Nash equilibrium is a QRE limit point.
        
        A Nash equilibrium is a QRE limit point if the quantal response
        at high lambda converges to the Nash strategy.
        
        Args:
            nash: Nash equilibrium strategy
            lambda_test: Lambda value for testing convergence
            
        Returns:
            True if Nash is a QRE limit point
        """
        qr = self.quantal_response(lambda_test, nash)
        return np.linalg.norm(nash - qr) < 1e-3
    
    def find_closest_nash(self, point: QREPoint, nash_list: List[np.ndarray]) -> tuple:
        """
        Find the Nash equilibrium closest to a given QRE point.
        
        Args:
            point: QRE point
            nash_list: List of Nash equilibria
            
        Returns:
            (index, distance) of closest Nash or (None, inf) if list is empty
        """
        if not nash_list:
            return None, float('inf')
        
        distances = [np.linalg.norm(point.pi - nash) for nash in nash_list]
        min_idx = np.argmin(distances)
        
        return min_idx, distances[min_idx]
    
    def find_all_branches(self, lambda_max: float = 50.0) -> List[List[QREPoint]]:
        """
        Find all QRE branches for the game.
        
        Algorithm:
        1. Find Nash equilibria and identify QRE limit points
        2. Trace principal branch from centroid (λ=0)
        3. Remove Nash closest to principal branch endpoint
        4. Trace branches from remaining QRE limit points
        
        Args:
            lambda_max: Maximum lambda value for tracing
            
        Returns:
            List of branches, each containing QRE points
        """
        branches = []
        visited = set()  # Track explored branch directions
        
        # Step 1: Get Nash equilibria and filter for QRE limit points
        nash_equilibria = find_nash_with_pygambit(self.payoff_matrix)
        qre_limit_points = [nash for nash in nash_equilibria if self.is_qre_limit_point(nash)]
        
        remaining_nash = qre_limit_points.copy()
        
        # Step 2: Trace from centroid (principal branch)
        centroid_point = np.array([0.0, 1/3, 1/3])
        
        # Handle potential singularity at λ=0
        try:
            centroid_tangent = self.find_tangent(centroid_point[0], centroid_point[1:3])
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            # Use small positive λ where system is better behaved
            centroid_point[0] = self.DEFAULT_LAMBDA_FALLBACK
            try:
                centroid_tangent = self.find_tangent(centroid_point[0], centroid_point[1:3])
            except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as e:
                raise ValueError(
                    f"Unable to compute tangent at centroid. "
                    f"This may indicate a degenerate game structure. Error: {e}"
                )
        
        # Ensure lambda is increasing for principal branch
        if centroid_tangent[0] < 0:
            centroid_tangent = -centroid_tangent
        
        # Trace principal branch
        branch_key = self._hash_branch_key(centroid_point, centroid_tangent)
        if branch_key not in visited:
            visited.add(branch_key)
            principal_branch = self.trace_branch(
                centroid_point, 
                centroid_tangent, 
                lambda_max,
                branches=branches,
                visited=visited
            )
            
            if len(principal_branch) > 1:
                branches.append(principal_branch)
                
                # Step 3: Remove Nash closest to principal branch endpoint
                endpoint = principal_branch[-1]
                idx, dist = self.find_closest_nash(endpoint, remaining_nash)
                if idx is not None and dist < 0.1:
                    remaining_nash.pop(idx)
        
        # Step 4: Trace from remaining Nash equilibria
        nash_attempts = 0
        while remaining_nash and nash_attempts < 10:  # Safety limit
            nash = remaining_nash.pop(0)
            nash_attempts += 1
            
            # Try different starting lambdas
            start_lambda = self.NASH_START_LAMBDA
            branch_found = False
            
            for attempt in range(self.NASH_MAX_ATTEMPTS):
                if branch_found:
                    break
                    
                nash_point = np.array([start_lambda, nash[0], nash[1]])
                
                # Project onto QRE manifold
                projected_point = self._project_onto_manifold(nash_point)
                if projected_point is None:
                    start_lambda += self.NASH_LAMBDA_INCREMENT
                    continue
                
                try:
                    # Find tangent at projected point
                    nash_tangent = self.find_tangent(projected_point[0], projected_point[1:3])
                    
                    # Trace toward λ=0
                    if nash_tangent[0] > 0:
                        nash_tangent = -nash_tangent
                    
                    # Check if direction already explored
                    branch_key = self._hash_branch_key(projected_point, nash_tangent)
                    if branch_key in visited:
                        # Try opposite direction
                        nash_tangent = -nash_tangent
                        branch_key = self._hash_branch_key(projected_point, nash_tangent)
                        if branch_key in visited:
                            start_lambda += self.NASH_LAMBDA_INCREMENT
                            continue
                    
                    visited.add(branch_key)
                    
                    # Trace branch with larger initial step
                    nash_branch = self.trace_branch(
                        projected_point,
                        nash_tangent,
                        lambda_max,
                        initial_step=0.1,  # Larger step for Nash branches
                        max_steps=5000,
                        branches=branches,
                        visited=visited
                    )
                    
                    if len(nash_branch) > self.MIN_BRANCH_LENGTH:
                        if not self._is_duplicate_branch(nash_branch, branches):
                            branches.append(nash_branch)
                            branch_found = True
                            
                            # Remove Nash closest to branch endpoint
                            endpoint = nash_branch[-1]
                            idx, dist = self.find_closest_nash(endpoint, remaining_nash)
                            if idx is not None and dist < 0.1:
                                remaining_nash.pop(idx)
                        
                except (np.linalg.LinAlgError, ValueError, ArithmeticError):
                    # Could not trace from this point
                    pass
                
                start_lambda += self.NASH_LAMBDA_INCREMENT

        return branches
    
    def find_qre_at_lambda(self, target_lambda: float, branches: List[List[QREPoint]]) -> List[QREPoint]:
        """
        Find all QRE solutions at a specific lambda value.
        
        Interpolates along branches to find points where lambda equals the target.
        
        Args:
            target_lambda: Lambda value to find QRE solutions at
            branches: List of QRE branches
            
        Returns:
            List of QRE points at the target lambda
        """
        qre_solutions = []
        
        for branch in branches:
            # Check for lambda crossings in branch
            for i in range(len(branch) - 1):
                lambda1 = branch[i].lambda_val
                lambda2 = branch[i + 1].lambda_val
                
                # Check if target_lambda is between these points
                if (lambda1 <= target_lambda <= lambda2) or (lambda2 <= target_lambda <= lambda1):
                    # Interpolate or project to find exact solution
                    if abs(lambda2 - lambda1) < 1e-6:
                        # Lambda values nearly identical
                        candidate_strategy = branch[i].pi
                    else:
                        # Linear interpolation
                        t = (target_lambda - lambda1) / (lambda2 - lambda1)
                        interpolated_strategy = (1 - t) * branch[i].pi + t * branch[i + 1].pi
                        
                        # Project onto manifold for accuracy
                        point = np.array([target_lambda, interpolated_strategy[0], interpolated_strategy[1]])
                        projected = self._project_onto_manifold(point)
                        
                        if projected is not None:
                            candidate_strategy = np.array([
                                projected[1], 
                                projected[2], 
                                1 - projected[1] - projected[2]
                            ])
                        else:
                            candidate_strategy = interpolated_strategy
                    
                    # Check if this is a new solution
                    is_new = True
                    for existing in qre_solutions:
                        if np.linalg.norm(existing.pi - candidate_strategy) < 1e-2:
                            is_new = False
                            break
                            
                    if is_new:
                        qre_solutions.append(QREPoint(target_lambda, candidate_strategy))
                        
        return qre_solutions