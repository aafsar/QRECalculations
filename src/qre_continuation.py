"""
Continuation method for finding all QRE branches in 3x3 symmetric games.
Based on pseudo-arclength predictor-corrector algorithm.
"""

import numpy as np
from scipy.linalg import solve, svd
from dataclasses import dataclass
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class QREPoint:
    """Represents a point on a QRE branch."""
    lambda_val: float
    pi1: np.ndarray  # Player 1 mixed strategy
    pi2: np.ndarray  # Player 2 mixed strategy
    
    
class QREContinuation:
    """
    Computes all QRE branches for a 3x3 symmetric game using continuation methods.
    """
    
    def __init__(self, payoff_matrix: np.ndarray, tol: float = 1e-12):
        """
        Initialize with a 3x3 payoff matrix for player 1.
        Player 2's payoffs are the transpose (symmetric game).
        """
        self.A = payoff_matrix
        self.tol = tol
        self.n = 3  # Number of strategies
        
    def expected_payoff(self, pi: np.ndarray) -> np.ndarray:
        """Compute expected payoffs for each strategy."""
        return self.A @ pi
    
    def quantal_response(self, lambda_val: float, pi: np.ndarray) -> np.ndarray:
        """Compute the quantal response for given lambda and mixed strategy."""
        if lambda_val == 0:
            return np.ones(self.n) / self.n
        
        u = self.expected_payoff(pi)
        # Prevent overflow
        max_u = np.max(u)
        exp_vals = np.exp(lambda_val * (u - max_u))
        return exp_vals / np.sum(exp_vals)
    
    def residual_full(self, lambda_val: float, pi: np.ndarray) -> np.ndarray:
        """Full residual F(λ, π) = π - QR(λ, π)"""
        return pi - self.quantal_response(lambda_val, pi)
    
    def residual_reduced(self, lambda_val: float, x: np.ndarray) -> np.ndarray:
        """
        Reduced residual in (π1, π2) coordinates.
        x = [π1, π2], π3 = 1 - π1 - π2
        """
        pi = np.array([x[0], x[1], 1 - x[0] - x[1]])
        res_full = self.residual_full(lambda_val, pi)
        return res_full[:2]  # Return only first 2 components
    
    def jacobian_analytical(self, lambda_val: float, x: np.ndarray) -> np.ndarray:
        """
        Compute analytical Jacobian of reduced residual with respect to (λ, π1, π2).
        
        For a symmetric game, the residual is:
        F_i(λ, π) = π_i - exp(λ * u_i) / Σ_j exp(λ * u_j)
        
        where u_i = (A @ π)_i is the expected payoff for strategy i.
        
        The Jacobian has components:
        - ∂F_i/∂λ: derivative with respect to lambda
        - ∂F_i/∂π_j: derivatives with respect to strategy probabilities
        
        Returns 2x3 matrix for the reduced system.
        """
        # Reconstruct full probability vector
        pi = np.array([x[0], x[1], 1 - x[0] - x[1]])
        
        # Expected payoffs
        u = self.A @ pi
        
        if lambda_val == 0:
            # Special case: when λ = 0, QR is uniform distribution
            J = np.zeros((2, 3))
            
            # ∂F_i/∂λ at λ=0
            # QR = [1/3, 1/3, 1/3], so derivative involves u_i - mean(u)
            u_mean = np.mean(u)
            for i in range(2):
                J[i, 0] = -(u[i] - u_mean) / 3
            
            # ∂F_i/∂π_j at λ=0
            # Since QR doesn't depend on π when λ=0, only diagonal terms remain
            J[0, 1] = 1.0  # ∂F_1/∂π_1
            J[1, 2] = 1.0  # ∂F_2/∂π_2
            
            return J
        
        # General case: λ > 0
        # Compute exponentials with numerical stability
        max_u = np.max(u)
        exp_vals = np.exp(lambda_val * (u - max_u))
        sum_exp = np.sum(exp_vals)
        
        # Quantal response probabilities
        qr = exp_vals / sum_exp
        
        # Initialize Jacobian
        J = np.zeros((2, 3))
        
        # 1. Derivatives with respect to λ
        # ∂F_i/∂λ = -∂QR_i/∂λ = -QR_i * (u_i - Σ_j QR_j * u_j)
        weighted_avg_u = np.sum(qr * u)
        for i in range(2):
            J[i, 0] = -qr[i] * (u[i] - weighted_avg_u)
        
        # 2. Derivatives with respect to π_1 and π_2
        # For the reduced system, we need derivatives with respect to x = [π_1, π_2]
        # Note: π_3 = 1 - π_1 - π_2, so ∂π_3/∂π_1 = -1, ∂π_3/∂π_2 = -1
        
        # First compute ∂F_i/∂π_k for the full system
        # ∂F_i/∂π_k = δ_ik - ∂QR_i/∂π_k
        # where ∂QR_i/∂π_k = λ * QR_i * (A_ik - Σ_j QR_j * A_jk)
        
        for i in range(2):  # For F_1 and F_2
            for k in range(2):  # For π_1 and π_2
                # Direct contribution from π_k
                if i == k:
                    J[i, k+1] = 1.0
                
                # Contribution through QR
                # ∂QR_i/∂π_k considering chain rule for π_3
                weighted_A_k = np.sum(qr * self.A[:, k])
                weighted_A_2 = np.sum(qr * self.A[:, 2])  # For π_3 = 1 - π_1 - π_2
                
                # Total derivative considering π_3 dependency
                J[i, k+1] -= lambda_val * qr[i] * (self.A[i, k] - weighted_A_k - 
                                                   (self.A[i, 2] - weighted_A_2))
        
        return J
    
    def find_tangent(self, lambda_val: float, x: np.ndarray) -> np.ndarray:
        """
        Find unit tangent vector to the curve at (λ, x).
        Returns 3D vector [t_λ, t_x1, t_x2].
        """
        # Find null space of J using SVD (use analytical Jacobian)
        J = self.jacobian_analytical(lambda_val, x)
        _, _, Vt = svd(J)
        # The last column of V (last row of Vt) is the null vector
        tangent = Vt[-1, :]
        
        # Normalize and ensure positive lambda direction initially
        if tangent[0] < 0:
            tangent = -tangent
            
        return tangent / np.linalg.norm(tangent)
    
    def corrector_step(self, m_pred: np.ndarray, m_prev: np.ndarray, 
                      t_prev: np.ndarray, h: float, max_iter: int = 500) -> tuple:
        """
        Newton corrector for the augmented system with adaptive damping.
        Returns (m_new, newton_iters) or (None, None) on failure.
        """
        from numpy.linalg import LinAlgError
        
        m_new = m_pred.copy()
        
        for it in range(max_iter):
            lam, x = m_new[0], m_new[1:3]
            F = self.residual_reduced(lam, x)           # shape (2,)
            c = t_prev @ (m_new - m_prev) - h           # scalar
            # Prevent division by very small h
            scale = max(h, 1e-10)
            R = np.hstack((F, c / scale))
            
            if np.linalg.norm(R, np.inf) < self.tol:
                return m_new, it                         # converged
            
            J = self.jacobian_analytical(lam, x)         # (2×3)
            J_aug = np.vstack((J, t_prev.reshape(1, -1)))  # (3×3)
            
            try:
                delta = solve(J_aug, -R)                 # Newton equation: J @ delta = -R
            # except (LinAlgError, ValueError):
            except LinAlgError:
                return None, None                        # singular => possible bifurcation
            
            step = 1.0
            while step > 1e-3:                        # Armijo loop
                cand = m_new + step * delta
                lam_c, x_c = cand[0], cand[1:3]
                # stay in simplex
                if (x_c[0] < -1e-12 or x_c[1] < -1e-12 or x_c.sum() > 1 + 1e-12):
                    step *= 0.5
                    continue
                F_c = self.residual_reduced(lam_c, x_c)
                c_c = t_prev @ (cand - m_prev) - h
                R_c = np.hstack((F_c, c_c / scale))
                if np.linalg.norm(R_c, np.inf) < np.linalg.norm(R, np.inf):
                    m_new, R = cand, R_c             # accept & refresh
                    break
                step *= 0.5
            else:
                return None, None

            if step * np.linalg.norm(delta) < self.tol:  # scaled step size (C-3)
                return m_new, it + 1
        return (m_new, max_iter) if np.linalg.norm(R, np.inf) < self.tol else (None, None)
    
    def trace_branch(self, m0: np.ndarray, t0: np.ndarray, 
                    lambda_max: float = 50.0, h0: float = 0.005, 
                    max_steps: int = 20000) -> List[QREPoint]:
        """
        Trace a single branch starting from m0 with initial tangent t0.
        """
        # Constants for boundary and loop detection
        LOOP_TOL = 1e-7  # loop detection tolerance  
        MAX_H = 0.05      # absolute cap on step length

        branch = []
        m = m0.copy()
        t = t0.copy()
        h = h0
        newton_history = []
        
        # Add initial point
        branch.append(self._make_qre_point(m))
        
        for _ in range(max_steps):
            # Predictor
            m_pred = m + h * t
            
            # Corrector
            m_new, nit = self.corrector_step(m_pred, m, t, h)
            
            if m_new is None:
                # Failed to converge, reduce step size
                h *= 0.5
                if h < 1e-8:
                    break
                continue
            
            # Check bounds
            lambda_val = m_new[0]
            x = m_new[1:3]
            
            if lambda_val < 0 or lambda_val > lambda_max:
                break
                
            # Check if we're still in the simplex
            if x[0] < -self.tol or x[1] < -self.tol or x[0] + x[1] > 1 + self.tol:
                break

            # Reconstruct full probability vector
            pi1 = np.array([m_new[1], m_new[2], 1 - m_new[1] - m_new[2]])
            
            # Loop detection (T-4)
            if any(np.linalg.norm(pi1 - q.pi1, np.inf) < LOOP_TOL for q in branch):
                break
            
            # Accept step
            branch.append(self._make_qre_point(m_new))
            
            # Update tangent
            t_new = self.find_tangent(m_new[0], m_new[1:3])
            
            # Ensure continuity of tangent direction
            if np.dot(t, t_new) < 0:
                t_new = -t_new
                
            m = m_new
            t = t_new
            
            # Adaptive step size (more conservative)
            # TODO: Can there be an adaptive step size? Perhaps check what Bland & Turocy (2024) do.
            h = min(0.05, h * 1.1)
        return branch
    
    def _make_qre_point(self, m: np.ndarray) -> QREPoint:
        """Convert reduced coordinates to QREPoint."""
        lambda_val = m[0]
        x = m[1:3]
        pi = np.array([x[0], x[1], 1 - x[0] - x[1]])
        return QREPoint(lambda_val, pi, pi)  # Symmetric game
    
    def _is_on_branch(self, point: np.ndarray, branch: List[QREPoint], tol: float = 1e-4) -> bool:
        """Check if a point is already on a branch."""
        for qre_point in branch:
            if np.linalg.norm(qre_point.pi1[:2] - point[1:3]) < tol:
                return True
        return False
    
    def _project_onto_manifold(self, m: np.ndarray, max_iter: int = 30) -> Optional[np.ndarray]:
        """
        Project a point onto the QRE manifold using Newton's method.
        Keeps lambda fixed and adjusts strategy to satisfy QRE conditions.
        """
        m_proj = m.copy()
        
        lambda_val = m_proj[0]
        for _ in range(max_iter):
            F = self.residual_reduced(lambda_val, m_proj[1:3])
            if np.linalg.norm(F) < self.tol:
                return m_proj
            J = self.jacobian_analytical(lambda_val, m_proj[1:3])[:, 1:3]
            # solve or pseudo‑inverse
            try:
                delta = solve(J, -F)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(J) @ (-F)
            step = 1.0
            while step > 1e-3:
                cand = m_proj.copy()
                cand[1:3] += step * delta
                if (cand[1] < 0) or (cand[2] < 0) or (cand[1]+cand[2] > 1):
                    step *= 0.5
                    continue
                F_c = self.residual_reduced(lambda_val, cand[1:3])
                if np.linalg.norm(F_c) < np.linalg.norm(F):
                    m_proj = cand
                    F = F_c
                    break
                step *= 0.5
            else:
                return None
            if step * np.linalg.norm(delta) < self.tol:
                return m_proj
                
        return None
    
    
    def _is_duplicate_branch(self, new_branch: List[QREPoint], 
                            existing_branches: List[List[QREPoint]], 
                            endpoint_tol: float = 1e-1) -> bool:
        """
        Check if a branch is a duplicate of any existing branch by comparing endpoints.
        Checks all four combinations: start-start, start-end, end-start, end-end.
        """
        if len(new_branch) < 2:
            return False
            
        new_start = new_branch[0].pi1
        new_end = new_branch[-1].pi1
        
        for existing in existing_branches:
            if len(existing) < 2:
                continue
                
            exist_start = existing[0].pi1
            exist_end = existing[-1].pi1
            
            # Check all four combinations
            if (np.linalg.norm(new_start - exist_start) < endpoint_tol and 
                np.linalg.norm(new_end - exist_end) < endpoint_tol):
                return True  # Same direction
                
            if (np.linalg.norm(new_start - exist_end) < endpoint_tol and 
                np.linalg.norm(new_end - exist_start) < endpoint_tol):
                return True  # Opposite direction
                
        return False
    
    def find_all_branches(self, nash_equilibria: List[np.ndarray], 
                         lambda_max: float = 50.0) -> List[List[QREPoint]]:
        """
        Find all QRE branches by:
        1. Starting from centroid (principal branch)
        2. Starting from each Nash equilibrium
        3. Detecting and following bifurcations
        """
        branches = []
        
        # 1. Principal branch from centroid
        m0 = np.array([0.0, 1/3, 1/3])
        try:
            t0 = self.find_tangent(m0[0], m0[1:3])
        except Exception:
            # fallback: small λ
            m0[0] = 1e-2
            t0 = self.find_tangent(m0[0], m0[1:3])
        
        for direction in (t0, -t0):
            branch = self.trace_branch(m0, direction, lambda_max)
            if len(branch) > 1 and not self._is_duplicate_branch(branch, branches):
                branches.append(branch)
        
        # 2. Branches from Nash equilibria
        
        for nash in nash_equilibria:
            # Start from high lambda near Nash, but adaptively try higher values if needed
            start_lambda = 20.0
            max_attempts = 3
            
            for _ in range(max_attempts):
                m_nash = np.array([start_lambda, nash[0], nash[1]])
                
                # Project onto QRE manifold before computing tangent
                m_nash_proj = self._project_onto_manifold(m_nash)
                if m_nash_proj is None:
                    start_lambda += 10.0
                    continue
                
                # Try both tangent directions
                try:
                    t_nash = self.find_tangent(m_nash_proj[0], m_nash_proj[1:3])
                    branch_found = False
                    
                    for direction in [t_nash, -t_nash]:
                        branch = self.trace_branch(m_nash_proj, direction, lambda_max)
                        if len(branch) > 10:  # Significant branch
                            # Check if this is a duplicate
                            if not self._is_duplicate_branch(branch, branches):
                                branches.append(branch)
                                branch_found = True
                    
                    # If we found a branch, no need to try higher lambda values
                    if branch_found:
                        break
                        
                except:
                    pass
                
                # Try higher lambda for next attempt
                start_lambda += 10.0
                    
        return branches
    
    def find_qre_at_lambda(self, target_lambda: float, branches: List[List[QREPoint]]) -> List[QREPoint]:
        """
        Find all QRE solutions at a specific lambda value by interpolating branches.
        """
        qre_solutions = []
        
        for branch in branches:
            # Find points that bracket target_lambda
            for i in range(len(branch) - 1):
                lambda1 = branch[i].lambda_val
                lambda2 = branch[i + 1].lambda_val
                
                # Check if target_lambda is between these points
                if (lambda1 <= target_lambda <= lambda2) or (lambda2 <= target_lambda <= lambda1):
                    # Linear interpolation
                    t = (target_lambda - lambda1) / (lambda2 - lambda1)
                    pi1_interp = (1 - t) * branch[i].pi1 + t * branch[i + 1].pi1
                    pi2_interp = (1 - t) * branch[i].pi2 + t * branch[i + 1].pi2

                    # TODO: Would it make sense to project this into QRE manifold?
                    
                    # Check if this is a new solution
                    is_new = True
                    for sol in qre_solutions:
                        if np.linalg.norm(sol.pi1 - pi1_interp) < 1e-1:
                            is_new = False
                            break
                            
                    if is_new:
                        qre_solutions.append(QREPoint(target_lambda, pi1_interp, pi2_interp))
                        
        return qre_solutions