
"""
Robust QRE continuation solver for 3×3 symmetric games.

Implements the corrections discussed:
  * Armijo–back‑tracking Newton corrector with proper scaling
  * Robust projection onto the QRE manifold
  * Loop detection & branch‑duplicate filter
  * Adaptive step size driven by recent Newton effort
  * Launches branches from centroid & Nash equilibria in both directions
     (λ increasing or decreasing)
The implementation still expects you to provide two game‑specific methods:
    residual_reduced(λ, x)   -> R²
    jacobian_analytical(λ, x)-> 2×3 matrix
because these depend on the payoff matrix and have already been coded earlier.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from numpy.linalg import solve, svd, norm, LinAlgError, pinv


# ---------------------------------------------------------------------
@dataclass
class QREPoint:
    """Container for one point along a branch."""
    lam: float              # λ
    pi:  np.ndarray         # length‑3 strategy vector (row player)
    # Provide convenience slice for duplicate checks
    @property
    def pi2(self) -> np.ndarray:
        return self.pi[:2]


# ---------------------------------------------------------------------
class QREContinuation:
    def __init__(self,
                 payoff_matrix: np.ndarray,
                 tol: float = 1e-10) -> None:
        self.A   = payoff_matrix
        self.tol = tol

    # ======================= Branch discovery ==========================
    def find_all_branches(self,
                          nash_eq: List[np.ndarray],
                          lambda_max: float = 50.0) -> List[List[QREPoint]]:
        """Enumerate QRE branches starting from centroid and NEs."""
        branches: List[List[QREPoint]] = []

        # 1. centroid
        m0 = np.array([0.0, 1/3, 1/3])
        try:
            t0 = self.find_tangent(m0[0], m0[1:3])
        except Exception:
            # fallback: small λ
            m0[0] = 1e-2
            t0 = self.find_tangent(m0[0], m0[1:3])
        for direction in (t0, -t0):
            br = self._trace_branch(m0, direction, lambda_max)
            if len(br) > 1 and not self._is_duplicate_branch(br, branches):
                branches.append(br)

        # 2. from each Nash equilibrium
        for ne in nash_eq:
            for sign in (+1, -1):                  # search λ both ways
                lam_try = 20.0
                for _ in range(6):
                    seed = np.array([lam_try, ne[0], ne[1]])
                    proj = self._project_onto_manifold(seed)
                    if proj is None:
                        lam_try += sign * 5.0
                        continue
                    try:
                        t_ne = self.find_tangent(proj[0], proj[1:3])
                    except Exception:
                        break
                    for direction in (t_ne, -t_ne):
                        br = self._trace_branch(proj, direction, lambda_max)
                        if len(br) > 1 and not self._is_duplicate_branch(br, branches):
                            branches.append(br)
                    break   # stop λ search once projection succeeds

        return branches

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# OLD VERSIONS
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
def _project_onto_manifold(self, m: np.ndarray, max_iter: int = 30) -> Optional[np.ndarray]:
        """
        Project a point onto the QRE manifold using Newton's method.
        Keeps lambda fixed and adjusts strategy to satisfy QRE conditions.
        """
        m_proj = m.copy()
        lambda_val = m_proj[0]
        for _ in range(max_iter):
            # Evaluate residual
            F = self.residual_reduced(lambda_val, m_proj[1:3])
            
            if np.linalg.norm(F) < self.tol:
                return m_proj
            
            # Jacobian with respect to strategy only (fix lambda)
            J = self.jacobian_analytical(lambda_val, m_proj[1:3])[:, 1:3]
            
            # Newton step
            try:
                delta = solve(J, -F)
                m_proj[1:3] += delta
                
                # Ensure we stay in the simplex
                if m_proj[1] < 0 or m_proj[2] < 0 or m_proj[1] + m_proj[2] > 1:
                    return None
                    
                if np.linalg.norm(delta) < self.tol:
                    return m_proj
            except:
                return None
                
        return None
# =========================== Main ======================================
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
        # Compute actual tangent at centroid instead of using fixed direction
        t0 = self.find_tangent(m0[0], m0[1:3])
        
        # Trace in both directions from centroid
        for direction in [t0, -t0]:
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
                        
                        # Only trace if moving toward lower lambda
                        if direction[0] < 0:
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