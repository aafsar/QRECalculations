
# Code Review and Improvement Plan  
**Modules reviewed:** `qre_continuation.py`, `nash_pygambit.py`  
**Scope:** 3 × 3 symmetric 2‑player normal–form games  
**Author of review:** ChatGPT (o3) — 22 Jun 2025  

---

## 1 Algorithmic issues & fixes  

| # | Problem | Impact | Patch (snippet) |
|---|---------|--------|-----------------|
|1|**Newton sign error** in `corrector_step` | Slow or stalled convergence |```python
delta = solve(J_aug, -R)
m_new += delta
```|
|2|λ = 0 **Jacobian cross‑terms omitted** | Principal branch departs centroid with zig‑zag | add<br>```python
J[0,2] = -1.0;  J[1,1] = -1.0
```|
|3|Null‑space **direction ambiguity** | Branch flips at turning points |```python
from scipy.linalg import null_space
tangent = null_space(J)[:,0]
if tangent[0] < 0: tangent *= -1
```|
|4|Seeding branch **off manifold** | Many starts from NE fail | Project seed with Newton before calling `find_tangent` (see helper below). |
|5|Duplicate & signature tolerance too loose | Distinct equilibria merged | tighten tol to `1e‑6`; include λ in `_branch_signature`. |
|6|Minute probabilities retained from Gambit | Phantom pure branches | Zero‑out entries `<1e‑10` and renormalise. |

### Helper: seed projection
```python
def newton_project(m, fun, jac, tol=1e-12, max_it=30):
    for _ in range(max_it):
        F = fun(m[0], m[1:3])
        if np.linalg.norm(F) < tol: return m
        J = jac(m[0], m[1:3])[:,1:]  # fix λ
        m[1:3] += solve(J, -F)
    return None
```

---

## 2 Suggested defensive programming additions  

* **Runtime simplex check** (already present, but keep `tol=1e‑12`).  
* **Adaptive step controller**: decrease `h` on any corrector failure; increase slowly on success.  
* **Logging hooks** (`logging` module) instead of silent `continue` for easier debugging.  

---

## 3 Empirical validation plan  

1. **Regression set** of 10 textbook 3×3 symmetric games (e.g. Matching Pennies variant, Rock–Paper–Scissors with bonuses, 3‑action coordination).  
2. Verify that:  
   * principal branch connects centroid to each (and only each) **stable** NE;  
   * corrected code finds identical QRE sets as Govindan–Wilson (2009) homotopy solver.  
3. Monte‑carlo stress‑test over 1 000 random payoff draws ∼ U[0,100].  

Expected result after patches: > 98 % of games yield all theoretical branches, with average branch‑trace time < 0.2 s on M3‑Max.

