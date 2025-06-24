# Algorithmic Issues & Detailed Fixes for QRE Continuation Code  
*Modules concerned: `qre_continuation.py`, `nash_pygambit.py`*  
*Author: ChatGPT (o3) – 22 Jun 2025*  

---

## Overview  

The continuation pipeline aims to trace all logit‐quantal–response equilibria (QRE) branches of a **3 × 3 symmetric two‑player game** as the precision parameter \(\lambda\) increases from 0 (centroid) to large values (near Nash equilibria).  
During review we identified **six algorithmic flaws** that can prevent convergence, lose branches, or merge distinct equilibria.  
For each flaw we explain:

1. **What exactly is wrong.**  
2. **Why it breaks the algorithm.**  
3. **How to diagnose it in runtime output.**  
4. **Concrete patch** – a drop‑in code snippet or refactor note.

---

## 3 Null‑Space Direction Ambiguity  

| | |
|-|-|
| **Location** | `find_tangent()` – tangent vector from SVD of Jacobian |
| **Bug** | Uses `Vt[-1]` without consistency check; SVD can flip sign (and occasionally pick another null vector if rank > 1). |
| **Symptom** | Branch direction suddenly reverses near fold points; produces duplicated points or overlaps in λ‑ordered list. |
| **Fix** | Compute orthonormal null‑space, choose first vector, and align with previous tangent: |
| **Patch** |
```python
from scipy.linalg import null_space
nullvec = null_space(J)
τ = nullvec[:, 0]
if τ[0] < 0:   # maintain orientation so λ increases
    τ *= -1
τ /= np.linalg.norm(τ)
return τ
```
|
| **Reasoning** | Continuation requires a *continuous* tangent orientation to predict the next point; sign flips violate the predictor assumption. |

---

**End of document**  
