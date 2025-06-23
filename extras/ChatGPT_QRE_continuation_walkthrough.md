# Implementation Walkthrough: Continuation Method for QRE in 3×3 Symmetric 2‑Player Games

## 1  Brief summary

For a symmetric $3\times3$ two‑player normal‑form game every Logit‑Quantal‑Response‑Equilibrium (LQRE) is a point $(\lambda,\boldsymbol\pi)\in\mathbb R_{+}\times\Delta^{2}$ satisfying

\[
\pi_i\;=\;\frac{\exp\bigl(\lambda\,u_i(\boldsymbol\pi)\bigr)}
                {\sum_{j=1}^3 \exp\bigl(\lambda\,u_j(\boldsymbol\pi)\bigr)},
\qquad i=1,2,3.
\]

As $\lambda$ rises from $0$ to $+\infty$, the solution set forms one‑dimensional **continuation curves** (“branches”) that may turn, merge, or connect distinct Nash equilibria.  
A **pseudo‑arclength predictor–corrector continuation** algorithm reliably traces *all* branches:

1. **Seed** at $(\lambda_0,\boldsymbol\pi_0)=(0,(\tfrac13,\tfrac13,\tfrac13))$ (principal curve) *and* at each Nash equilibrium.  
2. **Predictor**: step a small arclength $h$ along the tangent.  
3. **Corrector**: Newton iterations on an augmented system that enforces both the QRE fixed‑point conditions and the arclength constraint, so turning points pose no problem.  
4. **Branch‑switching**: detect bifurcations and launch continuation in the opposite tangent direction to cover all arms.

The remainder of the document gives engineering‑level details, Julia code skeletons, test games, and a task breakdown.

---

## 2  Mathematical set‑up

### 2.1  Game representation
* Let $A\in\mathbb R^{3\times3}$ be the row‑player payoff matrix; symmetry implies the column‑player payoff is $A^{\top}$.
* A mixed strategy is $\boldsymbol\pi=(\pi_1,\pi_2,\pi_3)\in\Delta^{2}:=\{\pi_i\ge0,\;\sum_i\pi_i=1\}$.

### 2.2  Logit quantal‑response map
\[
QR_i(\lambda,\boldsymbol\pi)=
\frac{\exp\!\bigl(\lambda\,[A\boldsymbol\pi]_i\bigr)}
     {\sum_{j=1}^3 \exp\!\bigl(\lambda\,[A\boldsymbol\pi]_j\bigr)}.
\]
Define residual
\[
F(\lambda,\boldsymbol\pi)=\boldsymbol\pi-QR(\lambda,\boldsymbol\pi),
\quad g(\boldsymbol\pi)=\mathbf1^{\top}\boldsymbol\pi-1.
\]

### 2.3  Reduced (2‑D) simplex coordinates  
Eliminate the simplex constraint via
\[
\mathbf x=(\pi_1,\pi_2)^{\top},\qquad \pi_3=1-\pi_1-\pi_2,
\]
so the reduced residual is $F_r(\lambda,\mathbf x)\in\mathbb R^{2}$.

### 2.4  Augmented pseudo‑arclength system  
Given previous continuation point $\mathbf m_k=(\lambda_k,\mathbf x_k)$ and unit tangent $\mathbf t_{\!k}$, solve
\[
\begin{cases}
F_r(\lambda,\mathbf x)=\mathbf0 & (2\text{ eqs})\\[4pt]
\mathbf t_{\!k}^{\top}(\mathbf m-\mathbf m_k)-h=0 & (1\text{ eq})
\end{cases}
\tag{\*}
\]
for $\mathbf m=(\lambda,\mathbf x)\in\mathbb R^{3}$.

---

## 3  Numerical algorithm (predictor–corrector)

| Phase | Goal | Key operations |
|-------|------|----------------|
| **Tangent** | Find null‑vector of Jacobian: $J\,\mathbf t=\mathbf0$, $\mathbf e_\lambda^{\top}\mathbf t=1$. Solve via SVD or $QR$. |
| **Predictor** | $\mathbf m^{\text{pred}}=\mathbf m_k+h\,\mathbf t$. |
| **Corrector** | Newton on system (\*). Augmented Jacobian  

$\displaystyle\tilde J=\begin{bmatrix}J\\ \mathbf t_{\!k}^{\top}\end{bmatrix}$  

(3×3) ⇒ cheap LU factorisation. |
| **Acceptance** | Tolerances: $\|F_r\|_\infty<10^{-12}$ and Newton step $<10^{-10}$. Adapt $h\!\to\!1.2h$ on fast convergence, else $0.8h$. |
| **Termination** | Stop branch when (i) $\lambda>\lambda_{\max}$ (e.g. 50), or (ii) $\boldsymbol\pi$ within $10^{-10}$ of a Nash equilibrium, or (iii) arclength budget exhausted. |

---

## 4  Implementation guide in *Julia*

### 4.1  Core functions

```julia
using LinearAlgebra, StaticArrays, ForwardDiff

struct SymGame3x3
    A::SMatrix{3,3,Float64,9}
end

# Expected payoffs
exp_payoff(g::SymGame3x3, π::SVector{3,Float64}) = g.A * π

# Reduced residual F_r
function F_r(g::SymGame3x3, λ::Float64, x::SVector{2,Float64})
    π = SVector(x[1], x[2], 1 - sum(x))
    u = exp_payoff(g, π)
    z = @. exp(λ*u)
    qr = z / sum(z)
    return π[1:2] - qr[1:2]
end

# Wrapper for AD
Fr_wrap(m::SVector{3,Float64}, g) = F_r(g, m[1], m[2:3])
```

### 4.2  Continuation driver (skeletal)

```julia
function continue_QRE(g::SymGame3x3; λmax=50.0, h0=1e-2, maxsteps=10_000)
    m  = @SVector [0.0, 1/3, 1/3]   # (λ, π1, π2)
    t  = @SVector [1.0, 0.0, 0.0]   # initial tangent along λ
    h  = h0
    branch = [m]

    for _ in 1:maxsteps
        m_pred = m + h*t            # predictor

        # corrector
        m_new = m_pred
        for _ in 1:20
            F  = Fr_wrap(m_new, g)
            c  = dot(t, m_new - m) - h
            R  = @SVector [F[1], F[2], c]
            if maximum(abs, R) < 1e-12; break end

            J  = ForwardDiff.jacobian(z -> Fr_wrap(z, g), m_new)
            Jaug = [J; transpose(t)]
            Δm  = Jaug \ R
            m_new -= Δm
            if norm(Δm) < 1e-10; break end
        end

        λ = m_new[1]
        if λ > λmax; break end
        push!(branch, m_new)

        # new tangent
        J = ForwardDiff.jacobian(z -> Fr_wrap(z, g), m_new)
        tλ = 1.0
        tv = -(J[:, 2:3] \ (J[:, 1] * tλ))
        t  = @SVector [tλ, tv[1], tv[2]] / norm(@SVector [tλ, tv[1], tv[2]])

        m = m_new
        h = min(0.1, 1.2h)
    end
    return branch    # vector of (λ, π1, π2)
end
```

### 4.3  Bifurcation & branch enumeration
1. **Detect** turning points by sign change in $t_\lambda$ or singularity of $J$.
2. **Switch**: push $(\mathbf m, -\mathbf t)$ onto a queue of unexplored directions.
3. **Loop** breadth‑first until queue empty.  
A symmetric $3\times3$ game has at most **7** Nash equilibria, so enumeration is fast.

---

## 5  Verification suite

| Test game | Payoff matrix $A$ | Expected branch picture |
|-----------|------------------|-------------------------|
| Matching pennies‑style | $\begin{pmatrix}75&25&25\\25&75&25\\25&25&75\end{pmatrix}$ | Single principal curve. |
| Pure‑coordination | $\operatorname{diag}(90,90,90)$ | Three pure‑Nash branches. |
| Rock–paper–scissors | $\begin{pmatrix}0&60&30\\30&0&60\\60&30&0\end{pmatrix}$ | Flat line $\pi=(\tfrac13,\tfrac13,\tfrac13)$ for all $\lambda$. |

Check that residuals $\|F_r\|<10^{-10}$ and that limits approach Nash equilibria.

---

## 6  Extensions & practical advice
* **Analytic Jacobian** cuts ForwardDiff overhead in half.
* **Parallel exploration**: run each unexplored branch via `Threads.@spawn`.
* **Likelihood tracing**: evaluate sample likelihood along each curve and pick maxima on the fly.

---

## 7  Key references
1. Allgower, E. L. & Georg, K. (2003). *Introduction to Numerical Continuation Methods*.  
2. Bland, M. & Turocy, T. L. (2024). *Quantal Response Equilibrium as a Structural Model for Estimation*.  
3. McKelvey, R. D. & Palfrey, T. R. (1995). “Quantal Response Equilibria for Normal Form Games.” *Games and Economic Behavior*.  
4. Champagnat, R. *et al.* (2023). *BifurcationKit.jl: Numerical Continuation in Julia*.

---

## 8  Actionable task breakdown

| # | Deliverable | Description & acceptance criteria |
|---|-------------|-----------------------------------|
| **T1** | Core module `QRE3x3.jl` | Data structs, quantal‑response, analytic Jacobian. Unit tests on toy games pass. |
| **T2** | Continuation engine | Predictor–corrector with adaptive step; residual tol $10^{-10}$. |
| **T3** | Branch manager | Queue, bifurcation detection, full enumeration on random games. |
| **T4** | Benchmark harness | Runtime & memory profile; < 0.1 s per branch on M3 Max laptop. |
| **T5** | Documentation notebook | Reproduce λ–π plots; explanatory text. |
| **T6** | Integration hooks | API `all_branches(g; λmax=50)` returns vector of branches for estimation pipeline. |
