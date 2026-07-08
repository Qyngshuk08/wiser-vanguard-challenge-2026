# Multi-Asset Portfolio Construction — QUBO Formulation

*Working document — Vanguard x WISER Quantum+AI Challenge 2026*

## 1. Start from the classical problem

The classical Markowitz mean-variance objective is:

```
maximize   μᵀw − q · wᵀΣw
subject to Σ wᵢ = 1
           wᵢ ≥ 0  (long-only, optional)
           sector/liquidity/turnover constraints
```

Where:
- `w` = vector of portfolio weights (continuous, one per asset)
- `μ` = expected return vector
- `Σ` = covariance matrix (risk)
- `q` = risk-aversion coefficient (tunable "goal" knob: growth vs. drawdown control)

This is a convex quadratic program classically. The challenge is turning it into a
**QUBO** (Quadratic Unconstrained Binary Optimization) — the form QAOA/VQE and
quantum annealers actually consume: `minimize xᵀQx` where `x ∈ {0,1}ⁿ`.

## 2. Two ways to binarize — pick based on which sub-problem you're solving

### Option A: Asset Selection QUBO (simpler, good starting baseline)
Binary variable `xᵢ ∈ {0,1}`: "is asset i in the portfolio," equal-weighted or
weighted post-hoc among selected assets.

```
H = q · xᵀΣx − μᵀx + P·(Σxᵢ − B)²
```
- `B` = target number of holdings (budget/cardinality)
- `P` = penalty coefficient, large enough that constraint violations always cost
  more than any feasible objective improvement (rule of thumb: `P > max(|μᵢ|) + q·max(Σᵢᵢ)·B`)

This is the classic Qiskit-finance-tutorial style formulation. **Use this as your
Week 1 baseline** — it's fast to implement and lets you validate your whole pipeline
end-to-end before adding complexity.

### Option B: Weighted Allocation QUBO (matches the real problem statement better)
Encode each continuous weight `wᵢ` as a `K`-bit binary expansion:

```
wᵢ = (wmax / (2^K − 1)) · Σₖ 2^k · x_{i,k}
```

Substitute into the Markowitz objective. The `wᵀΣw` term expands into cross-terms
between every bit of every asset pair — this is where your QUBO matrix `Q` gets
dense. Budget constraint `Σwᵢ = 1` becomes a squared-penalty term over all bits.

**This is more faithful to "multi-asset allocation across equities/bonds/etc." but
costs you `n × K` qubits instead of `n`.** With n=10 assets and K=4 bits of
precision, that's 40 qubits — check what your simulator/hardware can realistically
handle before committing (this is literally the "scalability" judging criterion).

## 3. Adding the real-world constraints (this is where you differentiate)

| Constraint | Classical form | QUBO treatment |
|---|---|---|
| Sector exposure limit | `Σ_{i∈sector s} wᵢ ≤ Uₛ` | Add slack binary variables to convert `≤` into `=`, then penalize `(Σw − U − slack)²` |
| Turnover / transaction cost | `cost ∝ Σ\|wᵢ − wᵢ^old\|` | Absolute value isn't quadratic — linearize with auxiliary variables: `wᵢ − wᵢ^old ≤ d⁺ᵢ`, `wᵢ^old − wᵢ ≤ d⁻ᵢ`, cost = `Σ(d⁺ᵢ + d⁻ᵢ)`, then penalty-encode |
| Cardinality (max N holdings) | `Σ yᵢ ≤ N` | Same slack trick as sector limits; couple `yᵢ` to whether `wᵢ`'s bits are all zero |
| Liquidity floor | `wᵢ ≤ liquidity_i · cap` | Simple upper bound — restrict which bit-strings are reachable, or penalize violations directly |

**Practical advice:** don't try to cram every constraint in during Week 1. Get the
unconstrained + budget-only QUBO working and validated against a classical solver
first. Add one constraint at a time, re-validate each time. This is also literally
deliverable #4 in the brief ("Build baseline mean-variance optimizer, then add
constraints and scenario penalties").

## 4. Choosing penalty coefficients (the part everyone underestimates)

Penalty terms that are too small → solver finds "optimal" solutions that break
your constraints. Too large → the penalty term dominates and drowns out the actual
objective landscape, making the QAOA cost landscape hard to optimize (barren
plateau risk).

Standard approach: set each `P` slightly above the maximum possible objective
swing from violating that constraint by one unit, then sweep `P` empirically and
check constraint-satisfaction rate on your classical validator.

## 5. Next steps (in order)

1. Pick synthetic asset universe (suggest: 8–12 assets across equities/bonds/
   commodities/currencies, so it's small enough to simulate exactly with brute
   force for validation, but big enough to be interesting)
2. Generate synthetic μ, Σ (I can help you do this realistically — e.g. sampling
   from a factor model rather than pure random noise, so correlations look real)
3. Implement Option A (selection QUBO) end-to-end: formulation → classical
   brute-force validator → QAOA on simulator
4. Confirm QAOA output matches brute-force optimum on the small case
5. Scale up, add constraints one at a time, and only then compare against a
   classical benchmark (e.g. CPLEX/Gurobi or scipy) for the judging criteria

---
*Next document: Call Center Staffing QUBO formulation, reusing this same
penalty-method framework.*
