"""
Portfolio Construction — Baseline Selection QUBO (Option A from formulation doc)
Vanguard x WISER Quantum Challenge 2026

Pipeline:
  1. Load synthetic asset data (expected returns, covariance)
  2. Formulate QUBO: minimize q * x^T Sigma x - mu^T x   s.t.  sum(x) = B
  3. Solve classically (brute force + exact eigensolver) -> ground truth
  4. Solve with QAOA on a noiseless simulator
  5. Compare: objective value, selected assets, runtime
"""

import time
import numpy as np
import pandas as pd
from itertools import combinations

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
mu_df = pd.read_csv("data/expected_returns.csv")
cov_df = pd.read_csv("data/covariance_matrix.csv", index_col=0)

asset_ids = mu_df["asset_id"].tolist()
mu = mu_df["expected_return_annual"].values
sigma = cov_df.values
n = len(asset_ids)

# Tunable "goals" knobs (per brief deliverable #5)
RISK_AVERSION_Q = 0.5   # growth (low) <-> drawdown control (high)
BUDGET_B = 4            # target number of holdings (cardinality)
PENALTY_P = 2.0         # constraint-violation penalty; see formulation doc sec 4

print(f"Loaded {n} assets: {asset_ids}")
print(f"Risk aversion q={RISK_AVERSION_Q}, budget B={BUDGET_B}\n")

# ---------------------------------------------------------------------------
# 2. Formulate as a QuadraticProgram, then convert to QUBO
# ---------------------------------------------------------------------------
qp = QuadraticProgram(name="portfolio_selection")
for aid in asset_ids:
    qp.binary_var(name=aid)

# Objective: minimize q * x^T Sigma x - mu^T x
# NOTE: Qiskit's QuadraticProgram stores one coefficient per unordered pair
# (i,j). To correctly represent x^T Sigma x = sum_i Sigma_ii x_i + 2*sum_{i<j}
# Sigma_ij x_i x_j, off-diagonal terms must be doubled -- otherwise the QUBO
# silently encodes the wrong objective (verified via brute-force mismatch
# during development; always cross-check a hand-built QUBO like this).
linear = {aid: -mu[i] for i, aid in enumerate(asset_ids)}
quadratic = {}
for i in range(n):
    for j in range(n):
        if i == j:
            quadratic[(asset_ids[i], asset_ids[j])] = RISK_AVERSION_Q * sigma[i, j]
        elif i < j:
            quadratic[(asset_ids[i], asset_ids[j])] = 2 * RISK_AVERSION_Q * sigma[i, j]
qp.minimize(linear=linear, quadratic=quadratic)

# Constraint: sum(x_i) == B  (budget/cardinality)
qp.linear_constraint(
    linear={aid: 1 for aid in asset_ids}, sense="==", rhs=BUDGET_B, name="budget"
)

print(qp.prettyprint())

# Convert to unconstrained QUBO (penalty method, per formulation doc)
qubo_converter = QuadraticProgramToQubo(penalty=PENALTY_P)
qubo = qubo_converter.convert(qp)

# ---------------------------------------------------------------------------
# 3a. Ground truth via brute force (fully independent check)
# ---------------------------------------------------------------------------
def brute_force(mu, sigma, q, B):
    best_cost, best_combo = np.inf, None
    for combo in combinations(range(n), B):
        x = np.zeros(n)
        x[list(combo)] = 1
        cost = q * x @ sigma @ x - mu @ x
        if cost < best_cost:
            best_cost, best_combo = cost, combo
    return best_cost, best_combo


t0 = time.time()
bf_cost, bf_combo = brute_force(mu, sigma, RISK_AVERSION_Q, BUDGET_B)
bf_time = time.time() - t0
bf_assets = [asset_ids[i] for i in bf_combo]
print(f"\n[Brute force]  cost={bf_cost:.6f}  assets={bf_assets}  time={bf_time*1000:.2f}ms")

# ---------------------------------------------------------------------------
# 3b. Exact eigensolver via Qiskit (validates the QUBO conversion itself)
# ---------------------------------------------------------------------------
exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
t0 = time.time()
exact_result = exact_solver.solve(qp)
exact_time = time.time() - t0
exact_assets = [asset_ids[i] for i, v in enumerate(exact_result.x) if v > 0.5]
print(f"[Exact eigensolver]  cost={exact_result.fval:.6f}  assets={exact_assets}  time={exact_time*1000:.2f}ms")

# ---------------------------------------------------------------------------
# 4. QAOA on a noiseless statevector simulator
# ---------------------------------------------------------------------------
# IMPORTANT: without an explicit transpiler, QAOA's Pauli-evolution gates stay
# as opaque high-level instructions and the sampler falls back to a very slow
# general matrix-exponentiation path (observed: >90s hang on 12 qubits).
# Transpiling to a basic gate set first makes this a normal gate-level
# simulation (observed: ~0.05s per circuit eval on 12 qubits).
sampler = StatevectorSampler()
pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
qaoa_mes = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=300), reps=3, transpiler=pm)
qaoa_solver = MinimumEigenOptimizer(qaoa_mes)

t0 = time.time()
qaoa_result = qaoa_solver.solve(qp)
qaoa_time = time.time() - t0
qaoa_assets = [asset_ids[i] for i, v in enumerate(qaoa_result.x) if v > 0.5]
print(f"[QAOA simulator]     cost={qaoa_result.fval:.6f}  assets={qaoa_assets}  time={qaoa_time*1000:.2f}ms")

# ---------------------------------------------------------------------------
# 5. Comparison summary
# ---------------------------------------------------------------------------
print("\n=== VALIDATION SUMMARY ===")
print(f"Brute force optimum matches exact eigensolver: {set(bf_combo) == set(i for i,v in enumerate(exact_result.x) if v > 0.5)}")
print(f"QAOA found the true optimum: {set(qaoa_assets) == set(bf_assets)}")
print(f"QAOA cost gap vs optimum: {qaoa_result.fval - bf_cost:.6f}")
print(f"Constraint satisfied (selected == B): {len(qaoa_assets) == BUDGET_B}")

results_summary = pd.DataFrame({
    "method": ["brute_force", "exact_eigensolver", "qaoa_simulator"],
    "cost": [bf_cost, exact_result.fval, qaoa_result.fval],
    "assets_selected": [", ".join(bf_assets), ", ".join(exact_assets), ", ".join(qaoa_assets)],
    "runtime_ms": [bf_time*1000, exact_time*1000, qaoa_time*1000],
    "n_qubits": [n, n, n],
})
results_summary.to_csv("results/baseline_validation_results.csv", index=False)
print("\nSaved: baseline_validation_results.csv")
