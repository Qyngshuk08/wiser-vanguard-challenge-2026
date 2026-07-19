"""
Portfolio QUBO -- Multi-Start COBYLA
Vanguard x WISER Quantum Challenge 2026

Runs QAOA training multiple times with independent random initial points,
keeps the best result. Cheap improvement flagged early in this project but
never built -- worth doing now with remaining time before Aug 7.

Compares directly against the single-run baseline (gap ~3.6%, cost -0.1738)
already reported in the deck.
"""
import time
import numpy as np
import pandas as pd

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

mu_df = pd.read_csv("data/expected_returns.csv")
cov_df = pd.read_csv("data/covariance_matrix.csv", index_col=0)
asset_ids = mu_df["asset_id"].tolist()
mu = mu_df["expected_return_annual"].values
sigma = cov_df.values
n = len(asset_ids)

RISK_AVERSION_Q = 0.5
BUDGET_B = 4
N_STARTS = 8  # independent random restarts

qp = QuadraticProgram(name="portfolio_selection")
for aid in asset_ids:
    qp.binary_var(name=aid)
linear = {aid: -mu[i] for i, aid in enumerate(asset_ids)}
quadratic = {}
for i in range(n):
    for j in range(n):
        if i == j:
            quadratic[(asset_ids[i], asset_ids[j])] = RISK_AVERSION_Q * sigma[i, j]
        elif i < j:
            quadratic[(asset_ids[i], asset_ids[j])] = 2 * RISK_AVERSION_Q * sigma[i, j]
qp.minimize(linear=linear, quadratic=quadratic)
qp.linear_constraint(linear={aid: 1 for aid in asset_ids}, sense="==", rhs=BUDGET_B, name="budget")

# Ground truth for comparison
exact_result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
true_optimum = exact_result.fval
print(f"True optimum: {true_optimum:.4f}\n")

sampler = StatevectorSampler()
pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])

results = []
t0_total = time.time()
for start in range(N_STARTS):
    rng = np.random.default_rng(start)  # different seed per start
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=200), reps=3, transpiler=pm,
                initial_point=rng.uniform(0, np.pi, 6))
    solver = MinimumEigenOptimizer(qaoa)
    t0 = time.time()
    result = solver.solve(qp)
    elapsed = time.time() - t0
    selected = [asset_ids[i] for i, v in enumerate(result.x) if v > 0.5]
    feasible = len(selected) == BUDGET_B
    gap = (result.fval - true_optimum) if feasible else None
    results.append({"start": start, "cost": result.fval, "feasible": feasible,
                     "gap": gap, "time_s": elapsed, "assets": selected})
    print(f"Start {start}: cost={result.fval:.4f}  feasible={feasible}  "
          f"gap={gap:.4f}" if gap is not None else f"Start {start}: cost={result.fval:.4f}  feasible={feasible}  INFEASIBLE")

total_time = time.time() - t0_total

feasible_results = [r for r in results if r["feasible"]]
print(f"\n=== SUMMARY ===")
print(f"Total time for {N_STARTS} starts: {total_time:.1f}s")
print(f"Feasible results: {len(feasible_results)}/{N_STARTS}")

if feasible_results:
    best = min(feasible_results, key=lambda r: r["cost"])
    print(f"\nBest result (multi-start): cost={best['cost']:.4f}  gap={best['gap']:.4f} "
          f"({best['gap']/abs(true_optimum)*100:.1f}%)  assets={best['assets']}")
    print(f"Single-run baseline (from deck): cost=-0.1738  gap=0.0065 (3.6%)")

    if best["gap"] < 0.0065:
        improvement = (0.0065 - best["gap"]) / 0.0065 * 100
        print(f"\nMulti-start IMPROVED the gap by {improvement:.0f}% relative to the single-run baseline.")
    else:
        print(f"\nMulti-start did NOT improve on the single-run baseline in this run -- report honestly.")

pd.DataFrame(results).to_csv("multistart_results.csv", index=False)
print("\nSaved: multistart_results.csv")
