"""
Staffing QUBO -- Multi-Start COBYLA
Vanguard x WISER Quantum Challenge 2026

Same technique as the Portfolio version: multiple independent random
restarts, keep the best. Compares against the single-run baseline (gap
300, ~14%) already reported in the deck.
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

agents_df = pd.read_csv("data/agents.csv")
shifts_df = pd.read_csv("data/shifts.csv")
demand_df = pd.read_csv("data/demand_forecast.csv")

agent_ids = agents_df["agent_id"].tolist()
shift_ids = shifts_df["shift_id"].tolist()
n_agents = len(agent_ids)
cost_per_hour = dict(zip(agents_df["agent_id"], agents_df["hourly_cost"]))
shift_hours = dict(zip(shifts_df["shift_id"], shifts_df["hours"]))

AGENT_THROUGHPUT_CALLS_PER_HOUR = 6
required_agents = {}
for _, s in shifts_df.iterrows():
    mask = (demand_df["interval_start_hour"] >= s["start_hour"]) & (demand_df["interval_start_hour"] < s["end_hour"])
    total_calls = demand_df.loc[mask, "forecast_calls"].sum()
    intervals = mask.sum()
    calls_per_hour = total_calls / (intervals * 0.5)
    required_agents[s["shift_id"]] = max(1, int(np.ceil(calls_per_hour / AGENT_THROUGHPUT_CALLS_PER_HOUR)))

COVERAGE_PENALTY = 500.0
N_STARTS = 8

def vname(a, s):
    return f"{a}_{s}"

var_names = [vname(a, s) for a in agent_ids for s in shift_ids]
n = len(var_names)

qp = QuadraticProgram(name="staffing")
for v in var_names:
    qp.binary_var(name=v)
linear = {vname(a, s): cost_per_hour[a] * shift_hours[s] for a in agent_ids for s in shift_ids}
quadratic = {}
for s in shift_ids:
    req = required_agents[s]
    vars_s = [vname(a, s) for a in agent_ids]
    for v in vars_s:
        linear[v] = linear.get(v, 0) + COVERAGE_PENALTY * (1 - 2 * req)
    for i in range(len(vars_s)):
        for j in range(i + 1, len(vars_s)):
            key = (vars_s[i], vars_s[j])
            quadratic[key] = quadratic.get(key, 0) + 2 * COVERAGE_PENALTY
qp.minimize(linear=linear, quadratic=quadratic)
for a in agent_ids:
    qp.linear_constraint(linear={vname(a, s): 1 for s in shift_ids}, sense="<=", rhs=1, name=f"one_shift_{a}")

exact_result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
true_optimum = exact_result.fval
print(f"True optimum: {true_optimum:.0f}\n")

sampler = StatevectorSampler()
pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])

results = []
t0_total = time.time()
for start in range(N_STARTS):
    rng = np.random.default_rng(start)
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=200), reps=3, transpiler=pm,
                initial_point=rng.uniform(0, np.pi, 6))
    solver = MinimumEigenOptimizer(qaoa)
    t0 = time.time()
    result = solver.solve(qp)
    elapsed = time.time() - t0
    assignment = {v.split("_")[0]: v.split("_")[1] for v, val in zip(var_names, result.x) if val > 0.5}
    gap = result.fval - true_optimum
    results.append({"start": start, "cost": result.fval, "gap": gap, "time_s": elapsed,
                     "n_assigned": len(assignment)})
    print(f"Start {start}: cost={result.fval:.0f}  gap={gap:.0f}  agents_deployed={len(assignment)}")

total_time = time.time() - t0_total

print(f"\n=== SUMMARY ===")
print(f"Total time for {N_STARTS} starts: {total_time:.1f}s")
best = min(results, key=lambda r: r["cost"])
print(f"\nBest result (multi-start): cost={best['cost']:.0f}  gap={best['gap']:.0f}")
print(f"Single-run baseline (from deck): cost=-1884  gap=300 (~14%)")

if best["gap"] < 300:
    improvement = (300 - best["gap"]) / 300 * 100
    print(f"\nMulti-start IMPROVED the gap by {improvement:.0f}% relative to the single-run baseline.")
else:
    print(f"\nMulti-start did NOT improve on the single-run baseline in this run -- report honestly.")

pd.DataFrame(results).to_csv("multistart_staffing_results.csv", index=False)
print("\nSaved: multistart_staffing_results.csv")
