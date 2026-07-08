"""
Call Center Staffing — Baseline QUBO (4 agents x 3 shifts = 12 qubits)
Vanguard x WISER Quantum Challenge 2026

Reuses the exact validation pattern from the Portfolio track:
  1. Build required-staffing-per-interval from the demand forecast
  2. Formulate QUBO: minimize labor cost + coverage-gap penalty
     s.t. each agent works at most one shift
  3. Validate: brute force -> exact eigensolver -> QAOA (simulator, with the
     transpiler fix already known from the Portfolio debugging session)
"""

import time
import numpy as np
import pandas as pd
from itertools import product

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ---------------------------------------------------------------------------
# 1. Load data, reduce to required coverage per shift (simplification for the
#    12-qubit baseline: treat each shift as covering a fixed block, and reduce
#    demand to "agents needed" per shift by summing forecasted calls handled
#    per shift window / an assumed agent throughput).
# ---------------------------------------------------------------------------
agents_df = pd.read_csv("data/agents.csv")
shifts_df = pd.read_csv("data/shifts.csv")
demand_df = pd.read_csv("data/demand_forecast.csv")

agent_ids = agents_df["agent_id"].tolist()
shift_ids = shifts_df["shift_id"].tolist()
n_agents = len(agent_ids)
n_shifts = len(shift_ids)

AGENT_THROUGHPUT_CALLS_PER_HOUR = 6  # assumed calls one agent can handle per hour

# Required agent-count per shift = total forecasted calls falling in that
# shift's window / throughput / shift hours (simple coverage estimate)
required_agents = {}
for _, s in shifts_df.iterrows():
    mask = (demand_df["interval_start_hour"] >= s["start_hour"]) & (demand_df["interval_start_hour"] < s["end_hour"])
    total_calls_in_shift = demand_df.loc[mask, "forecast_calls"].sum()
    # total_calls_in_shift is calls per 30-min interval summed -> convert to hourly load
    intervals_in_shift = mask.sum()
    calls_per_hour_in_shift = total_calls_in_shift / (intervals_in_shift * 0.5)
    required_agents[s["shift_id"]] = max(1, int(np.ceil(calls_per_hour_in_shift / AGENT_THROUGHPUT_CALLS_PER_HOUR)))

print("Required agents per shift (derived from demand forecast):")
for s, r in required_agents.items():
    print(f"  {s}: {r} agents needed")

# ---------------------------------------------------------------------------
# 2. Formulate QUBO
#    x_{a,s} = 1 if agent a works shift s
#    Objective: minimize labor cost + coverage-gap penalty per shift
#    Constraint: each agent works at most one shift (sum_s x_{a,s} <= 1)
# ---------------------------------------------------------------------------
var_names = [f"{a}_{s}" for a in agent_ids for s in shift_ids]
n = len(var_names)  # 4 agents x 3 shifts = 12 qubits

cost_per_hour = dict(zip(agents_df["agent_id"], agents_df["hourly_cost"]))
shift_hours = dict(zip(shifts_df["shift_id"], shifts_df["hours"]))

COVERAGE_PENALTY = 500.0  # must dominate labor cost (~200-250/agent-shift) or the
                           # QUBO's true optimum becomes "hire nobody" -- caught by
                           # the empty-assignment result during validation, same
                           # penalty-tuning lesson as the Portfolio formulation doc.
MAX_ONE_SHIFT_PENALTY = 20.0

qp = QuadraticProgram(name="staffing")
for v in var_names:
    qp.binary_var(name=v)


def vname(a, s):
    return f"{a}_{s}"


# --- Linear cost term: labor cost of assigning agent a to shift s ---
linear = {}
for a in agent_ids:
    for s in shift_ids:
        linear[vname(a, s)] = cost_per_hour[a] * shift_hours[s]

# --- Coverage-gap penalty per shift: COVERAGE_PENALTY * (required_s - sum_a x_{a,s})^2 ---
# Expand the square into linear + quadratic terms added to the objective.
quadratic = {}
for s in shift_ids:
    req = required_agents[s]
    vars_s = [vname(a, s) for a in agent_ids]
    # (req - sum x)^2 = req^2 - 2*req*sum(x) + sum(x)^2
    # sum(x)^2 = sum(x_i) + 2*sum_{i<j} x_i x_j   (since x_i^2 = x_i for binary)
    for v in vars_s:
        linear[v] = linear.get(v, 0) + COVERAGE_PENALTY * (1 - 2 * req)
    for i in range(len(vars_s)):
        for j in range(i + 1, len(vars_s)):
            key = (vars_s[i], vars_s[j])
            quadratic[key] = quadratic.get(key, 0) + 2 * COVERAGE_PENALTY

qp.minimize(linear=linear, quadratic=quadratic)

# --- Constraint: each agent works at most one shift ---
for a in agent_ids:
    qp.linear_constraint(
        linear={vname(a, s): 1 for s in shift_ids}, sense="<=", rhs=1, name=f"one_shift_{a}"
    )

print(f"\nProblem size: {n} qubits ({n_agents} agents x {n_shifts} shifts)")

# ---------------------------------------------------------------------------
# 3a. Brute force ground truth (independent check)
# ---------------------------------------------------------------------------
def cost_of_assignment(assignment):
    """assignment: dict agent_id -> shift_id or None"""
    x = {v: 0 for v in var_names}
    for a, s in assignment.items():
        if s is not None:
            x[vname(a, s)] = 1
    xv = np.array([x[v] for v in var_names])
    return qp.objective.evaluate(xv), xv


def brute_force():
    best_cost, best_assignment = np.inf, None
    options = shift_ids + [None]  # None = not working
    for combo in product(options, repeat=n_agents):
        assignment = dict(zip(agent_ids, combo))
        cost, _ = cost_of_assignment(assignment)
        if cost < best_cost:
            best_cost, best_assignment = cost, assignment
    return best_cost, best_assignment


t0 = time.time()
bf_cost, bf_assignment = brute_force()
bf_time = time.time() - t0
print(f"\n[Brute force] cost={bf_cost:.2f}  assignment={bf_assignment}  time={bf_time*1000:.1f}ms")

# ---------------------------------------------------------------------------
# 3b. Exact eigensolver (validates the QUBO conversion)
# ---------------------------------------------------------------------------
exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
t0 = time.time()
exact_result = exact_solver.solve(qp)
exact_time = time.time() - t0
exact_assignment = {
    v.split("_")[0]: v.split("_")[1]
    for v, val in zip(var_names, exact_result.x) if val > 0.5
}
print(f"[Exact eigensolver] cost={exact_result.fval:.2f}  assignment={exact_assignment}  time={exact_time*1000:.1f}ms")
print(f"MATCH with brute force: {abs(exact_result.fval - bf_cost) < 1e-6}")

# ---------------------------------------------------------------------------
# 4. QAOA on simulator (transpiler fix from Portfolio track applied directly)
# ---------------------------------------------------------------------------
sampler = StatevectorSampler()
pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
qaoa_mes = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=300), reps=3, transpiler=pm)
qaoa_solver = MinimumEigenOptimizer(qaoa_mes)

t0 = time.time()
qaoa_result = qaoa_solver.solve(qp)
qaoa_time = time.time() - t0
qaoa_assignment = {
    v.split("_")[0]: v.split("_")[1]
    for v, val in zip(var_names, qaoa_result.x) if val > 0.5
}
print(f"[QAOA simulator] cost={qaoa_result.fval:.2f}  assignment={qaoa_assignment}  time={qaoa_time*1000:.1f}ms")
print(f"Cost gap vs optimum: {qaoa_result.fval - bf_cost:.2f}")

# ---------------------------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------------------------
pd.DataFrame({
    "method": ["brute_force", "exact_eigensolver", "qaoa_simulator"],
    "cost": [bf_cost, exact_result.fval, qaoa_result.fval],
    "runtime_ms": [bf_time*1000, exact_time*1000, qaoa_time*1000],
    "n_qubits": [n, n, n],
}).to_csv("results/staffing_baseline_validation_results.csv", index=False)
print("\nSaved: staffing_baseline_validation_results.csv")
