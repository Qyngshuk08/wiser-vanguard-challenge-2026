"""
Disruption Replanning Demo — Call Center Staffing
Vanguard x WISER Quantum Challenge 2026

Demonstrates the stretch-goal differentiator: when call volume spikes
mid-day, how fast can we re-optimize staffing vs. a full cold re-solve?

Scenario: S2 (10am-6pm) sees a 40% volume spike (e.g. unplanned product
issue driving Support calls). Re-derive required_agents, re-solve the QUBO,
and compare against the original static plan.
"""

import time
import numpy as np
import pandas as pd
from itertools import product

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

agents_df = pd.read_csv("data/agents.csv")
shifts_df = pd.read_csv("data/shifts.csv")

agent_ids = agents_df["agent_id"].tolist()
shift_ids = shifts_df["shift_id"].tolist()
n_agents = len(agent_ids)
cost_per_hour = dict(zip(agents_df["agent_id"], agents_df["hourly_cost"]))
shift_hours = dict(zip(shifts_df["shift_id"], shifts_df["hours"]))

COVERAGE_PENALTY = 500.0


def vname(a, s):
    return f"{a}_{s}"


def build_qp(required_agents):
    var_names = [vname(a, s) for a in agent_ids for s in shift_ids]
    qp = QuadraticProgram(name="staffing")
    for v in var_names:
        qp.binary_var(name=v)

    linear = {}
    for a in agent_ids:
        for s in shift_ids:
            linear[vname(a, s)] = cost_per_hour[a] * shift_hours[s]

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
    return qp, var_names


def brute_force(qp, var_names):
    best_cost, best_assignment = np.inf, None
    options = shift_ids + [None]
    for combo in product(options, repeat=n_agents):
        x = {v: 0 for v in var_names}
        for a, s in zip(agent_ids, combo):
            if s is not None:
                x[vname(a, s)] = 1
        xv = np.array([x[v] for v in var_names])
        cost = qp.objective.evaluate(xv)
        if cost < best_cost:
            best_cost, best_assignment = cost, dict(zip(agent_ids, combo))
    return best_cost, best_assignment


def solve_qaoa(qp, warm_start_point=None):
    sampler = StatevectorSampler()
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
    kwargs = dict(sampler=sampler, optimizer=COBYLA(maxiter=300), reps=3, transpiler=pm)
    if warm_start_point is not None:
        kwargs["initial_point"] = warm_start_point
    qaoa_mes = QAOA(**kwargs)
    solver = MinimumEigenOptimizer(qaoa_mes)
    t0 = time.time()
    result = solver.solve(qp)
    elapsed = time.time() - t0
    return result, elapsed


# ---------------------------------------------------------------------------
# Scenario A: original static demand (from staffing_qubo_baseline.py run)
# ---------------------------------------------------------------------------
required_original = {"S1": 2, "S2": 2, "S3": 1}
qp_original, var_names = build_qp(required_original)

print("=== BASELINE (static plan, pre-disruption) ===")
t0 = time.time()
bf_cost_orig, bf_assignment_orig = brute_force(qp_original, var_names)
print(f"Brute-force optimum: {bf_cost_orig:.0f}  {bf_assignment_orig}  ({(time.time()-t0)*1000:.1f}ms)")

qaoa_result_orig, qaoa_time_orig = solve_qaoa(qp_original)
print(f"QAOA (cold start): {qaoa_result_orig.fval:.0f}  time={qaoa_time_orig:.2f}s")

# ---------------------------------------------------------------------------
# Scenario B: disruption -- S2 (10am-6pm) sees a 40% volume spike
# ---------------------------------------------------------------------------
required_disrupted = {"S1": 2, "S2": 3, "S3": 1}  # S2 requirement jumps 2->3
qp_disrupted, _ = build_qp(required_disrupted)

print("\n=== DISRUPTION: S2 demand +40% (2 -> 3 agents required) ===")
t0 = time.time()
bf_cost_dis, bf_assignment_dis = brute_force(qp_disrupted, var_names)
print(f"Brute-force optimum: {bf_cost_dis:.0f}  {bf_assignment_dis}  ({(time.time()-t0)*1000:.1f}ms)")

# --- Cold re-solve: QAOA from scratch, no knowledge of the prior solution ---
qaoa_result_cold, qaoa_time_cold = solve_qaoa(qp_disrupted)
print(f"\n[Cold re-solve]  QAOA cost={qaoa_result_cold.fval:.0f}  time={qaoa_time_cold:.2f}s")

# --- Warm-started re-solve: reuse the optimized QAOA angles from the
#     original (pre-disruption) solve as the initial point. This is the
#     realistic "manager hits re-optimize after a disruption" scenario --
#     the circuit *structure* (problem size, shift/agent count) is unchanged,
#     only the coverage requirement shifted, so the previous angles are a
#     reasonable warm start rather than random initialization. ---
prev_optimal_point = qaoa_result_orig.min_eigen_solver_result.optimal_point
qaoa_result_warm, qaoa_time_warm = solve_qaoa(qp_disrupted, warm_start_point=prev_optimal_point)
print(f"[Warm-started re-solve]  QAOA cost={qaoa_result_warm.fval:.0f}  time={qaoa_time_warm:.2f}s")

print("\n=== SUMMARY ===")
print(f"Cold re-solve time:  {qaoa_time_cold:.2f}s   (gap vs optimum: {qaoa_result_cold.fval - bf_cost_dis:.0f})")
print(f"Warm re-solve time:  {qaoa_time_warm:.2f}s   (gap vs optimum: {qaoa_result_warm.fval - bf_cost_dis:.0f})")
if qaoa_time_warm < qaoa_time_cold:
    speedup = (qaoa_time_cold - qaoa_time_warm) / qaoa_time_cold * 100
    print(f"Warm start reduced re-solve time by {speedup:.1f}%")
else:
    print("Warm start did not reduce time in this run -- report honestly, "
          "COBYLA's iteration count is fixed regardless of starting point; "
          "the benefit (if any) would show as faster CONVERGENCE per iteration, "
          "not necessarily fewer iterations under a fixed maxiter budget.")

results = pd.DataFrame({
    "scenario": ["baseline_qaoa", "disruption_cold", "disruption_warm"],
    "cost": [qaoa_result_orig.fval, qaoa_result_cold.fval, qaoa_result_warm.fval],
    "time_s": [qaoa_time_orig, qaoa_time_cold, qaoa_time_warm],
    "true_optimum": [bf_cost_orig, bf_cost_dis, bf_cost_dis],
})
results.to_csv("results/disruption_replanning_results.csv", index=False)
print("\nSaved: disruption_replanning_results.csv")
