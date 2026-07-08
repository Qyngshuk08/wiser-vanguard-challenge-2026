"""
Call Center Staffing — Constrained QUBO (skill-specific coverage)
Extends the validated 08_staffing_qubo_baseline.py: replaces aggregate
per-shift headcount coverage with SKILL-specific coverage (Sales vs Support),
matching the brief's "agents across shifts, skills, and communication
channels" requirement more faithfully.

Break rules: NOT modeled as QUBO variables. Each shift definition already
implies a fixed unpaid break (documented below), following the formulation
doc's guidance to keep fixed structural rules out of the qubit count rather
than encoding every operational detail as a variable.
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
demand_df = pd.read_csv("data/demand_forecast.csv")

agent_ids = agents_df["agent_id"].tolist()
shift_ids = shifts_df["shift_id"].tolist()
n_agents = len(agent_ids)
n_shifts = len(shift_ids)
cost_per_hour = dict(zip(agents_df["agent_id"], agents_df["hourly_cost"]))
shift_hours = dict(zip(shifts_df["shift_id"], shifts_df["hours"]))
agent_skills = {row["agent_id"]: set(row["skills"].split(",")) for _, row in agents_df.iterrows()}

# Break rule (structural, not a QUBO variable): every 8-hour shift includes
# one mandatory unpaid 30-minute break, scheduled at the shift's midpoint by
# policy. This is a fixed operational rule, not a decision the optimizer
# needs to make -- modeling it as a variable would add qubits for zero
# decision value. Documented here so it's auditable, per the brief's
# transparency requirement.
BREAK_RULE = "30-min unpaid break at shift midpoint, fixed by policy (not optimized)"

AGENT_THROUGHPUT_CALLS_PER_HOUR = 6
SKILLS = ["Sales", "Support"]

# Required agents PER SKILL per shift (replaces the old aggregate coverage)
required_by_skill = {}
for _, s in shifts_df.iterrows():
    mask = (demand_df["interval_start_hour"] >= s["start_hour"]) & (demand_df["interval_start_hour"] < s["end_hour"])
    intervals_in_shift = mask.sum()
    for skill in SKILLS:
        calls = demand_df.loc[mask & (demand_df["queue"] == skill), "forecast_calls"].sum()
        calls_per_hour = calls / (intervals_in_shift * 0.5)
        required_by_skill[(s["shift_id"], skill)] = max(0, int(np.ceil(calls_per_hour / AGENT_THROUGHPUT_CALLS_PER_HOUR)))

print("Required agents per shift PER SKILL:")
for (s, k), r in required_by_skill.items():
    print(f"  {s} / {k}: {r} agents needed")

COVERAGE_PENALTY = 500.0


def vname(a, s):
    return f"{a}_{s}"


var_names = [vname(a, s) for a in agent_ids for s in shift_ids]
n = len(var_names)

qp = QuadraticProgram(name="staffing_skilled")
for v in var_names:
    qp.binary_var(name=v)

linear = {vname(a, s): cost_per_hour[a] * shift_hours[s] for a in agent_ids for s in shift_ids}
quadratic = {}

# Skill-specific coverage penalty: for each (shift, skill), penalize the gap
# between required and actual skilled coverage. An agent contributes to a
# skill's coverage count only if they have that skill.
for s in shift_ids:
    for skill in SKILLS:
        req = required_by_skill[(s, skill)]
        skilled_agents = [a for a in agent_ids if skill in agent_skills[a]]
        vars_sk = [vname(a, s) for a in skilled_agents]
        if not vars_sk:
            continue
        for v in vars_sk:
            linear[v] = linear.get(v, 0) + COVERAGE_PENALTY * (1 - 2 * req)
        for i in range(len(vars_sk)):
            for j in range(i + 1, len(vars_sk)):
                key = (vars_sk[i], vars_sk[j])
                quadratic[key] = quadratic.get(key, 0) + 2 * COVERAGE_PENALTY

qp.minimize(linear=linear, quadratic=quadratic)
for a in agent_ids:
    qp.linear_constraint(linear={vname(a, s): 1 for s in shift_ids}, sense="<=", rhs=1, name=f"one_shift_{a}")

print(f"\nProblem size: {n} qubits ({n_agents} agents x {n_shifts} shifts)")
print(f"Break rule (structural, not modeled as a variable): {BREAK_RULE}")


def brute_force():
    options = shift_ids + [None]
    best_cost, best_assignment = np.inf, None
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


def coverage_report(assignment):
    lines = []
    for s in shift_ids:
        for skill in SKILLS:
            covered = sum(1 for a, sh in assignment.items() if sh == s and skill in agent_skills[a])
            req = required_by_skill[(s, skill)]
            lines.append(f"    {s}/{skill}: covered={covered} required={req} {'OK' if covered>=req else 'SHORT'}")
    return "\n".join(lines)


t0 = time.time()
bf_cost, bf_assignment = brute_force()
bf_time = time.time() - t0
print(f"\n[Brute force] cost={bf_cost:.0f}  assignment={bf_assignment}  time={bf_time*1000:.1f}ms")
print(coverage_report(bf_assignment))

exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
t0 = time.time()
exact_result = exact_solver.solve(qp)
exact_time = time.time() - t0
exact_assignment = {v.split("_")[0]: v.split("_")[1] for v, val in zip(var_names, exact_result.x) if val > 0.5}
print(f"\n[Exact eigensolver] cost={exact_result.fval:.0f}  assignment={exact_assignment}  time={exact_time*1000:.1f}ms")
print(f"MATCH with brute force: {abs(exact_result.fval - bf_cost) < 1e-6}")

sampler = StatevectorSampler()
pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
qaoa_mes = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=300), reps=3, transpiler=pm)
qaoa_solver = MinimumEigenOptimizer(qaoa_mes)

t0 = time.time()
qaoa_result = qaoa_solver.solve(qp)
qaoa_time = time.time() - t0
qaoa_assignment = {v.split("_")[0]: v.split("_")[1] for v, val in zip(var_names, qaoa_result.x) if val > 0.5}
print(f"\n[QAOA] cost={qaoa_result.fval:.0f}  assignment={qaoa_assignment}  time={qaoa_time:.2f}s")
print(coverage_report(qaoa_assignment))
print(f"Gap vs optimum: {qaoa_result.fval - bf_cost:.0f}")

pd.DataFrame({
    "method": ["brute_force", "exact_eigensolver", "qaoa"],
    "cost": [bf_cost, exact_result.fval, qaoa_result.fval],
    "runtime_s": [bf_time, exact_time, qaoa_time],
}).to_csv("results/staffing_skilled_results.csv", index=False)
print("\nSaved: staffing_skilled_results.csv")
