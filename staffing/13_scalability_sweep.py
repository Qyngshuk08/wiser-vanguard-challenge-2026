"""
Scalability Sweep — Call Center Staffing QUBO
Same methodology as the Portfolio scalability sweep: push agent x shift count
up until brute force / exact eigensolver / QAOA break down.
"""
import time
import signal
import math
import numpy as np
import pandas as pd
from itertools import product

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

COVERAGE_PENALTY = 500.0
TIME_BUDGET_S = 45
EXACT_QUBIT_LIMIT = 20


class TimeoutException(Exception):
    pass


def with_timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutException()
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


def generate_staffing_problem(n_agents, n_shifts, seed=7):
    r = np.random.default_rng(seed)
    agent_ids = [f"AG{i+1}" for i in range(n_agents)]
    shift_ids = [f"S{i+1}" for i in range(n_shifts)]
    cost_per_hour = {a: r.integers(22, 32) for a in agent_ids}
    shift_hours = {s: 8 for s in shift_ids}
    # required coverage per shift -- scale demand with problem size, keep it
    # a genuine capacity squeeze like the original baseline (slightly more
    # required agent-slots than agents available, forcing real trade-offs)
    total_capacity = n_agents
    required = {}
    remaining = int(np.ceil(total_capacity * 1.15))  # ~15% over-capacity demand
    for i, s in enumerate(shift_ids):
        share = max(1, remaining // (n_shifts - i))
        required[s] = share
        remaining -= share
    return agent_ids, shift_ids, cost_per_hour, shift_hours, required


def vname(a, s):
    return f"{a}_{s}"


def build_qp(agent_ids, shift_ids, cost_per_hour, shift_hours, required):
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
        req = required[s]
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


@with_timeout(TIME_BUDGET_S)
def run_brute_force(agent_ids, shift_ids, qp, var_names):
    n_agents = len(agent_ids)
    options = shift_ids + [None]
    best_cost = np.inf
    for combo in product(options, repeat=n_agents):
        x = {v: 0 for v in var_names}
        for a, s in zip(agent_ids, combo):
            if s is not None:
                x[vname(a, s)] = 1
        xv = np.array([x[v] for v in var_names])
        cost = qp.objective.evaluate(xv)
        if cost < best_cost:
            best_cost = cost
    return best_cost


@with_timeout(TIME_BUDGET_S)
def run_exact(qp):
    solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    return solver.solve(qp)


@with_timeout(TIME_BUDGET_S)
def run_qaoa(qp):
    sampler = StatevectorSampler()
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
    qaoa_mes = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=200), reps=3, transpiler=pm)
    solver = MinimumEigenOptimizer(qaoa_mes)
    return solver.solve(qp)


results = []
# n_agents x n_shifts = qubits. Staffing grows faster than Portfolio's
# selection encoding (same n_agents but more shifts multiplies qubits), so
# expect the wall to arrive at a SMALLER total problem "size" in real terms.
configs = [(4, 3), (5, 4), (6, 4), (7, 5), (8, 5)]

for n_agents, n_shifts in configs:
    n_qubits = n_agents * n_shifts
    print(f"\n{'='*50}\n{n_agents} agents x {n_shifts} shifts = {n_qubits} qubits\n{'='*50}")
    agent_ids, shift_ids, cost_per_hour, shift_hours, required = generate_staffing_problem(n_agents, n_shifts)
    qp, var_names = build_qp(agent_ids, shift_ids, cost_per_hour, shift_hours, required)
    row = {"n_agents": n_agents, "n_shifts": n_shifts, "n_qubits": n_qubits}

    # Brute force -- (n_shifts+1)^n_agents grows fast; cap search space
    search_space = (n_shifts + 1) ** n_agents
    if search_space > 2_000_000:
        row["brute_force_time_s"] = None
        row["brute_force_cost"] = None
        print(f"  Brute force: SKIPPED -- search space {search_space:,} too large")
    else:
        try:
            t0 = time.time()
            bf_cost = run_brute_force(agent_ids, shift_ids, qp, var_names)
            row["brute_force_time_s"] = time.time() - t0
            row["brute_force_cost"] = bf_cost
            print(f"  Brute force: {bf_cost:.0f} in {row['brute_force_time_s']:.2f}s (search space {search_space:,})")
        except TimeoutException:
            row["brute_force_time_s"] = None
            row["brute_force_cost"] = None
            print(f"  Brute force: TIMED OUT (>{TIME_BUDGET_S}s)")

    # Exact eigensolver
    if n_qubits > EXACT_QUBIT_LIMIT:
        row["exact_time_s"] = None
        row["exact_cost"] = None
        print(f"  Exact eigensolver: SKIPPED -- {n_qubits} qubits exceeds dense-diagonalization memory limit")
    else:
        try:
            t0 = time.time()
            exact_result = run_exact(qp)
            row["exact_time_s"] = time.time() - t0
            row["exact_cost"] = exact_result.fval
            print(f"  Exact eigensolver: {exact_result.fval:.0f} in {row['exact_time_s']:.2f}s")
        except TimeoutException:
            row["exact_time_s"] = None
            row["exact_cost"] = None
            print(f"  Exact eigensolver: TIMED OUT (>{TIME_BUDGET_S}s)")

    # QAOA
    try:
        t0 = time.time()
        qaoa_result = run_qaoa(qp)
        row["qaoa_time_s"] = time.time() - t0
        row["qaoa_cost"] = qaoa_result.fval
        ref = row.get("exact_cost") if row.get("exact_cost") is not None else row.get("brute_force_cost")
        gap = qaoa_result.fval - ref if ref is not None else None
        row["qaoa_gap"] = gap
        print(f"  QAOA: {qaoa_result.fval:.0f} in {row['qaoa_time_s']:.2f}s"
              + (f"  gap={gap:.0f}" if gap is not None else "  (no reference)"))
    except TimeoutException:
        row["qaoa_time_s"] = None
        row["qaoa_cost"] = None
        row["qaoa_gap"] = None
        print(f"  QAOA: TIMED OUT (>{TIME_BUDGET_S}s)")

    results.append(row)

    if row.get("qaoa_time_s") is None:
        print(f"\nQAOA broke down at {n_qubits} qubits ({n_agents}x{n_shifts}). Stopping sweep.")
        break

df = pd.DataFrame(results)
df.to_csv("results/staffing_scalability_sweep_results.csv", index=False)
print("\nSaved: staffing_scalability_sweep_results.csv")
print(df.to_string(index=False))
